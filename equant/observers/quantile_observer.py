import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver


__all__ = [
    'QuantileObserver',
    'QuantilePerChannelObserver'
]


class QuantileObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    min and max of quantile values.

    This observer computes the quantization parameters based on the minimums 
    and maximums of quantile of the incoming tensors. The module
    records the minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        quantile: Quantile for statistics.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(
        self,
        quantile=0.99999,
        reduction='mean',
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )
        if reduction not in ['mean', 'max']:
            raise NotImplementedError(f'Invalid reduction method. Expected one of ["mean", "max"], but got {reduction}')

        self.quantile = quantile
        self.reduction = reduction
        self.total_elements = 0

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))
        self.total_elements = 0

    def forward(self, x_orig):

        if x_orig.numel() == 0:
            return x_orig
        
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)

        y = x.flatten(start_dim=1)
        min_val = torch.quantile(y, dim=1, q=1 - self.quantile)
        max_val = torch.quantile(y, dim=1, q=self.quantile)

        b = len(min_val)
        if self.reduction == 'max':
            min_val = torch.min(min_val.min(), self.min_val)
            max_val = torch.max(max_val.max(), self.max_val)
        elif self.reduction == 'mean':
            if self.total_elements > 0:
                alpha = self.total_elements / (self.total_elements + b)
                beta = b / (self.total_elements + b)
                min_val = alpha * self.min_val + beta * min_val.mean()
                max_val = alpha * self.max_val + beta * max_val.mean()
            else:
                min_val = min_val.mean()
                max_val = max_val.mean()

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        self.total_elements += b
        
        return x_orig
    

class QuantilePerChannelObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel quantile values.

    This observer uses the tensor quantile statistics to compute the per channel
    quantization parameters. The module records the minimum and maximum for 
    quantile of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        quantile: Quantile for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(
        self,
        quantile=0.99999,
        reduction='mean',
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )
        if reduction not in ['mean', 'max']:
            raise NotImplementedError(f'Invalid reduction method. Expected one of ["mean", "max"], but got {reduction}')

        self.quantile = quantile
        self.reduction = reduction
        self.total_elements = 0

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val = torch.rand(0,)
        self.max_val = torch.rand(0,)
        self.total_elements = 0

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis

        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            min_val = torch.quantile(y, dim=1, q=1 - self.quantile)
            max_val = torch.quantile(y, dim=1, q=self.quantile)
        else:
            min_val = torch.quantile(y, dim=1, q=1 - self.quantile)
            max_val = torch.quantile(y, dim=1, q=self.quantile)

            if self.reduction == 'max':
                min_val = torch.min(min_val, self.min_val)
                max_val = torch.max(max_val, self.max_val)
            elif self.reduction == 'mean':
                alpha = self.total_elements / (self.total_elements + 1)
                beta = 1 / (self.total_elements + 1)
                min_val = alpha * self.min_val + beta * min_val
                max_val = alpha * self.max_val + beta * max_val

        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        self.total_elements += 1

        return x_orig
