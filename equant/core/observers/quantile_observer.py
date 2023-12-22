import torch
from torch.ao.quantization.utils import is_per_tensor, is_per_channel
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver


__all__ = [
    'QuantileObserver',
    'QuantilePerChannelObserver'
]


class QuantileObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    quantile of the min and max values.

    This observer computes the quantization parameters based on the quantile 
    of minimums and maximums of the incoming tensors. The module
    records the minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        quantile: Quantile for min/max.
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
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        **kwargs
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "QuantileObserver's qscheme only support \
                    torch.per_tensor_symmetric and torch.per_tensor_affine."
            )
        self.quantile = quantile
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            **kwargs
        )

        self.min_vals = []
        self.max_vals = []

    @torch.jit.export
    def calculate_quantile(self):

        if len(self.min_vals) == 0 or len(self.max_vals) == 0:
            return self.min_val, self.max_val

        min_vals = torch.cat(self.min_vals)
        max_vals = torch.cat(self.max_vals)

        min_val = torch.mean(min_vals)
        max_val = torch.mean(max_vals)
        
        return min_val, max_val

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        min_val, max_val = self.calculate_quantile()
        return self._calculate_qparams(min_val, max_val)
    
    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""

        self.min_vals = []
        self.max_vals = []

        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))

    @torch.jit.export
    def extra_repr(self):
        min_val, max_val = self.calculate_quantile()
        return f"min_val={min_val}, max_val={max_val}"

    def forward(self, x_orig):

        if x_orig.numel() == 0:
            return x_orig
        
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)

        # min_val, max_val = torch.aminmax(x.flatten(start_dim=1), dim=1)
        min_val = torch.quantile(x.flatten(start_dim=1), dim=1, q=1 - self.quantile)
        max_val = torch.quantile(x.flatten(start_dim=1), dim=1, q=self.quantile)

        self.min_vals.append(min_val)
        self.max_vals.append(max_val)

        return x_orig
    

class QuantilePerChannelObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the minimum and maximum of 
    incoming tensors, and uses this statistic to compute the quantization
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
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        **kwargs
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "QuantilePerChannelObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            **kwargs
        )
        self.quantile = quantile

        self.min_vals = []
        self.max_vals = []

    @torch.jit.export
    def calculate_quantile(self):
        
        if len(self.min_vals) == 0 or len(self.max_vals) == 0:
            return self.min_val, self.max_val

        min_vals = torch.stack(self.min_vals)
        max_vals = torch.stack(self.max_vals)

        min_val = torch.quantile(min_vals, 1 - self.quantile, dim=0)
        max_val = torch.quantile(max_vals, self.quantile, dim=0)

        return min_val, max_val

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        min_val, max_val = self.calculate_quantile()
        return self._calculate_qparams(min_val, max_val)
    
    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""

        self.min_vals = []
        self.max_vals = []

        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))

    @torch.jit.export
    def extra_repr(self):
        min_val, max_val = self.calculate_quantile()
        return f"min_val={min_val}, max_val={max_val}"

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

        min_val, max_val = torch.aminmax(y, dim=1)

        self.min_vals.append(min_val)
        self.max_vals.append(max_val)

        return x_orig
