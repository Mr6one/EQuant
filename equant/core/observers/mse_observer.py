import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver


__all__ = [
    'MSEObserver',
    'MSEPerChannelObserver'
]


def quantization_loss(x, scale, zero_point, quant_min, quant_max, ch_axis=-1):

    def mse_loss(y_pred, y_true, ch_axis=-1, p=2.0):
        loss = (y_pred - y_true).pow(p)
        if ch_axis == -1:
            return loss.mean()
        else:
            x_dim = y_pred.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[ch_axis] = 0
            new_axis_list[0] = ch_axis
            loss = loss.permute(new_axis_list)
            return loss.mean(1)

    if ch_axis != -1:
        x_q = torch.fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max)
    else:
        x_q = torch.fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max)

    loss = mse_loss(x_q, x, ch_axis)
    return loss


class MSEObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the MSE loss.

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
        n_steps=100,
        scale_range_ratio = [0.01, 1],
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
        self.n_steps = n_steps
        self.scale_range_ratio = scale_range_ratio
        self.total_elements = 0

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))
        self.total_elements = 0

    def tune_scale(self, x, scale):
        # we already know that scheme is symmetric
        best_zero_point = torch.zeros(scale.size(), dtype=torch.int64, device=x.device)
        if self.dtype in [torch.quint8, torch.uint8]:
            if self.has_customized_qrange:
                best_zero_point = best_zero_point.new_full(best_zero_point.size(), (self.quant_min + self.quant_max) // 2)
            else:
                best_zero_point = best_zero_point.new_full(best_zero_point.size(), 128)

        min_loss = float('inf')
        best_scale = scale.clone()
        alpha, beta = self.scale_range_ratio
        for i in range(self.n_steps):
            s = alpha * scale + (beta - alpha) * scale * (i + 1) / self.n_steps
            s = torch.max(s, self.eps)
            loss = quantization_loss(x, s, best_zero_point, self.quant_min, self.quant_max)
            min_loss = loss.item()
            best_scale = torch.where(loss < min_loss, s, best_scale)
            min_loss = torch.min(min_loss, loss)

        return best_scale, best_zero_point

    def tune_scale_offset(self, x, scale, zero_point):
        min_loss = float('inf')
        best_scale = scale.clone()
        best_zero_point = zero_point.clone()
        alpha, beta = self.scale_range_ratio
        for i in range(self.n_steps):
            s = alpha * scale + (beta - alpha) * scale * (i + 1) / self.n_steps
            s = torch.max(s, self.eps)
            for zp in range(self.quant_min, self.quant_max + 1):
                zp.resize_(s.shape)
                loss = quantization_loss(x, s, zp, self.quant_min, self.quant_max)
                best_scale = torch.where(loss < min_loss, s, best_scale)
                best_zero_point = torch.where(loss < min_loss, zp, best_zero_point)
                min_loss = torch.min(min_loss, loss)

        return best_scale, best_zero_point
    
    @torch.no_grad()
    def tune_qparams(self, x, scale, zero_point, min_val, max_val):
        if self.qscheme == torch.per_tensor_symmetric:
            scale, zero_point = self.tune_scale(x, scale)
            min_val = -scale * (self.quant_max - self.quant_min) / 2
            max_val = scale * (self.quant_max - self.quant_min) / 2
        else:
            scale, zero_point = self.tune_scale_offset(x, scale, zero_point)
            min_val = scale * (self.quant_min - zero_point)
            max_val = scale * (self.quant_max - zero_point)

        return min_val, max_val

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val, max_val = torch.aminmax(x)
        
        scale, zero_point = self._calculate_qparams(min_val, max_val)
        min_val, max_val = self.tune_qparams(x, scale, zero_point, min_val, max_val)

        if self.total_elements > 0:
            alpha = self.total_elements / (self.total_elements + 1)
            beta = 1 / (self.total_elements + 1)
            min_val = alpha * self.min_val + beta * min_val
            max_val = alpha * self.max_val + beta * max_val

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        self.total_elements += 1

        return x_orig
    

class MSEPerChannelObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the MSE loss.

    This observer computes the quantization parameters based on the minimums 
    and maximums of quantile of the incoming tensors. The module
    records the minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

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
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(
        self,
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
        self.total_elements = 0

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor([]))
        self.max_val.copy_(torch.tensor([]))
        self.total_elements = 0

    def tune_scale(self, x, scale):
        # we already know that scheme is symmetric
        best_zero_point = torch.zeros(scale.size(), dtype=torch.int64, device=x.device)
        if self.dtype in [torch.quint8, torch.uint8]:
            if self.has_customized_qrange:
                best_zero_point = best_zero_point.new_full(best_zero_point.size(), (self.quant_min + self.quant_max) // 2)
            else:
                best_zero_point = best_zero_point.new_full(best_zero_point.size(), 128)

        min_loss = float('inf')
        best_scale = scale.clone()
        alpha, beta = self.scale_range_ratio
        for i in range(self.n_steps):
            s = alpha * scale + (beta - alpha) * scale * (i + 1) / self.n_steps
            s = torch.max(s, self.eps)
            loss = quantization_loss(x, s, best_zero_point, self.quant_min, self.quant_max, self.ch_axis)
            min_loss = loss.item()
            best_scale = torch.where(loss < min_loss, s, best_scale)
            min_loss = torch.min(min_loss, loss)

        return best_scale, best_zero_point

    def tune_scale_offset(self, x, scale, zero_point):
        min_loss = float('inf')
        best_scale = scale.clone()
        best_zero_point = zero_point.clone()
        alpha, beta = self.scale_range_ratio
        for i in range(self.n_steps):
            s = alpha * scale + (beta - alpha) * scale * (i + 1) / self.n_steps
            s = torch.max(s, self.eps)
            for zp in range(self.quant_min, self.quant_max + 1):
                zp.resize_(s.shape)
                loss = quantization_loss(x, s, zp, self.quant_min, self.quant_max, self.ch_axis)
                best_scale = torch.where(loss < min_loss, s, best_scale)
                best_zero_point = torch.where(loss < min_loss, zp, best_zero_point)
                min_loss = torch.min(min_loss, loss)

        return best_scale, best_zero_point
    
    @torch.no_grad()
    def tune_qparams(self, x, scale, zero_point, min_val, max_val):
        if self.qscheme == torch.per_tensor_symmetric:
            scale, zero_point = self.tune_scale(x, scale)
            min_val = -scale * (self.quant_max - self.quant_min) / 2
            max_val = scale * (self.quant_max - self.quant_min) / 2
        else:
            scale, zero_point = self.tune_scale_offset(x, scale, zero_point)
            min_val = scale * (self.quant_min - zero_point)
            max_val = scale * (self.quant_max - zero_point)

        return min_val, max_val

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)

        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        min_val, max_val = torch.aminmax(y, dim=1)

        scale, zero_point = self._calculate_qparams(min_val, max_val)
        min_val, max_val = self.tune_qparams(x, scale, zero_point, min_val, max_val)
        
        if self.total_elements > 0:
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
