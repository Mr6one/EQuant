import torch
import torch.nn as nn

from torch.ao.quantization import MovingAverageMinMaxObserver
from equant.quantizers.fixed_qparams import FixedQParamsFakeQuantize


__all__ = [
    'FakeQuantizeLSQ'
]


class FakeQuantizeLSQ(FixedQParamsFakeQuantize):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, grad_factor=1.0, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)

        self.initialization = False
        self.grad_factor = grad_factor
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0.0], dtype=torch.float32))

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                if self.initialization:
                    raise RuntimeError("FakeQuantizeLSQ doesn't support dynamical shapes")
                self.scale = nn.Parameter(torch.ones_like(_scale, dtype=torch.float32))
                self.zero_point.resize_(_zero_point.shape)
                self.initialization = True
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, 
                    self.activation_post_process.quant_max, self.grad_factor
                )
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.activation_post_process.quant_min, 
                    self.activation_post_process.quant_max, self.grad_factor
                )
        return X
