import torch
import torch.nn as nn
from torch import Tensor

from torch.ao.quantization.fake_quantize import FakeQuantize as _FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver

from typing import Union


def disable_fake_quant(mod: nn.Module) -> None:
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if issubclass(type(mod), _FakeQuantize):
        mod.disable_fake_quant()


def enable_fake_quant(mod: nn.Module) -> None:
    """
    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    if issubclass(type(mod), _FakeQuantize):
        mod.enable_fake_quant()


def disable_observer(mod: nn.Module) -> None:
    """
    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_observer)

    """
    if issubclass(type(mod), _FakeQuantize):
        mod.disable_observer()


def enable_observer(mod: nn.Module) -> None:
    """
    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_observer)

    """
    if issubclass(type(mod), _FakeQuantize):
        mod.enable_observer()

    
def reset_observer(mod: nn.Module) -> None:
    """
    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_observer)

    """
    if issubclass(type(mod), _FakeQuantize):
        mod.activation_post_process.reset_min_max_vals()


class RoundSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class GradScaler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
    

def round_ste(
    x: Tensor
) -> Tensor:
    return RoundSTE.apply(x)


def grad_scaler(
    x: Tensor, 
    scale: Union[float, Tensor]
) -> Tensor:
    return GradScaler.apply(x, scale)


def fake_quantize(
    x: Tensor, 
    scale: Tensor, 
    zero_point: Tensor, 
    quant_min: float, 
    quant_max: float
) -> Tensor:

    max_zero_point = int(quant_max - quant_min)

    zero_point = round_ste(zero_point).clip(0, max_zero_point)
    scale = scale.clip(1e-10)

    x = scale * ((round_ste(x / scale) + zero_point).clip(int(quant_min), int(quant_max)) - zero_point)
    return x


def fake_quantize_lsq(
    x: Tensor, 
    scale: Tensor, 
    zero_point: Tensor, 
    quant_min: float, 
    quant_max: float, 
    grad_scale: float = 1.0
) -> Tensor:

    max_zero_point = int(quant_max - quant_min)
    grad_scale = grad_scale / ((x.numel() * quant_max) ** 0.5)
    
    scale = scale.clip(1e-10)
    zero_point = round_ste(zero_point).clip(0, max_zero_point)

    scale = grad_scaler(scale, grad_scale)
    zero_point = grad_scaler(zero_point, grad_scale)

    x = scale * ((round_ste(x / scale) + zero_point).clip(int(quant_min), int(quant_max)) - zero_point)
    return x


class FakeQuantizeTorchBase(_FakeQuantize):
    pass


class FakeQuantize(_FakeQuantize):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)

        self.shape = None
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.zero_point = nn.Parameter(torch.tensor([0.0])) # TODO: freeze it after clip(min=0), e.g. relu or relu6

    def fake_quantize(self, x):
        x = fake_quantize(x, self.scale, self.zero_point, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return x

    def forward(self, x: Tensor) -> Tensor:

        if self.shape is None:
            if self.activation_post_process.qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                ch_axis = self.activation_post_process.ch_axis
                self.shape = [1] * len(x.shape)
                self.shape[ch_axis] = x.shape[ch_axis]
            else:
                self.shape = [1]

        if self.observer_enabled[0] == 1:
            self.activation_post_process(x.detach())

            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)

            if self.scale.shape != _scale.shape:
                self.scale.data = self.scale.data.resize_(_scale.shape)
                self.zero_point.data = self.zero_point.data.resize_(_zero_point.shape)

            self.scale.data = self.scale.data.copy_(_scale).reshape(self.shape)
            self.zero_point.data = self.zero_point.data.copy_(_zero_point).reshape(self.shape)

        if self.fake_quant_enabled[0] == 1:
            x = self.fake_quantize(x)

        return x


class FakeQuantizeLSQ(FakeQuantize):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)

    def fake_quantize(self, x):
        x = fake_quantize_lsq(x, self.scale, self.zero_point, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return x
