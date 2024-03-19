import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize, _is_fake_quant_script_module


__all__ = [
    'reset_observer'
]


def reset_observer(mod: nn.Module) -> None:
    """
    Resets observation statistics for this module, if applicable.
    """
    if isinstance(mod, FakeQuantize) or _is_fake_quant_script_module(mod):
        mod.activation_post_process.reset_min_max_vals()
