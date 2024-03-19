import torch.nn as nn
from typing import Union


__all__ = [
    'has_bn',
    'quantized'
]


def has_bn(module: Union[nn.Module, nn.Sequential]) -> bool:
    
    BNS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    if not isinstance(module, nn.Sequential):
        module = nn.Sequential(module)
    
    for m in module:
        if isinstance(m, BNS):
            return True
        
    return False


def quantized(module: nn.Module) -> bool:
    return hasattr(module, 'weight_fake_quant')

