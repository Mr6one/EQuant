import torch.nn as nn
import torch.ao.nn.qat as nnqat

from typing import Union

from equant.core.match.module import quantized


__all__ = [
    'decompose_module',
    'wrap_into_sequential'
]


def wrap_into_sequential(module: nn.Module) -> nn.Sequential:
    
    if not isinstance(module, nn.Sequential):
        module = nn.Sequential(module)

    return module


def decompose_quant_module(module: nn.Module) -> Union[nn.Module, nn.Sequential]:

    if not quantized(module):
        raise RuntimeError(f'Module {type(module)} is not quantized')

    FP_LINEAR_TO_QAT = {
        nn.Conv1d: nnqat.Conv1d,
        nn.Conv2d: nnqat.Conv2d,
        nn.Conv3d: nnqat.Conv3d,
        nn.Linear: nnqat.Linear
    }
    
    if hasattr(module, 'bn'):
        bn = module.bn
    else:
        bn = None
    
    float_module = module.to_float()
    
    if issubclass(type(float_module), nn.Sequential):
        float_module, activation = float_module[0], float_module[1]
    else:
        activation = None
    
    float_module.qconfig = module.qconfig
    quant_base = FP_LINEAR_TO_QAT[type(float_module)].from_float(float_module)

    quant_base.weight = module.weight
    quant_base.bias = module.bias
    quant_base.weight_fake_quant = module.weight_fake_quant

    modules = [quant_base]

    if bn:
        modules.append(bn)

    if activation:
        modules.append(activation)

    modules = nn.Sequential(*modules)
    if len(modules) == 1:
        modules = modules[0]

    return modules


def decompose_module(module: nn.Module, to_float: bool = False) -> Union[nn.Module, nn.Sequential]:
    
    if quantized(module):
        module = decompose_quant_module(module)

    module = wrap_into_sequential(module)

    modules = []
    for m in module:

        if quantized(m):
            m = m.to_float()

        modules.append(m)

    if len(modules) == 1:
        return modules[0]
    
    modules = nn.Sequential(*modules)
    return modules
