import copy
import torch.nn as nn
import torch.fx as fx

import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
from torch.nn.utils.parametrize import type_before_parametrizations

from types import FunctionType
from typing import List, Tuple, Union


__all__ = [
    'has_bn',
    'quantized',
    'wrap_into_sequential',
    'decompose_module_to_float',
    'decompose_quant_module',
    'find_chain_forward',
    'find_chain_backward'
]


def has_bn(modules: nn.Module) -> bool:
    
    BNS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    if not issubclass(type(modules), nn.Sequential):
        modules = nn.Sequential(modules)

    for module in modules:
        if issubclass(type(module), BNS):
            return True
        
    return False


def quantized(module: nn.Module) -> bool:
    return hasattr(module, 'weight_fake_quant')


def float_fused(module: nn.Module) -> bool:
    return not quantized(module) and issubclass(type(module), nni._FusedModule)


def quant_fused(module: nn.Module) -> bool:
    return quantized(module) and issubclass(type(module), nni._FusedModule)


def wrap_into_sequential_helper(module: nn.Module) -> nn.Sequential:
    
    if not isinstance(module, nn.Sequential):
        module = nn.Sequential(module)

    return module


def wrap_into_sequential(module: nn.Module) -> nn.Sequential:
    
    if float_fused(module):
        return module
    
    if quant_fused(module):
        return decompose_quant_module(module)

    module = wrap_into_sequential_helper(module)
    return module


def decompose_quant_module(module: nn.Module) -> Union[nn.Module, nn.Sequential]:

    FP_LINEAR_TO_QAT = {
        nn.Conv1d: nnqat.Conv1d,
        nn.Conv2d: nnqat.Conv2d,
        nn.Conv3d: nnqat.Conv3d,
        nn.Linear: nnqat.Linear
    }
    
    bn = None
    if hasattr(module, 'bn'):
        bn = module.bn
    
    float_module = module.to_float()
    
    if issubclass(type(float_module), nn.Sequential):
        float_module, activation = float_module[0], float_module[1]
    else:
        activation = None
    
    float_module.qconfig = copy.deepcopy(module.qconfig)
    quant_base = FP_LINEAR_TO_QAT[type(float_module)].from_float(float_module)
    quant_base.weight_fake_quant = copy.deepcopy(module.weight_fake_quant)

    modules = [quant_base]

    if bn:
        modules.append(bn)

    if activation:
        modules.append(activation)

    modules = nn.Sequential(*modules)
    if len(modules) == 1:
        modules = modules[0]

    return modules


def decompose_module_to_float(module: nn.Module) -> Union[nn.Module, nn.Sequential]:
    
    if hasattr(module, 'weight_fake_quant'):
        module = decompose_quant_module(module)

    module = wrap_into_sequential_helper(module)

    modules = []
    for m in module:

        if hasattr(m, 'weight_fake_quant'):
            m = m.to_float()

        modules.append(m)

    if len(modules) == 1:
        return modules[0]
    
    modules = nn.Sequential(*modules)
    return modules


def decompose_module_to_types(module: nn.Module) -> Union[type, Tuple[type]]:

    module = decompose_module_to_float(module)
    module = wrap_into_sequential_helper(module)

    types = [type_before_parametrizations(m) for m in module]

    if len(types) == 1:
        return types[0]
    
    types = tuple(types)
    return types


def split_patterns(
    patterns: Union[List[type], Tuple[type]]
) -> Tuple[Tuple[type], Tuple[type]]:

    class_obj_patterns = []
    functional_patterns = []

    for pattern in patterns:

        if type(pattern) == FunctionType:
            functional_patterns.append(pattern)
        else:
            class_obj_patterns.append(pattern)

    class_obj_patterns = tuple(class_obj_patterns)
    functional_patterns = tuple(functional_patterns)

    return class_obj_patterns, functional_patterns


def match_functional_pattern(
    node: fx.Node,
    pattern: Tuple[nn.Module]
) -> int:
    
    if pattern in ['*', '**']:
        return 1
    
    _, functional_patterns = split_patterns(pattern)
    if node.target in functional_patterns:
        return 1
    
    return 0


def match_module_pattern(
    node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: Tuple[nn.Module],
    depth: int,
    backward: bool
) -> int:
    
    if patterns[depth] == '**':
        return 1
    
    module = graph_module.get_submodule(node.target)
    modules = wrap_into_sequential_helper(decompose_module_to_float(module))

    if backward:
        modules = modules[::-1]
    
    for i, module in enumerate(modules):

        if depth + i >= len(patterns):
            return i

        if patterns[depth + i] == '*':
            continue

        class_obj_patterns, _ = split_patterns(patterns[depth + i])
        if not issubclass(type(module), class_obj_patterns):
            return 0
    
    return len(modules)


def match_pattern(
    node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]],
    depth: int,
    backward: bool
) -> int:
    
    if node.op == 'call_function':
        return match_functional_pattern(node, patterns[depth])

    if node.op == 'call_module':
        return match_module_pattern(node, graph_module, patterns, depth, backward)
    
    return 0


def find_chain_helper(
    node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]],
    backward: bool,
    depth: int
) -> Union[None, List[fx.Node]]:
    
    if depth == len(patterns):
        return []
    
    if not backward and len(node.users) != 1:
        return None
    
    if backward and len(node.args) != 1:
        return None

    depth_stride = match_pattern(node, graph_module, patterns, depth, backward)
    if depth_stride > 0:

        if backward:
            next_node = node.args[0]
        else:
            next_node = next(iter(node.users))

        chain = find_chain_helper(next_node, graph_module, patterns, backward, depth + depth_stride)

        if chain is not None:
            chain.append(node)
            return chain
    
    return None


def find_chain_base(
    node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]],
    backward: bool
) -> Union[None, List[fx.Node]]:
    
    if backward:
        patterns = patterns[::-1]

    chain = find_chain_helper(node, graph_module, patterns, backward, depth=0)

    if chain is not None and not backward:
        chain = chain[::-1]

    return chain


def find_chain_forward(node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]]
) -> Union[None, List[fx.Node]]:
    
    return find_chain_base(node, graph_module, patterns, backward=False)


def find_chain_backward(node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]]
) -> Union[None, List[fx.Node]]:
    
    return find_chain_base(node, graph_module, patterns, backward=True)
