import torch.nn as nn
import torch.fx as fx

from types import FunctionType
from typing import List, Tuple, Union

from equant.core.match.decompose import _decompose_module


__all__ = [
    'find_chain_forward',
    'find_chain_backward'
]


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
    modules = _decompose_module(module)

    if backward:
        modules = modules[::-1]
    
    for i, module in enumerate(modules):

        if depth + i >= len(patterns):
            return i

        if patterns[depth + i] == '*':
            continue

        class_obj_patterns, _ = split_patterns(patterns[depth + i])
        if not isinstance(module, class_obj_patterns):
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
    
    if depth >= len(patterns):
        return []
    
    if not backward and len(node.users) != 1:
        return
    
    if backward and len(node.args) != 1:
        return

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


def find_chain_forward(
    node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]]
) -> Union[None, List[fx.Node]]:
    
    return find_chain_base(node, graph_module, patterns, backward=False)


def find_chain_backward(
    node: fx.Node,
    graph_module: fx.GraphModule,
    patterns: List[Tuple[nn.Module]]
) -> Union[None, List[fx.Node]]:
    
    return find_chain_base(node, graph_module, patterns, backward=True)
