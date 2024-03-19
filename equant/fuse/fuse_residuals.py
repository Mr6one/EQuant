import copy
import operator

import torch
import torch.nn as nn
import torch.fx as fx

from typing import Dict, Union

from equant.fuse.utils import replace_node_module, LINEAR


__all__ = [
    'fuse_residuals'
]


def create_identity_fused_linear(
    linear: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
):
    
    if isinstance(linear, nn.Linear):
        if linear.in_features != linear.out_features:
            raise RuntimeError(
                f'Fusing invalid residual connection. Make sure in_channels = out_channels. Got in_channels = {linear.in_features} and out_channels = {linear.out_features}'
            )

        linear.weight.data += torch.eye(linear.weight.dim(), device=linear.weight.device)
    else:
        if linear.in_channels != linear.out_channels:
            raise RuntimeError(
                f'Fusing invalid residual connection. Make sure in_channels = out_channels. Got in_channels = {linear.in_channels} and out_channels = {linear.out_channels}'
            )

        identity = torch.empty_like(linear.weight.data)
        nn.init.dirac_(identity, groups=linear.groups)
        linear.weight.data += identity
    return linear


def fuse_residual_helper(
    linear_node: fx.Node,
    add_node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:
    
    conv = named_modules[linear_node.target]
    conv_fused = create_identity_fused_linear(conv)

    named_modules[linear_node.target] = conv_fused
    replace_node_module(linear_node, named_modules, conv_fused)
    add_node.replace_all_uses_with(linear_node)
    graph_module.graph.erase_node(add_node)


def fuse_residuals(
    model: Union[nn.Module, fx.GraphModule],
    inplace: bool = False
) -> fx.GraphModule:
    
    '''
    Fuse linear(x) + x
    '''
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    named_modules = dict(model.named_modules())

    node: fx.Node
    for node in graph_module.graph.nodes:
        
        if node.op == 'call_function' and node.target in [operator.add, torch.add]:

            add_arg1 = node.args[0]
            add_arg2 = node.args[1]
            
            if add_arg1.op == 'call_module' and \
                isinstance(named_modules[add_arg1.target], LINEAR):
                
                if add_arg1.args[0] == add_arg2:
                    fuse_residual_helper(add_arg1, node, graph_module, named_modules)

            if add_arg2.op == 'call_module' and \
                isinstance(named_modules[add_arg2.target], LINEAR):

                if add_arg2.args[0] == add_arg1:
                    fuse_residual_helper(add_arg2, node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module
