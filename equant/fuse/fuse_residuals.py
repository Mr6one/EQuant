import copy
import operator

import torch
import torch.nn as nn
import torch.fx as fx

from typing import Dict, Union

from equant.fuse.utils import replace_node_module, CONVS


__all__ = [
    'fuse_residuals'
]


def create_identity_fused_conv(
    conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
):
    if conv.in_channels != conv.out_channels:
        raise RuntimeError(
            f'Fusing invalid residual connection. Make sure in_channels = out_channels. Your in_channels = {conv.in_channels} and out_channels = {conv.out_channels}'
        )

    identity = torch.empty_like(conv.weight.data)
    nn.init.dirac_(identity, groups=conv.groups)
    conv.weight.data += identity
    return conv


def fuse_residual_helper(
    conv_node: fx.Node,
    add_node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:
    
    conv = named_modules[conv_node.target]
    conv_fused = create_identity_fused_conv(conv)

    named_modules[conv_node.target] = conv_fused
    replace_node_module(conv_node, named_modules, conv_fused)
    add_node.replace_all_uses_with(conv_node)
    graph_module.graph.erase_node(add_node)


def fuse_residuals(
    model: Union[nn.Module, fx.GraphModule],
    inplace: bool = False
) -> fx.GraphModule:
    
    '''
    Fuse conv(x) + x
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
                isinstance(named_modules[add_arg1.target], CONVS):
                
                if add_arg1.args[0] == add_arg2:
                    fuse_residual_helper(add_arg1, node, graph_module, named_modules)

            if add_arg2.op == 'call_module' and \
                isinstance(named_modules[add_arg2.target], CONVS):

                if add_arg2.args[0] == add_arg1:
                    fuse_residual_helper(add_arg2, node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module
