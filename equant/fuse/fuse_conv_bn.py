import copy
import torch.nn as nn
import torch.fx as fx

from typing import Dict, Union

from equant.fuse.utils import _parent_name, replace_node_module, CONVS, BNS


__all__ = [
    'fuse_conv_bn'
]


def fuse_conv_bn_helper(
    conv_node: fx.Node,
    bn_node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:
    
    if len(conv_node.users) > 1:
        return
    
    conv = named_modules[conv_node.target]
    bn = named_modules[bn_node.target]
    
    fused_conv_bn = nn.utils.fuse_conv_bn_eval(conv, bn)
    
    named_modules[conv_node.target] = fused_conv_bn
    replace_node_module(conv_node, named_modules, fused_conv_bn)
    bn_node.replace_all_uses_with(conv_node)
    graph_module.graph.erase_node(bn_node)
    
    parent_name, name = _parent_name(bn_node.target)
    delattr(named_modules[parent_name], name)


def fuse_conv_bn(
    model: Union[nn.Module, fx.GraphModule],
    inplace: bool = False
) -> fx.GraphModule:
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    named_modules = dict(graph_module.named_modules())

    node: fx.Node
    for node in graph_module.graph.nodes:

        if node.op == 'call_module' and isinstance(named_modules[node.target], BNS):

            arg = node.args[0]

            if arg.op == 'call_module' and isinstance(named_modules[arg.target], CONVS):
                
                fuse_conv_bn_helper(arg, node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module