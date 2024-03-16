import copy

import torch
import torch.fx as fx
import torch.nn as nn
from torch.ao.quantization import disable_fake_quant, disable_observer

from typing import Iterable, List, Dict, Union

from equant.core.match.chain import _decompose_module
from equant.core.match import find_chain_forward, quantized
from equant.algorithms.smooth_quant.common import create_identity_layer_from_linear, \
    insert_module, add_observers, calibrate, smooth_quant_helper, LINEAR_LAYERS, ACTIVATIONS

from equant.algorithms.smooth_quant.utils import save_quantization_state


__all__ = [
    'smooth_quant'
]


def insert_identity_linear_layer(
    node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:

    linear2 = graph_module.get_submodule(node.target)
    linear2 = _decompose_module(linear2)[0]

    linear1 = create_identity_layer_from_linear(linear2)
    module_name = insert_module(linear1, graph_module, named_modules)

    with graph_module.graph.inserting_before(node):
        linear1_node = graph_module.graph.call_module(module_name, args=node.args, kwargs=node.kwargs)

    node.args = (linear1_node,)


def find_smooth_quant_chain(
    node: fx.Node,
    graph_module: fx.GraphModule
) -> List[fx.Node]:

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, ACTIVATIONS, LINEAR_LAYERS])

    if chain is not None:
        return chain

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, LINEAR_LAYERS])

    if chain is not None:
        return chain

    return None


def insert_identity_linear_layers(
    quantized_nodes: List[fx.Node],
    graph_module: fx.GraphModule
) -> fx.GraphModule:
    
    named_modules = dict(graph_module.named_modules())

    for node in quantized_nodes:
        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)
            module = _decompose_module(module)[0]
            if isinstance(module, LINEAR_LAYERS) and find_smooth_quant_chain(node, graph_module) is None:
                insert_identity_linear_layer(node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module


def get_quantized_nodes(
    graph_module: fx.GraphModule
) -> List[str]:
    
    node: fx.Node
    quantized_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)
            if quantized(module):
                quantized_nodes.append(node)

    return quantized_nodes


@torch.no_grad()
def smooth_quant(
    model: Union[nn.Module, fx.GraphModule],
    dataloader: Iterable,
    alpha: float = 0.5,
    absorb: bool = False,
    iters: Union[None, int] = None,
    quantile: float = 0.99999,
    inplace: bool = False
) -> fx.GraphModule:
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    quantized_nodes = get_quantized_nodes(graph_module)

    if not absorb:
        graph_module = insert_identity_linear_layers(quantized_nodes, graph_module)

    observer = add_observers(graph_module, quantile=quantile, nodes=quantized_nodes)

    with save_quantization_state(graph_module):
        graph_module.apply(disable_fake_quant).apply(disable_observer)
        calibrate(graph_module, dataloader, iters)
    
    activations_max_abs = observer.get_max_abs()
    observer.remove_hooks()
    del observer

    node: fx.Node
    for node in quantized_nodes:

        chain = find_smooth_quant_chain(node, graph_module)

        if chain is None:
            continue
        
        linear1 = graph_module.get_submodule(chain[0].target)
        linear2 = graph_module.get_submodule(chain[-1].target)

        # TODO: check if linear is actually linear (may be the embedding layer for instance)
        linear1 = _decompose_module(linear1)[0]
        linear2 = _decompose_module(linear2)[0]

        smooth_quant_helper(linear1, linear2, activations_max_abs[node.name], alpha=alpha)

    return graph_module
