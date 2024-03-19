import copy
import numpy as np

import torch
import torch.fx as fx
import torch.nn as nn
from torch import Tensor
from torch.hub import tqdm
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FakeQuantize, disable_observer, disable_fake_quant, enable_fake_quant, enable_observer

from typing import Iterable, List, Dict, Union

from equant.core.search import find_chain_forward, find_chain_backward, quantized
from equant.core.search.chain import _decompose_module
from equant.observers.utils import reset_observer
from equant.core.subgraph import create_subgraph
from equant.core.feature_extractor import collect_inputs_outputs_for_subgraph, model_forward
from equant.core.interpreter import DataInterpreter
from equant.algorithms.smooth_quant.common import create_identity_layer_from_linear, \
    insert_module, add_observers, calibrate, smooth_quant_helper, LINEAR_LAYERS, ACTIVATIONS


__all__ = [
    'smooth_quant_auto_tune'
]

QUANTIZERS = (FakeQuantize,)


def insert_identity_quant_linear_layer(
    node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:
    
    # TODO: Fix observer insertion rule (check onnx graph for this issue)

    linear2 = graph_module.get_submodule(node.target).to_float()

    if isinstance(linear2, nn.Sequential):
        linear2 = linear2[0]

    linear1 = create_identity_layer_from_linear(linear2)
    linear1.qconfig = None
    module_name = insert_module(linear1, graph_module, named_modules)

    with graph_module.graph.inserting_before(node):
        linear1_node = graph_module.graph.call_module(module_name, args=node.args, kwargs=node.kwargs)

    graph_module.meta['_observed_graph_module_attrs'].node_name_to_scope[module_name] = (linear1_node.target, type(linear1))

    quantizer = copy.deepcopy(graph_module.get_submodule(node.args[0].target))
    quantizer.activation_post_process.reset_min_max_vals()
    module_name = insert_module(quantizer, graph_module, named_modules)

    with graph_module.graph.inserting_after(linear1_node):
        obs_node = graph_module.graph.call_module(module_name, args=(linear1_node,))

    node.args = (obs_node,)


def find_smooth_quant_chain_with_quantizers_backward(
    node: fx.Node,
    graph_module: fx.GraphModule
) -> List[fx.Node]:

    chain = find_chain_backward(node.args[0], graph_module, patterns=[LINEAR_LAYERS, ACTIVATIONS, QUANTIZERS])

    if chain is not None:
        return chain

    chain = find_chain_backward(node.args[0], graph_module, patterns=[LINEAR_LAYERS, QUANTIZERS])

    if chain is not None:
        return chain
    
    return None


def find_smooth_quant_chain_with_quantizers_forward(
    node: fx.Node,
    graph_module: fx.GraphModule
) -> List[fx.Node]:

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, ACTIVATIONS, QUANTIZERS, LINEAR_LAYERS])

    if chain is not None:
        return chain

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, QUANTIZERS, LINEAR_LAYERS])

    if chain is not None:
        return chain
    
    return None


def insert_identity_quant_linear_layers(
    graph_module: fx.GraphModule
) -> fx.GraphModule:
    
    named_modules = dict(graph_module.named_modules())

    node: fx.Node
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)

            if not quantized(module):
                continue

            module = _decompose_module(module)[0]
            if isinstance(module, LINEAR_LAYERS) and \
                find_smooth_quant_chain_with_quantizers_backward(node, graph_module) is None:
                insert_identity_quant_linear_layer(node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module
    

def find_optimal_alpha(
    quant_subgraph_module: fx.GraphModule,
    quant_inputs: Iterable,
    iters: Union[None, int],
    fp_outputs: Iterable,
    chain: List[fx.Node],
    min_alpha: float,
    max_alpha: float,
    steps: int,
    scale: Tensor,
    device: torch.device
) -> float:
    
    quant_subgraph_module.apply(disable_observer).apply(enable_fake_quant)

    min_loss = float('inf')
    optimal_aplha = (min_alpha + max_alpha) / 2
    
    alphas = np.linspace(min_alpha, max_alpha, steps)
    alphas = np.append(optimal_aplha, alphas)
    
    if min_alpha < 0.5 < max_alpha and 0.5 not in alphas:
        alphas = np.append(alphas, 0.5)

    for alpha in alphas:
        quant_subgraph_module_tmp = copy.deepcopy(quant_subgraph_module).to(device)

        linear1 = quant_subgraph_module_tmp.get_submodule(chain[0].target)
        linear2 = quant_subgraph_module_tmp.get_submodule(chain[-1].target)

        linear1 = _decompose_module(linear1)[0]
        linear2 = _decompose_module(linear2)[0]

        smooth_quant_helper(linear1, linear2, scale, alpha=alpha)

        quant_subgraph_module_tmp.apply(reset_observer).apply(disable_fake_quant)
        calibrate(quant_subgraph_module_tmp, quant_inputs, iters)
        quant_subgraph_module_tmp.apply(disable_observer).apply(enable_fake_quant)
        
        loss = 0
        for i, (input, output) in enumerate(zip(quant_inputs, fp_outputs)):
            output = output.to(device)
            qoutput = model_forward(quant_subgraph_module_tmp, input, device)
            loss = loss * (i / (i + 1)) + F.mse_loss(qoutput, output).item() / (i + 1)

        if loss < min_loss:
            min_loss = loss
            optimal_aplha = alpha

    return optimal_aplha


def create_execution_plan(
    graph_module: fx.GraphModule
) -> List[List[fx.Node]]:
    
    chains2optimize = []
    for node in graph_module.graph.nodes:

        chain = find_smooth_quant_chain_with_quantizers_forward(node, graph_module)

        if chain is None:
            continue

        module = graph_module.get_submodule(chain[-1].target)
        if not quantized(module):
            continue

        chains2optimize.append(chain)

    return chains2optimize


@torch.no_grad()
def smooth_quant_auto_tune(
    model: Union[nn.Module, fx.GraphModule],
    dataloader: Iterable,
    min_alpha: float = 0.3,
    max_alpha: float = 0.7,
    steps: int = 10,
    absorb: bool = True,
    iters: Union[None, int] = None,
    quantile: float = 0.99999,
    inplace: bool = False,
    verbose: bool =  True
) -> fx.GraphModule:
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    if not absorb:
        graph_module = insert_identity_quant_linear_layers(graph_module)

    device = next(iter(graph_module.parameters())).device

    fp_graph_module = copy.deepcopy(graph_module).apply(disable_fake_quant).apply(disable_observer)
    graph_module.apply(enable_observer).apply(enable_fake_quant)

    chains2optimize = create_execution_plan(graph_module)
    pbar = tqdm(total=len(chains2optimize), desc='smooth quant auto-tune', initial=0, position=0, leave=True, disable=not verbose, delay=0)
    for chain in chains2optimize:
        pbar.update(1)
            
        linear1 = graph_module.get_submodule(chain[0].target)
        linear2 = graph_module.get_submodule(chain[-1].target)

        linear1 = _decompose_module(linear1)[0]
        linear2 = _decompose_module(linear2)[0]

        subgraph_module = create_subgraph(graph_module, node_names=[node.name for node in chain])
        subgraph_module.apply(disable_observer).apply(disable_fake_quant)
        quant_inputs, _ = collect_inputs_outputs_for_subgraph(graph_module, subgraph_module, dataloader)

        fp_subgraph_module = create_subgraph(fp_graph_module, node_names=[node.name for node in chain])
        _, fp_outputs = collect_inputs_outputs_for_subgraph(fp_graph_module, fp_subgraph_module, dataloader)
       
        observer = add_observers(subgraph_module, quantile=quantile)
        calibrate(subgraph_module, quant_inputs, iters)
        activations_max_abs = observer.get_max_abs()
        observer.remove_hooks()

        optimal_aplha = find_optimal_alpha(
            subgraph_module,
            quant_inputs,
            iters,
            fp_outputs,
            chain,
            min_alpha,
            max_alpha,
            steps,
            activations_max_abs[chain[-1].name],
            device
        )

        smooth_quant_helper(linear1, linear2, activations_max_abs[chain[-1].name], alpha=optimal_aplha)

    graph_module.apply(disable_observer)

    return graph_module


@torch.no_grad()
def fast_smooth_quant_auto_tune(
    model: Union[nn.Module, fx.GraphModule],
    dataloader: Iterable,
    min_alpha: float = 0.3,
    max_alpha: float = 0.7,
    steps: int = 10,
    absorb: bool = False,
    iters: Union[None, int] = None,
    quantile: float = 0.99999,
    cache_data: bool = False,
    inplace: bool = False
) -> fx.GraphModule:
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    if not absorb:
        graph_module = insert_identity_quant_linear_layers(graph_module)

    device = next(iter(graph_module.parameters())).device

    fp_graph_module = copy.deepcopy(graph_module).apply(disable_fake_quant).apply(disable_observer)
    graph_module.apply(enable_observer).apply(enable_fake_quant)

    interpreter = DataInterpreter(graph_module, cache_data=cache_data)
    interpreter.initialize_env(dataloader)

    fp_interpreter = DataInterpreter(fp_graph_module, cache_data=cache_data)
    fp_interpreter.initialize_env(dataloader)

    node: fx.Node
    fp_node: fx.Node
    for fp_node, node in zip(fp_graph_module.graph.nodes, graph_module.graph.nodes):

        chain = find_smooth_quant_chain_with_quantizers_forward(node, graph_module)

        if chain is None:
            fp_interpreter._run_node(fp_node)
            interpreter._run_node(node)
            continue

        module = graph_module.get_submodule(chain[-1].target)
        if not quantized(module):
            fp_interpreter._run_node(fp_node)
            interpreter._run_node(node)
            continue
            
        linear1 = graph_module.get_submodule(chain[0].target)
        linear2 = graph_module.get_submodule(chain[-1].target)

        linear1 = _decompose_module(linear1)[0]
        linear2 = _decompose_module(linear2)[0]

        subgraph_module = create_subgraph(graph_module, node_names=[node.name for node in chain])
        subgraph_module.apply(disable_observer).apply(disable_fake_quant)

        quant_inputs = interpreter.env[node.args[0]]
        fp_outputs = [model_forward(subgraph_module, data, device) for data in fp_interpreter.env[fp_node.args[0]]]

        observer = add_observers(subgraph_module, quantile=quantile)
        calibrate(subgraph_module, quant_inputs, iters)
        activations_max_abs = observer.get_max_abs()
        observer.remove_hooks()

        optimal_aplha = find_optimal_alpha(
            subgraph_module,
            quant_inputs,
            iters,
            fp_outputs,
            chain,
            min_alpha,
            max_alpha,
            steps,
            activations_max_abs[chain[-1].name],
            device
        )

        smooth_quant_helper(linear1, linear2, activations_max_abs[chain[-1].name], alpha=optimal_aplha)

        fp_interpreter._run_node(fp_node)
        interpreter._run_node(node)

    graph_module.apply(disable_observer)

    del interpreter
    del fp_interpreter

    return graph_module
