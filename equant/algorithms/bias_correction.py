import copy
import torch
import torch.nn as nn
import torch.fx as fx
from torch.ao.quantization import disable_observer, disable_fake_quant, enable_fake_quant

import warnings
from typing import Any, Iterable

from equant.core.match.chain import _decompose_module
from equant.core.match import has_bn, quantized
from equant.core.subgraph import create_subgraph
from equant.core.feature_extractor import collect_inputs_outputs_for_subgraph, model_forward
from equant.fuse.utils import replace_node_module
from equant.core.interpreter import DataInterpreter


__all__ = [
    'bias_correction'
]


LINEAR_LAYERS = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
)


@torch.no_grad()
def collect_decomposed_module_inputs_outputs(
    modules: nn.Sequential,
    inputs: Iterable,
    device: torch.device,
    start: int,
    end: int,
) -> Any: 
    
    modules_inputs = []
    modules_outputs = []

    for i, module in enumerate(modules):

        if i >= end:
            break
        
        if start <= i < end:
            modules_inputs.append(inputs)
        
        inputs = torch.cat([model_forward(module, input, device) for input in inputs])

        if start <= i < end:
            modules_outputs.append(inputs)

    if len(modules_inputs) == 1:
        modules_inputs = modules_inputs[0]

    if len(modules_outputs) == 1:
        modules_outputs = modules_outputs[0]

    return modules_inputs, modules_outputs


@torch.no_grad()
def bias_correction(
    graph_module: fx.GraphModule,
    dataloader: Iterable,
    inplace: bool = False,
    cache_data: bool = False
) -> fx.GraphModule:
    
    # TODO: Add progress bar
    
    if not inplace:
        graph_module = copy.deepcopy(graph_module)

    graph_module.apply(disable_observer)
    device = next(iter(graph_module.parameters())).device
    fp_graph_module = copy.deepcopy(graph_module).apply(disable_fake_quant)

    named_modules = dict(graph_module.named_modules())

    node: fx.Node
    fp_node: fx.Node
    for fp_node, node in zip(fp_graph_module.graph.nodes, graph_module.graph.nodes):

        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)

            if quantized(module):
                modules = _decompose_module(module)

                if not isinstance(modules[0], LINEAR_LAYERS):
                    continue

                if has_bn(modules):
                    warnings.warn(f'Skipping {module} optimization as it contains batch \
                                  normalization module. Consider using batchnorm fuse')
                    continue
                
                fp_module = fp_graph_module.get_submodule(fp_node.target)
                fp_modules = _decompose_module(fp_module)

                quant_subgraph = create_subgraph(graph_module, [node.name])
                quant_inputs = collect_inputs_outputs_for_subgraph(graph_module, quant_subgraph, dataloader)

                fp_subgraph = create_subgraph(fp_graph_module, [fp_node.name])
                fp_inputs = collect_inputs_outputs_for_subgraph(fp_graph_module, fp_subgraph, dataloader)
                
                _, quant_outputs = collect_decomposed_module_inputs_outputs(modules, quant_inputs, device, start=0, end=1)
                _, fp_outputs = collect_decomposed_module_inputs_outputs(fp_modules, fp_inputs, device, start=0, end=1)

                if isinstance(modules[0], nn.Linear):
                    fp_outputs = fp_outputs.transpose(0, -1)
                    quant_outputs = quant_outputs.transpose(0, -1)
                else:
                    fp_outputs = fp_outputs.transpose(0, 1)
                    quant_outputs = quant_outputs.transpose(0, 1)

                bias_correction = (fp_outputs - quant_outputs).flatten(start_dim=1).mean(dim=1)

                if module.bias is not None:
                    module.bias.data = module.bias.data + bias_correction
                else:
                    module.bias = nn.Parameter(bias_correction)

                replace_node_module(node, named_modules, module)

    graph_module.apply(enable_fake_quant)

    return graph_module


# NOTE: Subject of Deprication
@torch.no_grad()
def fast_bias_correction(
    graph_module: fx.GraphModule,
    dataloader: Iterable,
    inplace: bool = False,
    cache_data: bool = False
) -> fx.GraphModule:
    
    # TODO: Add progress bar
    
    if not inplace:
        graph_module = copy.deepcopy(graph_module)

    graph_module.apply(disable_observer)
    device = next(iter(graph_module.parameters())).device
    fp_graph_module = copy.deepcopy(graph_module).apply(disable_fake_quant)

    interpreter = DataInterpreter(graph_module, cache_data=cache_data)
    interpreter.initialize_env(dataloader)

    fp_interpreter = DataInterpreter(fp_graph_module, cache_data=cache_data)
    fp_interpreter.initialize_env(dataloader)

    named_modules = dict(graph_module.named_modules())

    node: fx.Node
    fp_node: fx.Node
    for fp_node, node in zip(fp_graph_module.graph.nodes, graph_module.graph.nodes):

        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)

            if quantized(module):
                modules = _decompose_module(module)

                if not isinstance(modules[0], LINEAR_LAYERS):
                    interpreter._run_node(node)
                    fp_interpreter._run_node(fp_node)
                    continue

                if has_bn(modules):
                    interpreter._run_node(node)
                    fp_interpreter._run_node(fp_node)
                    warnings.warn(f'Skipping {module} optimization as it contains batch \
                                  normalization module. Consider using batchnorm fuse')
                    continue
                
                fp_module = fp_graph_module.get_submodule(fp_node.target)
                fp_modules = _decompose_module(fp_module)
                
                _, quant_outputs = collect_decomposed_module_inputs_outputs(modules, interpreter.env[node.args[0]], device, start=0, end=1)
                _, fp_outputs = collect_decomposed_module_inputs_outputs(fp_modules, fp_interpreter.env[fp_node.args[0]], device, start=0, end=1)

                if isinstance(modules[0], nn.Linear):
                    fp_outputs = fp_outputs.transpose(0, -1)
                    quant_outputs = quant_outputs.transpose(0, -1)
                else:
                    fp_outputs = fp_outputs.transpose(0, 1)
                    quant_outputs = quant_outputs.transpose(0, 1)

                bias_correction = (fp_outputs - quant_outputs).flatten(start_dim=1).mean(dim=1)

                if module.bias is not None:
                    module.bias.data = module.bias.data + bias_correction
                else:
                    module.bias = nn.Parameter(bias_correction)

                replace_node_module(node, named_modules, module)
        
        interpreter._run_node(node)
        fp_interpreter._run_node(fp_node)

    del interpreter
    del fp_interpreter

    graph_module.apply(enable_fake_quant)

    return graph_module
