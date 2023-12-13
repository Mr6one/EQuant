import copy
import numpy as np

import torch
import torch.fx as fx
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fake_quantize import FakeQuantize

from typing import Iterable, List, Dict, Tuple, Union

from equant.quantize import quantize
from equant.core.match import decompose_module_to_float, find_chain_forward, wrap_into_sequential, find_chain_backward, quantized
from equant.core.quantizers.fake_quantize import disable_fake_quant, enable_fake_quant, disable_observer, enable_observer, reset_observer
from equant.core.subgraph import create_subgraph, collect_inputs_outputs_for_subgraph, model_forward
from equant.core.interpreter import DataInterpreter


__all__ = [
    'smooth_quant',
    'smooth_quant_auto_tune'
]


LINEAR_LAYERS = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
)

ACTIVATIONS = (nn.ReLU, F.relu)

QUANTIZERS = (FakeQuantize,)


def create_identity_layer_from_linear(
    module: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
) -> Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]:

    if isinstance(module, nn.Linear):
        linear = nn.Linear(module.in_features, module.in_features, bias=False)
        linear.weight.data = torch.eye(module.in_features)
    else:
        class_obj = type(module)
        linear = class_obj(module.in_channels, module.in_channels, kernel_size=1, groups=module.groups, bias=False)
        nn.init.dirac_(linear.weight, groups=module.groups)

    return linear


class QuantileObserver:

    def __init__(self, quantile: float = 1.0) -> None:
        super().__init__()

        self.module_to_node_name = {}
        self.min_vals = {}
        self.max_vals = {}
        self.hooks = []
        self.quantile = quantile

    def _forward_hook(self, module: nn.Module, input: Tuple[Tensor], _) -> None:
        input: Tensor = input[0].detach()
        dim = list(range(input.dim()))
        del dim[1]

        min_val = input.amin(dim=dim)
        max_val = input.amax(dim=dim)

        node_name = self.module_to_node_name[module]

        if node_name not in self.min_vals:
            self.min_vals[node_name] = []

        self.min_vals[node_name].append(min_val)

        if node_name not in self.max_vals:
            self.max_vals[node_name] = []

        self.max_vals[node_name].append(max_val)

    def get_max_abs(self) -> Tensor:

        max_abs = {}

        for node_name in self.min_vals:
            min_val = torch.stack(self.min_vals[node_name]).quantile(1 - self.quantile, dim=0)
            max_val = torch.stack(self.max_vals[node_name]).quantile(self.quantile, dim=0)
            max_abs[node_name] = torch.maximum(max_val.abs(), min_val.abs())

        return max_abs
    
    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()


def add_observers(
    graph_module: fx.GraphModule,
    quantile: float,
    nodes: Union[None, List[fx.Node]] = None,
) -> QuantileObserver:
    
    nodes = nodes or graph_module.graph.nodes
    
    observer = QuantileObserver(quantile)
    
    for node in nodes:

        if node.op != 'call_module':
            continue
        
        module = graph_module.get_submodule(node.target)
        observer.module_to_node_name[module] = node.name

        hook = module.register_forward_hook(observer._forward_hook)
        observer.hooks.append(hook)

    return observer


@torch.no_grad()
def calibrate(
    graph_module: fx.GraphModule,
    dataloader: Iterable,
    iters: Union[int, None]
) -> None:
    
    iters = iters or len(dataloader)
    iters = min(iters, len(dataloader))

    device = next(iter(graph_module.parameters())).device

    for i, data in enumerate(dataloader):

        _ = model_forward(graph_module, data, device)

        if i + 1 >= iters:
            break


def get_valid_name(
    prefix: str,
    named_modules: Dict[str, nn.Module]
) -> str:
    
    if prefix not in named_modules:
        return prefix
    
    i = 0
    name = prefix + '_' + str(i)
    while name in named_modules:
        i += 1
        name = prefix + '_' + str(i)

    return name


def get_abs_max(
    linear: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d],
    out_channel: bool
) -> Tensor:
    
    if isinstance(linear, nn.Linear):
        groups = 1
    else:
        groups = linear.groups
        
    size = linear.weight.size()
    out_channels, *_ = size
    dim = linear.weight.dim()

    weight = linear.weight.data.reshape(groups, out_channels // groups, *size[1:])

    if isinstance(linear, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        dims = [i for i in range(linear.weight.dim() + 1)]
        dims[1], dims[2] = 2, 1
        weight = weight.permute(dims)

    if out_channel:
        dim = [i for i in range(2, dim + 1)]
    else:
        dim = [1] + [i for i in range(3, dim + 1)]

    r = weight.abs().amax(dim=dim).flatten()
    return r


def insert_module(
    module: nn.Module,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module],
    prefix: str = 'mul'
) -> str:
    
    module_name = get_valid_name(prefix, named_modules)
    graph_module.add_module(module_name, module)
    named_modules[module_name] = module

    return module_name


def smooth_quant_helper(
    linear1: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d], 
    linear2: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d],
    activation_max_abs: Tensor,
    alpha: float,
    eps: float = 1e-8
) -> None:
    
    from equant.algorithms.cross_layer_equalization import scale_weight
    
    weight_max_abs = get_abs_max(linear2, out_channel=False)
    scale = activation_max_abs.pow(alpha) / weight_max_abs.pow(1 - alpha).clip(min=eps)
    scale = torch.clip(scale, min=1e-5)

    scale[activation_max_abs == 0] = 1.0
    # scale[weight_max_abs == 0] = 0.0

    if linear1.bias is not None:
        linear1.bias.data /= scale

    dim = linear1.weight.dim()
    for _ in range(dim - 1):
        scale = scale.unsqueeze(1)

    linear1.weight.data = scale_weight(linear1, scale, out_channel=True)
    linear2.weight.data = scale_weight(linear2, scale, out_channel=False)


def insert_identity_linear_layer(
    node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:

    linear2 = graph_module.get_submodule(node.target)
    linear2 = wrap_into_sequential(linear2)[0]

    linear1 = create_identity_layer_from_linear(linear2)
    module_name = insert_module(linear1, graph_module, named_modules)

    with graph_module.graph.inserting_before(node):
        linear1_node = graph_module.graph.call_module(module_name, args=node.args, kwargs=node.kwargs)

    node.args = (linear1_node,)


def find_smooth_quant_chain(
    node: fx.Node,
    graph_module: fx.GraphModule
) -> List[fx.Node]:

    chain = find_chain_backward(node, graph_module, patterns=[LINEAR_LAYERS, ACTIVATIONS, LINEAR_LAYERS])

    if chain is not None:
        return chain

    chain = find_chain_backward(node, graph_module, patterns=[LINEAR_LAYERS, LINEAR_LAYERS])

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
            module = wrap_into_sequential(module)[0]
            if isinstance(module, LINEAR_LAYERS) and find_smooth_quant_chain(node, graph_module) is None:
                insert_identity_linear_layer(node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module


def get_quantized_nodes(
    graph_module: fx.GraphModule,
    qconfig_mapping: QConfigMapping,
    dataloader: Iterable
) -> List[str]:
    
    qmodel = quantize(graph_module, qconfig_mapping, next(iter(dataloader)))
    named_modes = {node.name: node for node in graph_module.graph.nodes}
    
    node: fx.Node
    quantized_nodes = []
    for node in qmodel.graph.nodes:
        
        if node.op == 'call_module':

            module = qmodel.get_submodule(node.target)
            if quantized(module):
                quantized_nodes.append(named_modes[node.name])

    return quantized_nodes


@torch.no_grad()
def smooth_quant(
    model: Union[nn.Module, fx.GraphModule],
    qconfig_mapping: QConfigMapping,
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

    quantized_nodes = get_quantized_nodes(graph_module, qconfig_mapping, dataloader)

    if not absorb:
        graph_module = insert_identity_linear_layers(quantized_nodes, graph_module)

    observer = add_observers(graph_module, quantile=quantile, nodes=quantized_nodes)
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

        linear1 = wrap_into_sequential(linear1)[0]
        linear2 = wrap_into_sequential(linear2)[0]

        smooth_quant_helper(linear1, linear2, activations_max_abs[node.name], alpha=alpha)

    return graph_module


def insert_identity_quant_linear_layer(
    node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:
    
    # TODO: Fix observer insertion rule!

    linear2 = graph_module.get_submodule(node.target).to_float()

    if issubclass(type(linear2), nn.Sequential):
        linear2 = linear2[0]

    linear1 = create_identity_layer_from_linear(linear2)
    linear1.qconfig = None
    module_name = insert_module(linear1, graph_module, named_modules)

    with graph_module.graph.inserting_before(node):
        linear1_node = graph_module.graph.call_module(module_name, args=node.args, kwargs=node.kwargs)

    graph_module.meta["_observed_graph_module_attrs"].node_name_to_scope[module_name] = (linear1_node.target, type(linear1))

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

            module = wrap_into_sequential(decompose_module_to_float(module))[0]
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

        # linear1 = wrap_into_sequential(linear1)[0]
        # linear2 = wrap_into_sequential(linear2)[0]

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


@torch.no_grad()
def smooth_quant_auto_tune(
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
            
        print(chain)
        linear1 = graph_module.get_submodule(chain[0].target)
        linear2 = graph_module.get_submodule(chain[-1].target)

        # linear1 = wrap_into_sequential(linear1)[0]
        # linear2 = wrap_into_sequential(linear2)[0]

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
        print(optimal_aplha)

        smooth_quant_helper(linear1, linear2, activations_max_abs[chain[-1].name], alpha=optimal_aplha)

        fp_interpreter._run_node(fp_node)
        interpreter._run_node(node)

    graph_module.apply(disable_observer)

    del interpreter
    del fp_interpreter

    return graph_module
