import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.ao.quantization.fake_quantize import FakeQuantize

from typing import Iterable, List, Dict, Tuple, Union
from equant.core.feature_extractor import model_forward
from equant.core.search import find_chain_forward, quantized


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

    linear = linear.to(module.weight.device)
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
        input: Tensor = input[0].detach().transpose(0, 1).flatten(start_dim=1)

        min_val = input.quantile(1 - self.quantile, dim=1)
        max_val = input.quantile(self.quantile, dim=1)

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
            min_val = torch.stack(self.min_vals[node_name]).min(dim=0).values
            max_val = torch.stack(self.max_vals[node_name]).max(dim=0).values
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


def find_smooth_quant_chain_with_quantizers_forward(
    node: fx.Node,
    graph_module: fx.GraphModule
) -> List[fx.Node]:

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, ACTIVATIONS, QUANTIZERS, LINEAR_LAYERS])

    if chain is not None and quantized(graph_module.get_submodule(chain[-1].target)):
        return chain

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, QUANTIZERS, LINEAR_LAYERS])

    if chain is not None and quantized(graph_module.get_submodule(chain[-1].target)):
        return chain
    
    return None
