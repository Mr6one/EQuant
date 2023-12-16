import copy

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.fx as fx

from typing import List, Union

from equant.core.match import decompose_module, find_chain_forward


__all__ = [
    'cross_layer_equalization'
]


LINEAR_LAYERS = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
)

ACTIVATIONS = (nn.ReLU, F.relu)


def get_range(
    linear: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d],
    out_channel: bool
) -> Tensor:
    
    if issubclass(type(linear), nn.Linear):
        groups = 1
    else:
        groups = linear.groups
        
    size = linear.weight.size()
    out_channels, *_ = size
    dim = linear.weight.dim()

    weight = linear.weight.data.reshape(groups, out_channels // groups, *size[1:])

    if issubclass(type(linear), (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        weight = weight.transpose(1, 2)

    if out_channel:
        dim = [i for i in range(2, dim + 1)]
    else:
        dim = [1] + [i for i in range(3, dim + 1)]

    r = (weight.amax(dim=dim) - weight.amin(dim=dim)).flatten()
    return r


def scale_weight(
    linear: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d], 
    scale: Tensor, 
    out_channel: bool
) -> Tensor:
    
    if issubclass(type(linear), nn.Linear):
        groups = 1
    else:
        groups = linear.groups

    size = linear.weight.size()
    out_channels, *_ = size

    weight = linear.weight.data.reshape(groups, out_channels // groups, *size[1:])
    scale = scale.reshape(groups, -1, *scale.size()[1:])

    if issubclass(type(linear), (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        weight = weight.transpose(1, 2)

    if out_channel:
        scale = 1 / scale
    else:
        scale = scale.transpose(1, 2)

    weight = weight * scale.to(weight.device)

    if issubclass(type(linear), (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        weight = weight.transpose(1, 2)

    weight = weight.reshape(*size)

    return weight


def cross_layer_equalization_helper(
    linear1: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d], 
    linear2: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
) -> None:
    
    r1 = get_range(linear1, out_channel=True)
    r2 = get_range(linear2, out_channel=False)

    scale = (r1 / r2).sqrt()

    if linear1.bias is not None:
        linear1.bias.data /= scale

    dim = linear1.weight.dim()
    for _ in range(dim - 1):
        scale = scale.unsqueeze(1)

    linear1.weight.data = scale_weight(linear1, scale, out_channel=True)
    linear2.weight.data = scale_weight(linear2, scale, out_channel=False)


def find_cle_chain(
    node: fx.Node,
    graph_module: fx.GraphModule
) -> Union[None, List[fx.Node]]:

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, ACTIVATIONS, LINEAR_LAYERS])

    if chain is not None:
        return chain

    chain = find_chain_forward(node, graph_module, patterns=[LINEAR_LAYERS, LINEAR_LAYERS])

    if chain is not None:
        return chain
    
    return None


def cross_layer_equalization(
    model: Union[nn.Module, fx.GraphModule],
    inplace: bool = False
) -> fx.GraphModule:
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    node: fx.Node
    for node in graph_module.graph.nodes:
        
        chain = find_cle_chain(node, graph_module)

        if chain is None:
            continue
        
        linear1 = graph_module.get_submodule(chain[0].target)
        linear2 = graph_module.get_submodule(chain[-1].target)

        linear1 = decompose_module(linear1)[0]
        linear2 = decompose_module(linear2)[0]

        cross_layer_equalization_helper(linear1, linear2)

    return graph_module
