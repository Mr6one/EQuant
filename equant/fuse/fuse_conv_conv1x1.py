import copy

import torch
import torch.nn as nn
import torch.fx as fx
from torch import Tensor

from typing import Dict, Iterable, Union

from equant.fuse.utils import _parent_name, replace_node_module, CONVS, BNS


__all__ = [
    'fuse_conv_conv1x1'
]


def all_zeros(
    array: Iterable
) -> bool:
    
    return all([value == 0 for value in array])


def all_ones(
    array: Iterable
) -> bool:
    
    return all([value == 1 for value in array])


def convolve_two_tensors_nd(
    x: Tensor, 
    kernel: Tensor,
    dim: int = 1
) -> Tensor:
    
    if dim == 1:
        padding = tuple(size - 1 for size in kernel.size()[-1:])
        x = nn.functional.conv1d(x.transpose(0, 1), kernel.flip(-1), padding=padding).transpose(0, 1)
    elif dim == 2:
        padding = tuple(size - 1 for size in kernel.size()[-2:])
        x = nn.functional.conv2d(x.transpose(0, 1), kernel.flip(-1, -2), padding=padding).transpose(0, 1)
    elif dim == 3:
        padding = tuple(size - 1 for size in kernel.size()[-3:])
        x = nn.functional.conv3d(x.transpose(0, 1), kernel.flip(-1, -2, -3), padding=padding).transpose(0, 1)
    else:
        raise ValueError(f'Unsupported value for dim={dim}. Supports only dimensions from one to three.')

    return x


def fuse_two_convnd(
    conv1: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], 
    conv2: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    dim: int = 1
) -> Union[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], None]:

    '''
    Fuses two consecutive convolutions into one. 
    Supports convolutions stride=1, dilations=1. 
    The first convolution may have non-zero padding and groups > 1 while the second one should have zero-padding and groups=1.

    Returns:
        If fuse is possible returns fused convolution else returns None.
    '''

    if dim not in [1, 2, 3]:
        raise ValueError(f'Unsupported value for dim={dim}. Supports only dimensions from one to three.')

    if not all_ones(conv1.stride) or not all_ones(conv1.dilation):
        return
    
    if not all_ones(conv2.stride) or not all_zeros(conv2.padding) or \
        not all_ones(conv2.dilation) or conv2.groups != 1:
        return

    weight = convolve_two_tensors_nd(conv1.weight.data, conv2.weight.data, dim=dim)

    bias = torch.zeros(conv2.weight.size(0))
    if conv1.bias is not None:

        bias_data = conv1.bias.data
        for _ in range(dim + 1):
            bias_data = bias_data.unsqueeze(1)

        bias = convolve_two_tensors_nd(bias_data, conv2.weight.data, dim=dim).sum(dim=tuple(-d for d in range(1, dim + 1))).flatten()

    if conv2.bias is not None:
        bias = bias + conv2.bias

    kernel_size = (kernel_size1 + kernel_size2 - 1 for kernel_size1, kernel_size2 in zip(conv1.kernel_size, conv2.kernel_size))
    padding = (padding1 + padding2 for padding1, padding2 in zip(conv1.padding, conv2.padding))

    args = {
        'in_channels': conv1.in_channels, 
        'out_channels': conv2.out_channels, 
        'kernel_size': kernel_size, 
        'groups': conv1.groups, 
        'padding': padding
    }

    if dim == 1:
        conv = nn.Conv1d(**args)
    elif dim == 2:
        conv = nn.Conv2d(**args)
    else:
        conv = nn.Conv3d(**args)

    conv.weight.data = weight
    conv.bias.data = bias

    return conv


def fuse_two_conv1d(
    conv1: nn.Conv1d, 
    conv2: nn.Conv1d,
) -> Union[nn.Conv1d, None]:
    return fuse_two_convnd(conv1, conv2, dim=1)


def fuse_two_conv2d(
    conv1: nn.Conv2d, 
    conv2: nn.Conv2d,
) -> Union[nn.Conv2d, None]:
    return fuse_two_convnd(conv1, conv2, dim=2)


def fuse_two_conv3d(
    conv1: nn.Conv3d, 
    conv2: nn.Conv3d,
) -> Union[nn.Conv3d, None]:
    return fuse_two_convnd(conv1, conv2, dim=3)


def fuse_two_convs_helper(
    conv1_node: fx.Node,
    conv2_node: fx.Node,
    graph_module: fx.GraphModule,
    named_modules: Dict[str, nn.Module]
) -> None:
    
    conv1 = named_modules[conv1_node.target]
    conv2 = named_modules[conv2_node.target]
    conv_fused = {
        nn.Conv1d: fuse_two_conv1d, 
        nn.Conv2d: fuse_two_conv2d, 
        nn.Conv3d: fuse_two_conv3d
    }[type(conv1)](conv1, conv2)

    if conv_fused is None:
        return

    named_modules[conv1_node.target] = conv_fused
    replace_node_module(conv1_node, named_modules, conv_fused)
    conv2_node.replace_all_uses_with(conv1_node)
    graph_module.graph.erase_node(conv2_node)
    
    parent_name, name = _parent_name(conv2_node.target)
    delattr(named_modules[parent_name], name)


def fuse_conv_conv1x1(
    model: Union[nn.Module, fx.GraphModule],
    inplace: bool = False
) -> fx.GraphModule:
    
    '''
    Fuse conv1x1(conv(x))
    '''
    
    if not inplace:
        model = copy.deepcopy(model)

    if not isinstance(model, fx.GraphModule):
        graph_module = fx.symbolic_trace(model)
    else:
        graph_module = model

    named_modules = dict(graph_module.named_modules())

    node: fx.Node
    for node in graph_module.graph.nodes:
        
        if node.op == 'call_module' and isinstance(named_modules[node.target], CONVS):
            
            arg = node.args[0]

            if arg.op == 'call_module' and isinstance(named_modules[arg.target], CONVS):
                
                if all_ones(named_modules[node.target].kernel_size):
                    fuse_two_convs_helper(arg, node, graph_module, named_modules)

    graph_module.graph.lint()
    graph_module.recompile()

    return graph_module
