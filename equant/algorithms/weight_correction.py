import copy
import warnings
from torch import Tensor
import torch.nn as nn
import torch.fx as fx
from torch.hub import tqdm

from typing import List

from equant.core.match import has_bn, quantized
from equant.core.match.decompose import _decompose_module, _decompose_quant_module


__all__ = [
    'weight_correction'
]


TRANSPOSED_LAYERS = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

LINEAR_LAYERS = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    *TRANSPOSED_LAYERS
)


def weight_correction_helper(
    fp_weight: Tensor,
    quant_weight: Tensor,
    transpose: bool = False
) -> Tensor:
    
    if not transpose:
        fp_weight = fp_weight.transpose(0, 1)
        quant_weight = quant_weight.transpose(0, 1)

    channel_variance: Tensor = fp_weight.flatten(start_dim=1).std(dim=1) / \
        quant_weight.flatten(start_dim=1).std(dim=1).clip(1e-5)

    for _ in range(fp_weight.dim() - 1):
        channel_variance = channel_variance.unsqueeze(1)

    variance_q_weight = quant_weight * channel_variance

    channel_mean = fp_weight.flatten(start_dim=1).mean(dim=1) - \
        variance_q_weight.flatten(start_dim=1).mean(dim=1)
    
    for _ in range(fp_weight.dim() - 1):
        channel_mean = channel_mean.unsqueeze(1)

    corrected_weight = channel_variance * fp_weight + channel_mean

    if not transpose:
        corrected_weight = corrected_weight.transpose(0, 1)

    return corrected_weight


def create_execution_plan(
    graph_module: fx.GraphModule
) -> List[nn.Module]:
    
    modules2optimize = []
    node: fx.Node
    for node in graph_module.graph.nodes:

        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)

            if quantized(module):

                modules = _decompose_module(module)

                if has_bn(modules):
                    warnings.warn(f'{module} optimization will be skipped as it contains \
                                  batch normalization module. Consider using batchnorm fuse')
                    continue

                if not isinstance(modules[0], LINEAR_LAYERS):
                    continue

                modules2optimize.append(module)

    return modules2optimize


def weight_correction(
    graph_module: fx.GraphModule,
    inplace: bool = False,
    verbose: bool = True
) -> fx.GraphModule:
    
    if not inplace:
        graph_module = copy.deepcopy(graph_module)

    modules2optimize = create_execution_plan(graph_module)
    pbar = tqdm(total=len(modules2optimize), desc='weight correction', initial=0, position=0, leave=True, disable=not verbose, delay=0)
    for module in modules2optimize:
        pbar.update(1)
        modules = _decompose_quant_module(module)
        fp_weight = modules[0].weight.data
        quant_weight = modules[0].weight_fake_quant(fp_weight)

        module.weight.data = weight_correction_helper(
            fp_weight, 
            quant_weight, 
            transpose=isinstance(modules[0], TRANSPOSED_LAYERS)
        )

    return graph_module
