from collections import defaultdict
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.fx as fx
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Iterable, List, Tuple, Union

from equant.core.match import has_bn, quantized, wrap_into_sequential
from equant.core.subgraph import create_subgraph, collect_inputs_outputs_for_subgraph, model_forward
from equant.core.quantizers.fake_quantize import FakeQuantizeTorchBase, disable_fake_quant, enable_fake_quant, disable_observer
from equant.core.interpreter import DataInterpreter


__all__ = [
    'adaround'
]


class AdaRoundFakeQuant(nn.Module):

    def __init__(
            self, 
            weight: nn.Parameter, 
            scale: Tensor, 
            zero_point: Tensor, 
            quant_min: Union[int, float], 
            quant_max: Union[int, float],
            gamma: float = -0.1,
            zeta: float = 1.1
        ) -> None:

        super().__init__()

        self.scale = scale.clip(1e-10).detach()
        self.quant_min = int(quant_min)
        self.quant_max = int(quant_max)

        max_zero_point = int(quant_max - quant_min)
        self.zero_point = zero_point.round().clip(0, max_zero_point)

        self.gamma = gamma
        self.zeta = zeta

        hidden = self._init_v(weight, self.scale, gamma, zeta)
        self.hidden = nn.Parameter(hidden)

    @staticmethod
    def _init_v(weight: Tensor, scale: Tensor, gamma: float, zeta: float) -> Tensor:
        weight_round = (weight / scale)
        weight_floor = (weight / scale).floor()
        bound = weight_round - weight_floor
        return -torch.log((zeta - gamma) / (bound - gamma) - 1)
    
    @staticmethod
    def _rectified_sigmoid(x: Tensor, zeta: float, gamma: float) -> Tensor:
        x = torch.clip(torch.sigmoid(x) * (zeta - gamma) + gamma, 0, 1)
        return x

    def forward(self, weight: Tensor, hard_round: bool = False) -> Tensor:
        hidden: Tensor = self._rectified_sigmoid(self.hidden, self.zeta, self.gamma)

        if hard_round:
            hidden = hidden.round()
        
        weight = (weight / self.scale).floor() + self.zero_point + hidden
        weight = self.scale * (weight.clip(self.quant_min, self.quant_max) - self.zero_point)
        return weight
    

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
        
        inputs = [model_forward(module, input, device) for input in inputs]

        if start <= i < end:
            modules_outputs.append(inputs)

    if len(modules_inputs) == 1:
        modules_inputs = modules_inputs[0]

    if len(modules_outputs) == 1:
        modules_outputs = modules_outputs[0]

    return modules_inputs, modules_outputs


def compute_reconstruction_loss(
    quant_output: Union[Tensor, List[Tensor]],
    fp_output: Union[Tensor, List[Tensor]]
) -> Tensor:
    
    if isinstance(quant_output, Tensor):
        quant_output = [quant_output]

    if isinstance(fp_output, Tensor):
        fp_output = [fp_output]
    loss = 0
    for out1, out2 in zip(quant_output, fp_output):
        loss = loss + F.mse_loss(out1, out2)

    loss = loss / len(quant_output)

    return loss


def compute_beta(
    max_iter: int, 
    curr_iter: int, 
    beta_range: Tuple, 
    warm_start: float
) -> float:
    
    if curr_iter >= max_iter:
        curr_iter = max_iter - 1

    start_beta, end_beta = beta_range
    warm_start_end_iter = warm_start * max_iter
    rel_iter = (curr_iter - warm_start_end_iter) / (max_iter - warm_start_end_iter)
    beta = end_beta + 0.5 * (start_beta - end_beta) * (1 + np.cos(rel_iter * np.pi))

    return beta


def compute_round_loss(
    fake_quant: AdaRoundFakeQuant, 
    curr_iter: int, 
    num_iters: int, 
    warm_start: float, 
    beta_range: Tuple, 
    reg_param: float
) -> torch.Tensor:

    if curr_iter < num_iters * warm_start:
        round_loss = torch.tensor(0.0)
    else:
        alpha = fake_quant._rectified_sigmoid(fake_quant.hidden, fake_quant.zeta, fake_quant.gamma)
        beta = compute_beta(num_iters, curr_iter, beta_range, warm_start)

        reg_term = (1 - (2 * alpha - 1).abs().pow(beta)).sum()
        round_loss = reg_param * reg_term

    return round_loss


def adaround_loss(
    quant_output: Union[Tensor, List[Tensor]],
    fp_output: Union[Tensor, List[Tensor]],
    fake_quant: AdaRoundFakeQuant,
    curr_iter: int,
    num_iters: int, 
    warm_start: float, 
    beta_range: Tuple, 
    reg_param: float
) -> Tensor:

    reconstruction_loss = compute_reconstruction_loss(quant_output, fp_output)
    round_loss = compute_round_loss(fake_quant, curr_iter, num_iters, warm_start, beta_range, reg_param)
    loss = reconstruction_loss + round_loss
    return loss, reconstruction_loss, round_loss


def optimize_module(
    module: nn.Module, 
    quant_inputs: Any, 
    fp_outputs: Any,
    num_iters: int, 
    lr: float,
    warm_start: float, 
    beta_range: Tuple, 
    reg_param: float,
    device: torch.device
) -> None:

    if not hasattr(module, 'weight_fake_quant'):
        raise RuntimeError('Optimized module should provide fakequant rule for weights')

    weight_fake_quant = module.weight_fake_quant
    module.weight_fake_quant = AdaRoundFakeQuant(
        module.weight, # TODO: optimize only linear layers to avoid error if weight attribute isn't defined?
        weight_fake_quant.scale.detach(), 
        weight_fake_quant.zero_point.detach(),
        weight_fake_quant.activation_post_process.quant_min,
        weight_fake_quant.activation_post_process.quant_max
    )

    optimizer = torch.optim.Adam(module.weight_fake_quant.parameters(), lr=lr)

    curr_iter = 0
    total_loss = defaultdict(int)
    while True:
        
        for quant_input, fp_output in zip(quant_inputs, fp_outputs):
            optimizer.zero_grad()
            quant_output = model_forward(module, quant_input, device)
            loss, reconstruction_loss, round_loss = adaround_loss(
                quant_output, 
                fp_output.to(device), 
                module.weight_fake_quant, 
                curr_iter, 
                num_iters, 
                warm_start, 
                beta_range, 
                reg_param
            )
            loss.backward()
            optimizer.step()

            curr_iter += 1
            total_loss['loss'] += loss.item()
            total_loss['reconstruction_loss'] += reconstruction_loss.item()
            total_loss['round_loss'] += round_loss.item()

            if (curr_iter + 1) % 100 == 0:
                
                avg_total_loss = total_loss['loss'] / 100
                avg_reconstruction_loss = total_loss['reconstruction_loss'] / 100
                avg_round_loss = total_loss['round_loss'] / 100
                
                print(f'num_iter: {curr_iter + 1}/{num_iters}, avg_total_loss: {avg_total_loss}, avg_reconstruction_loss: {avg_reconstruction_loss}, avg_round_loss: {avg_round_loss}')
                
                total_loss = defaultdict(int)

            if curr_iter >= num_iters:
                module.weight.data = module.weight_fake_quant(module.weight.data, hard_round=True)
                module.weight_fake_quant = weight_fake_quant
                return


def adaround(
    graph_module: fx.GraphModule,
    dataloader: Iterable,
    num_iters: int = 10000, 
    lr: float = 1e-3,
    warm_start: float = 0.2, 
    beta_range: Tuple = (20, 2), 
    reg_param: float = 0.01,
    inplace: bool = False,
    cache_data: bool = False
) -> fx.GraphModule:
    
    # TODO: Add progress bar
    
    if not inplace:
        graph_module = copy.deepcopy(graph_module)

    device = next(iter(graph_module.parameters())).device

    graph_module.apply(enable_fake_quant)
    fp_graph_module = copy.deepcopy(graph_module).apply(disable_fake_quant).apply(disable_observer)

    interpreter = DataInterpreter(graph_module, cache_data=cache_data)
    interpreter.initialize_env(dataloader)

    fp_interpreter = DataInterpreter(fp_graph_module, cache_data=cache_data)
    fp_interpreter.initialize_env(dataloader)

    graph_module.to(device)
    fp_graph_module.to(device)

    node: fx.Node
    fp_node: fx.Node
    for fp_node, node in zip(fp_graph_module.graph.nodes, graph_module.graph.nodes):

        if node.op == 'call_module':
            module = graph_module.get_submodule(node.target)

            if quantized(module):

                if has_bn(wrap_into_sequential(module)):
                    with torch.no_grad():
                        interpreter._run_node(node)
                        fp_interpreter._run_node(fp_node)
                    continue
                
                fp_module = fp_graph_module.get_submodule(fp_node.target)
                
                quant_inputs, _ = collect_decomposed_module_inputs_outputs([module], interpreter.env[node.args[0]], device, start=0, end=1)
                _, fp_outputs = collect_decomposed_module_inputs_outputs([fp_module], fp_interpreter.env[fp_node.args[0]], device, start=0, end=1)

                print(f'Optimizing {node.target}')
                module.apply(disable_observer)
                optimize_module(module, quant_inputs, fp_outputs, num_iters, lr, warm_start, beta_range, reg_param, device=device)
                print('-' * 50)

        with torch.no_grad():
            interpreter._run_node(node)
            fp_interpreter._run_node(fp_node)

    del interpreter
    del fp_interpreter

    return graph_module
