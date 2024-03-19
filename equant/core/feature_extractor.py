import copy
import torch
import torch.nn as nn
import torch.fx as fx
from torch import Tensor

from collections import OrderedDict
from typing import Any, Iterable, List, Tuple, Dict, Union

from equant.core.tracer import QTracer, create_qfeature_extractor


__all__ = [
    'model_forward',
    'collect_inputs_outputs_for_subgraph'
]


def model_forward(
    model: nn.Module, 
    data: Union[Tensor, List, Tuple, Dict], 
    device: torch.device
) -> Any:

    if isinstance(data, (list, tuple)):
        data = [d.to(device) for d in data]
        data = model(*data)
    elif isinstance(data, dict):
        data = {k: v.to(device) for k, v in data.items()}
        data = model(**data)
    elif isinstance(data, Tensor):
        data = data.to(device)
        data = model(data)
    else:
        raise RuntimeError(f'Unsupported input of type {type(data)}')
    
    return data


def create_feature_extractor(
    graph_module: fx.GraphModule,
    return_nodes: List[str]
) -> fx.GraphModule:
    
    graph_module = copy.deepcopy(graph_module)

    def to_strdict(n) -> Dict[str, str]:
        if isinstance(n, list):
            return {str(i): str(i) for i in n}
        return {str(k): str(v) for k, v in n.items()}

    return_nodes = to_strdict(return_nodes)

    # Remove existing output nodes (train mode)
    orig_output_nodes = []
    for n in reversed(graph_module.graph.nodes):
        if n.op == "output":
            orig_output_nodes.append(n)
    if not orig_output_nodes:
        raise ValueError("No output nodes found in graph_module.graph.nodes")

    for n in orig_output_nodes:
        graph_module.graph.erase_node(n)

    # Find nodes corresponding to return_nodes and make them into output_nodes
    nodes = [n for n in graph_module.graph.nodes]
    output_nodes = OrderedDict()
    for n in reversed(nodes):
        for query in return_nodes:
            if n.name == query:
                output_nodes[query] = n
                return_nodes.pop(query)
                break
    output_nodes = OrderedDict(reversed(list(output_nodes.items())))

    # And add them in the end of the graph
    with graph_module.graph.inserting_after(nodes[-1]):
        graph_module.graph.output(output_nodes)

    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()

    return graph_module


@torch.no_grad()
def collect_inputs_outputs_for_subgraph(
    model: fx.GraphModule, 
    subgraph: fx.GraphModule, 
    dataloader: Iterable
) -> Any:
    
    input_nodes = [node.name for node in subgraph.graph.nodes if node.op == 'placeholder']
    output_nodes = []

    for node in subgraph.graph.nodes:
        if node.op == 'output':
            if isinstance(node.args[0], (tuple, list)):
                for arg in node.args[0]:
                    output_nodes.append(arg.name)
            else:
                output_nodes.append(node.args[0].name)

    return_nodes = input_nodes + output_nodes
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    device = next(iter(model.parameters())).device

    inputs = []
    outputs = []

    for data in dataloader:

        data = model_forward(feature_extractor, data, device)
        
        data_inputs = []
        data_outputs = []

        for k, v in data.items():

            v = v.detach().cpu()

            if k in output_nodes:
                data_outputs.append(v)
            
            if k in input_nodes:
                data_inputs.append(v)

        if len(data_inputs) == 1:
            data_inputs = data_inputs[0]

        if len(data_outputs) == 1:
            data_outputs = data_outputs[0]

        inputs.append(data_inputs)
        outputs.append(data_outputs)

    return inputs, outputs


# NOTE: subject of deprication, remove this for release
@torch.no_grad()
def __collect_inputs_outputs_for_subgraph(
    model: fx.GraphModule, 
    subgraph: fx.GraphModule, 
    dataloader: Iterable
) -> Any:
    
    input_nodes = [node.name for node in subgraph.graph.nodes if node.op == 'placeholder']
    output_nodes = []

    for node in subgraph.graph.nodes:
        if node.op == 'output':
            if isinstance(node.args[0], (tuple, list)):
                for arg in node.args[0]:
                    output_nodes.append(arg.name)
            else:
                output_nodes.append(node.args[0].name)

    return_nodes = input_nodes + output_nodes
    tracer = QTracer()
    tracer.trace(model)

    '''We assume 1 to 1 nodes mapping before and after tracing'''
    # -----------------------------------------------------------------------------------------------------------------------
    if len(model.graph.nodes) - 1 != len(tracer.node_to_qualname):
        raise RuntimeError('Feature extractor has incompatible number of nodes')

    node_to_qualname = {node.name: qualname for node, qualname in zip(model.graph.nodes, tracer.node_to_qualname.values())}
    qualname_to_node = {qualname: node.name for node, qualname in zip(model.graph.nodes, tracer.node_to_qualname.values())}
    # -----------------------------------------------------------------------------------------------------------------------

    return_nodes = [node_to_qualname[node] for node in return_nodes]
    feature_extractor = create_qfeature_extractor(model, return_nodes=return_nodes)

    device = next(iter(model.parameters())).device

    inputs = []
    outputs = []

    for data in dataloader:

        data = model_forward(feature_extractor, data, device)
        
        data_inputs = []
        data_outputs = []

        for k, v in data.items():

            k = qualname_to_node[k]
            v = v.detach().cpu()

            if k in output_nodes:
                data_outputs.append(v)
            
            if k in input_nodes:
                data_inputs.append(v)

        if len(data_inputs) == 1:
            data_inputs = data_inputs[0]

        if len(data_outputs) == 1:
            data_outputs = data_outputs[0]

        inputs.append(data_inputs)
        outputs.append(data_outputs)

    return inputs, outputs