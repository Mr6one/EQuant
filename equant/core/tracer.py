# NOTE: subject of deprication as QTracer is only used by create_qfeature_extractor function
# which we replaced by more stable create_feature_extractor in feature_extractor.py

import torch
import torch.nn as nn
import torch.fx as fx

import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
from torch.ao.quantization.fake_quantize import FakeQuantize
from torchvision.models.feature_extraction import NodePathTracer, _set_default_tracer_kwargs, deepcopy, re, OrderedDict, _warn_graph_differences, DualGraphModule

from typing import Optional, Union, Dict, List, Any


class QTracer(NodePathTracer):

    '''
    Tracer for quantized models 
    '''

    @fx._compatibility.compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:

        return (
            (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
            and not isinstance(m, torch.nn.Sequential) or \
                # don't trace fused modules, qat modules and quantizers
                isinstance(m, (nni._FusedModule, nnqat.modules.linear.Linear, nnqat.modules.conv._ConvNd, 
                                     nnqat.modules.embedding_ops.Embedding, FakeQuantize)) 
        )
    

def qtrace(
    model: nn.Module
) -> fx.GraphModule:
    tracer = QTracer()
    graph = tracer.trace(model)
    name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
    graph_module = fx.GraphModule(tracer.root, graph, name)
    return graph_module
    

def create_qfeature_extractor(
    model: nn.Module,
    return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_diff_warning: bool = False,
) -> fx.GraphModule:
    
    '''
    Feature extractor adopted for quantized models 
    '''

    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training

    if all(arg is None for arg in [return_nodes, train_return_nodes, eval_return_nodes]):

        raise ValueError(
            "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"
        )

    if (train_return_nodes is None) ^ (eval_return_nodes is None):
        raise ValueError(
            "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"
        )

    if not ((return_nodes is None) ^ (train_return_nodes is None)):
        raise ValueError("If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified")

    # Put *_return_nodes into Dict[str, str] format
    def to_strdict(n) -> Dict[str, str]:
        if isinstance(n, list):
            return {str(i): str(i) for i in n}
        return {str(k): str(v) for k, v in n.items()}

    if train_return_nodes is None:
        return_nodes = to_strdict(return_nodes)
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)
    else:
        train_return_nodes = to_strdict(train_return_nodes)
        eval_return_nodes = to_strdict(eval_return_nodes)

    # Repeat the tracing and graph rewriting for train and eval mode
    tracers = {}
    graphs = {}
    mode_return_nodes: Dict[str, Dict[str, str]] = {"train": train_return_nodes, "eval": eval_return_nodes}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = QTracer(**tracer_kwargs)
        graph = tracer.trace(model)

        name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        graph_module = fx.GraphModule(tracer.root, graph, name)

        available_nodes = list(tracer.node_to_qualname.values())
        # FIXME We don't know if we should expect this to happen
        if len(set(available_nodes)) != len(available_nodes):
            raise ValueError(
                "There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues"
            )
        # Check that all outputs in return_nodes are present in the model
        for query in mode_return_nodes[mode].keys():
            # To check if a query is available we need to check that at least
            # one of the available names starts with it up to a .
            if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
                raise ValueError(
                    f"node: '{query}' is not present in model. Hint: use "
                    "`get_graph_node_names` to make sure the "
                    "`return_nodes` you specified are present. It may even "
                    "be that you need to specify `train_return_nodes` and "
                    "`eval_return_nodes` separately."
                )

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
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                # NOTE - Know cases where this happens:
                # - Node representing creation of a tensor constant - probably
                #   not interesting as a return node
                # - When packing outputs into a named tuple like in InceptionV3
                continue
            for query in mode_return_nodes[mode]:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:
                    output_nodes[mode_return_nodes[mode][query]] = n
                    mode_return_nodes[mode].pop(query)
                    break
        output_nodes = OrderedDict(reversed(list(output_nodes.items())))

        # And add them in the end of the graph
        with graph_module.graph.inserting_after(nodes[-1]):
            graph_module.graph.output(output_nodes)

        # Remove unused modules / parameters
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Keep track of the tracer and graph, so we can choose the main one
        tracers[mode] = tracer
        graphs[mode] = graph

    # Warn user if there are any discrepancies between the graphs of the
    # train and eval modes
    if not suppress_diff_warning:
        _warn_graph_differences(tracers["train"], tracers["eval"])

    # Build the final graph module
    graph_module = DualGraphModule(model, graphs["train"], graphs["eval"], class_name=name)

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module
