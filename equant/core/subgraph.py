import copy
import torch.nn as nn
import torch.fx as fx
from typing import Any, List, Tuple, Dict, Set, Union


def add_node_inputs(
    node: fx.Node, 
    subgraph_edges: Dict[str, Set[str]],
    input_nodes: Set[str]
) -> None:
    
    for arg in list(node.args) + list(node.kwargs.values()):
        def add_arg(arg):
            
            if isinstance(arg, fx.Node):
                subgraph_edges[node].add(arg)
                if arg.op != 'get_attr' and arg.op != 'output' and arg not in subgraph_edges:
                    input_nodes.add(arg.name)

            elif isinstance(arg, (tuple, list)):
                for subarg in arg:
                    add_arg(subarg)

        add_arg(arg)


def add_node_outputs(
    node: fx.Node, 
    subgraph_edges: Dict[str, Set],
    output_nodes: Set[str]
) -> None:
    
    for output in node.users:
        if output not in subgraph_edges:
            if node.op != 'get_attr':
                output_nodes.add(node.name)
        else:
            subgraph_edges[output].add(node)


def subgraph_structure(
    graph_module: fx.GraphModule,
    node_names: List[str]
) -> Tuple[Set[str], Set[str], Dict[str, Set[str]]]:
    
    named_nodes = {node.name: node for node in graph_module.graph.nodes}
    subgraph_edges = {named_nodes[node_name]: set() for node_name in node_names}

    input_nodes = set()
    output_nodes = set()

    for node in subgraph_edges:
        add_node_inputs(node, subgraph_edges, input_nodes)
        add_node_outputs(node, subgraph_edges, output_nodes)

    if len(input_nodes) < 1:
        raise RuntimeError(f"Subgraph doesn't have any inputs")
    
    if len(output_nodes) < 1:
        raise RuntimeError(f"Subgraph doesn't return anything")

    return input_nodes, output_nodes, subgraph_edges


def insert_input_nodes(
    subgraph: fx.GraphModule, 
    input_nodes: Set[str]
) -> Dict:

    visited = {}
    for input_node in input_nodes:
        node = subgraph.placeholder(input_node)
        visited[input_node] = node

    return visited


def fetch_attr(
    module: Union[nn.Module, fx.GraphModule], 
    target: str
) -> Any:
    target_atoms = target.split('.')
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def add_nodes_to_subgraph_module(
    node: fx.Node, 
    subgraph: fx.Graph, 
    subgraph_edges: Dict[fx.Node, Set[fx.Node]], 
    subgraph_named_modules: Dict[str, nn.Module], 
    graph_module: fx.GraphModule,
    visited: Dict[str, fx.Node]
) -> None:

    next_node: fx.Node
    for next_node in subgraph_edges[node]:

        if next_node.name not in visited:

            add_nodes_to_subgraph_module(
                next_node, 
                subgraph, 
                subgraph_edges, 
                subgraph_named_modules, 
                graph_module, 
                visited
            )

    node = subgraph.node_copy(node, lambda n : visited[n.name])

    if node.op == 'call_module':
        subgraph_named_modules[node.target] = copy.deepcopy(graph_module.get_submodule(node.target))
    elif node.op == 'get_attr':
        subgraph_named_modules[node.target] = fetch_attr(graph_module, node.target)
    else:
        pass
        
    visited[node.name] = node


def insert_intermediate_nodes(
    subgraph: fx.GraphModule, 
    graph_module: fx.GraphModule,
    subgraph_edges: Dict[fx.Node, Set[fx.Node]], 
    visited: Dict[str, fx.Node]
) -> Dict:
    
    subgraph_named_modules = {}
    node: fx.Node
    for node in subgraph_edges:
        
        if node.name not in visited:

            add_nodes_to_subgraph_module(
                node, 
                subgraph, 
                subgraph_edges, 
                subgraph_named_modules, 
                graph_module,
                visited
            )

    return subgraph_named_modules


def insert_output_nodes(
    subgraph: fx.GraphModule, 
    output_nodes: Set[str], 
    visited: Dict[str, fx.Node]
) -> None:
    
    last_node = list(subgraph.nodes)[-1]
    subgraph_result_nodes = tuple(visited[node_name] for node_name in output_nodes)
    with subgraph.inserting_after(last_node):

        if len(subgraph_result_nodes) == 1:
            subgraph_result_nodes = next(iter(subgraph_result_nodes))

        subgraph.output(result=subgraph_result_nodes)


def create_subgraph(
    graph_module: fx.GraphModule, 
    node_names: List[str]
) -> fx.GraphModule:
    
    input_nodes, output_nodes, edges = subgraph_structure(graph_module, node_names)
    
    subgraph = fx.Graph()

    visited = insert_input_nodes(subgraph, input_nodes)
    subgraph_named_modules = insert_intermediate_nodes(subgraph, graph_module, edges, visited)
    insert_output_nodes(subgraph, output_nodes, visited)

    subgraph.lint()
    subgraph_module = fx.GraphModule(subgraph_named_modules, subgraph)
    subgraph_module.graph.eliminate_dead_code()

    return subgraph_module
