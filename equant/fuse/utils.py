import torch
import torch.nn as nn
import torch.fx as fx

from typing import Dict, Tuple, Any

BNS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
LINEAR = (*CONVS, nn.Linear)


__all__ = [
    'replace_node_module'
]


def _parent_name(
    target: str
) -> Tuple[str, str]:
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(
    node: fx.Node, 
    named_modules: Dict[str, Any], 
    new_module: torch.nn.Module
) -> None:
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(named_modules[parent_name], name, new_module)
