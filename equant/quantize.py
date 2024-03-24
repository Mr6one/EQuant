import torch
import torch.fx as fx
from typing import Union

from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig, ConvertCustomConfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from typing import Tuple, Dict, Any

from equant.core.register import get_backend_config


__all__ = [
    'quantize',
    'convert'
]


def quantize(
    model: torch.nn.Module,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
    example_inputs: Tuple[Any, ...],
    prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> fx.GraphModule:
    
    if backend_config is None:
        backend_config = get_backend_config()

    qmodel = prepare_qat_fx(model, qconfig_mapping, example_inputs, prepare_custom_config, backend_config=backend_config)

    return qmodel


def convert(
    graph_module: fx.GraphModule,
    convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
    _remove_qconfig: bool = True,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> fx.GraphModule:
    
    if backend_config is None:
        backend_config = get_backend_config()

    graph_module.cpu()
    qmodel = convert_fx(graph_module, convert_custom_config, _remove_qconfig, qconfig_mapping, backend_config=backend_config)

    return qmodel
