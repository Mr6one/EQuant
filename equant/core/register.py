import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.ao.nn.quantized.reference as nnqr
from torch.ao.quantization.fx._lower_to_native_backend import STATIC_LOWER_FUSED_MODULE_MAP
from torch.ao.quantization.backend_config import BackendPatternConfig, DTypeConfig, ObservationType, BackendConfig
from torch.ao.quantization.backend_config.native import get_native_backend_config

from equant.core.modules import fuse, qat, quant


weighted_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float
)


def register_module_fuse(fuse_sequence, fuse_module, qat_module, quant_module, quant_base, dtype_config, backend_config=None):

    if backend_config is None:
        backend_config = BackendConfig()

    fused_sequence_config = BackendPatternConfig(fuse_sequence) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(dtype_config) \
    .set_fused_module(fuse_module) \
    .set_fuser_method(fuse_module.fuser_method)

    fused_module_config = BackendPatternConfig(fuse_module) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(dtype_config) \
    .set_qat_module(qat_module)

    qat_fused_module_config = BackendPatternConfig(qat_module) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(dtype_config) \
    .set_root_module(fuse_sequence[0]) \
    .set_reference_quantized_module(quant_base) 

    STATIC_LOWER_FUSED_MODULE_MAP[fuse_module] = (quant_base, quant_module)

    backend_config.set_backend_pattern_configs([fused_sequence_config, fused_module_config, qat_fused_module_config])

    return backend_config


def register_add_fuse(activation, backend_config=None):

    if backend_config is None:
        backend_config = BackendConfig()

    add_config = BackendPatternConfig((operator.add, activation)) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(DTypeConfig(input_dtype=torch.quint8, output_dtype=torch.quint8)) \
    ._set_num_tensor_args_to_observation_type({
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    })

    backend_config.set_backend_pattern_config(add_config)

    return backend_config


def get_backend_config(default=True):

    if default:
        backend_config = get_native_backend_config()
    else:
        backend_config = BackendConfig()

    # ------------------------- Conv2d + ReLU6 -------------------------

    backend_config = register_module_fuse(
        (nn.Conv2d, nn.ReLU6), 
        fuse.ConvReLU62d, qat.ConvReLU62d, quant.ConvReLU62d,
        nnqr.Conv2d, weighted_int8_dtype_config, backend_config
    )

    backend_config = register_module_fuse(
        (nn.Conv2d, F.relu6), 
        fuse.ConvReLU62d, qat.ConvReLU62d, quant.ConvReLU62d,
        nnqr.Conv2d, weighted_int8_dtype_config, backend_config
    )
    
    # --------------------- Conv2d + BN2d + ReLU6 ----------------------

    backend_config = register_module_fuse(
        (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6), 
        fuse.ConvBnReLU62d, qat.ConvBnReLU62d, quant.ConvReLU62d,
        nnqr.Conv2d, weighted_int8_dtype_config, backend_config
    )

    backend_config = register_module_fuse(
        (nn.Conv2d, nn.BatchNorm2d, F.relu6), 
        fuse.ConvBnReLU62d, qat.ConvBnReLU62d, quant.ConvReLU62d,
        nnqr.Conv2d, weighted_int8_dtype_config, backend_config
    )

    # -------------------------- Add + ReLU6 ---------------------------

    backend_config = register_add_fuse(
        nn.ReLU6,
        backend_config
    )

    # ---------------------- Linear + BN1d + ReLU ----------------------

    backend_config = register_module_fuse(
        (nn.Linear, nn.BatchNorm1d, nn.ReLU), 
        fuse.LinearBnReLU1d, qat.LinearBnReLU1d, quant.LinearReLU,
        nnqr.Linear, weighted_int8_dtype_config, backend_config
    )

    backend_config = register_module_fuse(
        (nn.Linear, nn.BatchNorm1d, F.relu), 
        fuse.LinearBnReLU1d, qat.LinearBnReLU1d, quant.LinearReLU,
        nnqr.Linear, weighted_int8_dtype_config, backend_config
    )

    return backend_config
