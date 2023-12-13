from __future__ import annotations

import yaml
import copy
from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping as _QConfigMapping, QConfig
from torch.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from equant.core.observers import QuantileObserver, QuantilePerChannelObserver
from equant.core.quantizers.fake_quantize import FakeQuantizeTorchBase, FakeQuantize, FakeQuantizeLSQ


__all__ = [
    'generate_qconfig',
    'generate_qconfig_mapping'
]


OBSERVERS = {
    'min_max': {
        'per_tensor': MinMaxObserver,
        'per_channel': PerChannelMinMaxObserver
    },

    'moving_average': {
        'per_tensor': MovingAverageMinMaxObserver,
        'per_channel': MovingAveragePerChannelMinMaxObserver
    },

    'quantile': {
        'per_tensor': QuantileObserver,
        'per_channel': QuantilePerChannelObserver
    },

    'histogram': {
        'per_tensor': HistogramObserver
    }
}


QUANTIZERS = {
    'torch_base': FakeQuantizeTorchBase,
    'base': FakeQuantize,
    'lsq': FakeQuantizeLSQ
}


def validate_quantizer(quantizer: nn.Module) -> bool:
    
    if quantizer not in QUANTIZERS:
        raise ValueError(f'Invalid quantizer {quantizer}')
    
    return True
    

def validate_observer(observer: nn.Module) -> bool:

    if observer not in OBSERVERS:
        raise ValueError(f'Invalid quantizer {observer}')
    
    return True


def validate_layers(layers: List[str], module_names: Set[str]) -> bool:
    for layer in layers:
        if layer not in module_names:
            raise ValueError(f'Layer {layer} not found in modules')
        
    return True


def validate_config(config: Dict, module_names: List[str]) -> bool:

    module_names = set(module_names)

    for i, subconfig1 in enumerate(config):

        validate_quantizer(subconfig1['weight']['quantizer'])
        validate_quantizer(subconfig1['activation']['quantizer'])
        
        validate_observer(subconfig1['weight']['observer'])
        validate_observer(subconfig1['activation']['observer'])

        validate_layers(subconfig1['layers'], module_names)

        for j in range(i + 1, len(config)):

            subconfig2 = config[j]

            layers1 = set(subconfig1['layers'])
            layers2 = set(subconfig2['layers'])
            common_layers = layers1.intersection(layers2)
            
            if common_layers:
                raise ValueError(f'Config contains different strategies for {common_layers}')
                
    return True


def parse_dtype(dtype: str) -> Tuple[str, str]:
    
    sign = dtype[0]

    if sign not in ['s', 'u']:
        raise ValueError(f'qtype must be in form [s/u][n_bits] (without brackets), got {dtype}')

    try:
        n_bits = float(dtype[1:])
    except ValueError:
        raise ValueError(f'qtype must be in form [s/u][n_bits] (without brackets), got {dtype}')
    
    if n_bits.is_integer():
        n_bits = int(n_bits)
    
    return sign, n_bits


def create_quantizer_from_dict(config: Dict) -> nn.Module:

    sign, n_bits = parse_dtype(config['dtype'])
    dtype = {'u': torch.quint8,'s': torch.qint8}[sign]

    if dtype == torch.qint8:
        quant_min, quant_max = -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1
    else:
        quant_min, quant_max = 0, 2 ** n_bits - 1

    qscheme = getattr(torch, config['qscheme'])
    granularity = '_'.join(config['qscheme'].split('_')[:2])

    quantizer_cls = QUANTIZERS[config['quantizer']]
    observer_cls = OBSERVERS[config['observer']][granularity]
    observer_kwargs = config.get('observer_kwargs', {})

    quantizer = quantizer_cls.with_args(observer=observer_cls, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **observer_kwargs)
    return quantizer


def generate_qconfig(model: nn.Module, config: Dict) -> Dict:

    named_modules = {name: module for name, module in model.named_modules() if len(list(module.children())) == 0}

    validate_config(config, named_modules.keys())

    qconfig = {}
    for subconfig in config:

        for layer in subconfig['layers']:

            if len(list(named_modules[layer].parameters())) == 0:

                qconfig[layer] = {
                    'activation': copy.deepcopy(subconfig['activation'])
                }

            else:

                qconfig[layer] = {
                    'weight': copy.deepcopy(subconfig['weight']),
                    'activation': copy.deepcopy(subconfig['activation'])
                }

    for layer in named_modules:

        if len(list(named_modules[layer].parameters())) == 0:
            continue

        if layer not in qconfig:
            qconfig[layer] = 'fp32'

    return qconfig


class QConfigMapping(_QConfigMapping):

    def __init__(self):
        super().__init__()

        self._qconfig = {} # TODO: add conversion qconfig_mapping -> qconfig

    @property
    def qconfig(self) -> Dict:
        return self._qconfig

    @classmethod
    def from_qconfig(cls, qconfig_dict: Dict) -> QConfigMapping:

        qconfig_mapping = cls()
        qconfig_mapping._qconfig = qconfig_dict

        layer: str
        strategy: Union[str, Dict]

        for layer, strategy in qconfig_dict.items():

            if strategy == 'fp32':
                continue

            weight_quantizer = create_quantizer_from_dict(strategy.get('weight', strategy['activation']))
            activation_quantizer = create_quantizer_from_dict(strategy['activation'])

            qconfig = QConfig(activation_quantizer, weight_quantizer)
            qconfig_mapping.set_module_name(layer, qconfig=qconfig)

        return qconfig_mapping

    @classmethod
    def from_file(cls, path: str) -> QConfigMapping:

        with open(path, 'r') as f:
            qconfig_dict = yaml.safe_load(f)

        return cls.from_qconfig(qconfig_dict)
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.safe_dump(self.qconfig, f, sort_keys=False)


def generate_qconfig_mapping(model: nn.Module, config: Dict) -> QConfigMapping:
    
    qconfig = generate_qconfig(model, config)
    qconfig_mapping = QConfigMapping.from_qconfig(qconfig)

    return qconfig_mapping
