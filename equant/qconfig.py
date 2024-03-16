from __future__ import annotations

import math
import yaml
import copy
from typing import Dict, List, Set, Tuple, Union, Any

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping as _QConfigMapping, QConfig
from torch.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from equant.core.observers import QuantileObserver, QuantilePerChannelObserver, MSEObserver, MSEPerChannelObserver
from equant.core.quantizers import FixedQParamsFakeQuantize, FakeQuantizeLSQ, FakeQuantizeLSQPlus


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

    'mse': {
        'per_tensor': MSEObserver,
        'per_channel': MSEPerChannelObserver
    },

    'histogram': {
        'per_tensor': HistogramObserver
    }
}


QUANTIZERS = {
    'fixed_qparams': FixedQParamsFakeQuantize,
    'lsq': FakeQuantizeLSQ,
    'lsq+': FakeQuantizeLSQPlus
}


def has_parameters(module: nn.Module) -> bool:
    return len(list(module.parameters())) != 0


def is_lead_node(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def validate_quantizer(quantizer: nn.Module) -> None:
    
    if quantizer not in QUANTIZERS:
        raise ValueError(f'Invalid quantizer {quantizer}')
    

def validate_observer(observer: nn.Module) -> None:

    if observer not in OBSERVERS:
        raise ValueError(f'Invalid quantizer {observer}')


def validate_layers(layers: List[str], module_names: Set[str]) -> None:
    for layer in layers:
        if layer not in module_names:
            raise ValueError(f'Layer {layer} not found in modules')


def validate_config(config: Dict, module_names: List[str]) -> None:

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


def replace_regex(config: Dict, named_modules: Dict[str, nn.Module]) -> None:
    for subconfig in config:
        if '*' in subconfig['layers']:
            subconfig['layers'] = named_modules.keys()


def _has_parent_in_config(module_name: str, config: Set[str]) -> bool:
    if len(module_name) == 0:
        return False
    if module_name in config:
        return True
    else:
        *parent, _ = module_name.rsplit('.', 1)
        parent = parent[0] if parent else ''
        return _has_parent_in_config(parent, config)


def replace_parent_names(config: Dict, layers: Set[str]) -> None:
    for subconfig in config:
        extra_layers = set()
        for layer in layers:
            if _has_parent_in_config(layer, set(subconfig['layers'])):
                extra_layers.add(layer)
        subconfig['layers'] = layers.intersection(subconfig['layers']).union(extra_layers)


def set_attribute_recursively(d: Dict, key: str, value: Any) -> None:
    *parent, child = key.split('.', 1)
    if len(parent) == 0:
        d[child] = value
    else:
        parent = parent[0]
        if parent not in d:
            d[parent] = {}
        set_attribute_recursively(d[parent], child, value)


def transform_to_tree(d: Dict) -> Dict:
    config = {}
    for k, v in d.items():
        set_attribute_recursively(config, k, v)
    return config


def generate_qconfig(model: nn.Module, config: Dict) -> Dict:

    named_modules = {name: module for name, module in model.named_modules() if is_lead_node(module)}

    replace_regex(config, named_modules)
    replace_parent_names(config, set(named_modules.keys()))
    validate_config(config, named_modules.keys())

    qconfig = {}
    for subconfig in config:

        for layer in subconfig['layers']:

            if not has_parameters(named_modules[layer]):

                qconfig[layer] = {
                    'activation': copy.deepcopy(subconfig['activation'])
                }

            else:

                qconfig[layer] = {
                    'weight': copy.deepcopy(subconfig['weight']),
                    'activation': copy.deepcopy(subconfig['activation'])
                }

    # To preserve the original order of modules
    qconfig = {layer: qconfig[layer] if layer in qconfig else 'fp32' for layer in named_modules}

    # NOTE: do we realy need this for readability?
    # qconfig = transform_to_tree(qconfig)

    return qconfig


def partial_qconfig_to_dict(partial_qconfig: Any, inverse_observers: Dict, inverse_quantizers: Dict) -> Dict:

    quantizer = inverse_quantizers[partial_qconfig.func]
    quantizer_arguments: Dict = copy.deepcopy(partial_qconfig.keywords)

    n_bits = math.log2(quantizer_arguments.pop('quant_max') - quantizer_arguments.pop('quant_min') + 1)
    if int(n_bits) == n_bits:
        n_bits = int(n_bits)

    sign = {
        torch.qint8: 's',
        torch.quint8: 'u'
    }[quantizer_arguments.pop('dtype')]

    qscheme = {
        torch.per_tensor_symmetric: 'per_tensor_symmetric',
        torch.per_tensor_affine: 'per_tensor_affine',
        torch.per_channel_symmetric: 'per_channel_symmetric',
        torch.per_channel_affine: 'per_channel_affine',
    }[quantizer_arguments.pop('qscheme')]

    config = {
        'dtype': sign + str(n_bits),
        'qscheme': qscheme,
        'observer': inverse_observers[quantizer_arguments.pop('observer')],
        'observer_kwargs': copy.deepcopy(quantizer_arguments),
        'quantizer': quantizer
    }

    if len(quantizer_arguments) == 0:
        del config['observer_kwargs']

    return config


class QConfigMapping(_QConfigMapping):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_qconfig(cls, qconfig_dict: Dict) -> QConfigMapping:

        qconfig_mapping = cls()

        layer: str
        strategy: Union[str, Dict]

        for layer, strategy in qconfig_dict.items():

            if strategy == 'fp32':
                qconfig = None
            else:
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
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def to_dict(self) -> Dict:
        qconfig = {}

        inverse_quantizers = {v: k for k, v in QUANTIZERS.items()}
        inverse_observers = {
            **{v['per_tensor']: k for k, v in OBSERVERS.items() if 'per_tensor' in v}, 
            **{v['per_channel']: k for k, v in OBSERVERS.items() if 'per_channel' in v}
        }

        for layer, config in self.module_name_qconfigs.items():

            if config is None:
                qconfig[layer] = 'fp32'
            else:
                qconfig[layer] = {
                    'weight': partial_qconfig_to_dict(config.weight.p, inverse_observers, inverse_quantizers),
                    'activation': partial_qconfig_to_dict(config.activation.p, inverse_observers, inverse_quantizers)
                }

        return qconfig


def generate_qconfig_mapping(model: nn.Module, config: Dict) -> QConfigMapping:
    
    qconfig = generate_qconfig(model, config)
    qconfig_mapping = QConfigMapping.from_qconfig(qconfig)

    return qconfig_mapping
