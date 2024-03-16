import torch.fx as fx
import torch.nn as nn

import copy
from typing import Union
from contextlib import contextmanager


@contextmanager
def save_quantization_state(model: Union[nn.Module, fx.GraphModule]):

    state = {'fake_quant_enabled': {}, 'observer_enabled': {}}
    for module in model.modules():
        if hasattr(module, 'fake_quant_enabled'):
            state['fake_quant_enabled'][module] = copy.deepcopy(module.fake_quant_enabled)

        if hasattr(module, 'observer_enabled'):
            state['observer_enabled'][module] =  copy.deepcopy(module.observer_enabled)

    try:
        yield
    finally:
        for module in model.modules():
            if hasattr(module, 'fake_quant_enabled'):
                module.fake_quant_enabled = state['fake_quant_enabled'][module]

            if hasattr(module, 'observer_enabled'):
                module.observer_enabled = state['observer_enabled'][module]
