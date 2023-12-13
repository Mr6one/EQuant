import torch.nn as nn
from torch.ao.nn.intrinsic.modules.fused import _FusedModule, type_before_parametrizations


class ConvReLU62d(_FusedModule):
    def __init__(self, conv, activation_fn):
        assert type_before_parametrizations(conv) == nn.Conv2d and type_before_parametrizations(activation_fn) == nn.ReLU6, \
            f'Incorrect types for input modules{type_before_parametrizations(conv)}{type_before_parametrizations(activation_fn)}'
        super().__init__(conv, activation_fn)

    @classmethod
    def fuser_method(cls, is_qat, conv, activation_fn):
        return cls(conv, activation_fn)


class ConvBnReLU62d(_FusedModule):
    def __init__(self, conv, bn, activation_fn):
        assert type_before_parametrizations(conv) == nn.Conv2d and type_before_parametrizations(bn) == nn.BatchNorm2d and \
            type_before_parametrizations(activation_fn) == nn.ReLU6, 'Incorrect types for input modules{}{}{}' \
            .format(type_before_parametrizations(conv), type_before_parametrizations(bn), type_before_parametrizations(activation_fn))
        super().__init__(conv, bn, activation_fn)

    @classmethod
    def fuser_method(cls, is_qat, conv, bn, activation_fn):
        return cls(conv, bn, activation_fn)
