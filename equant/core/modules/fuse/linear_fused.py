import torch.nn as nn
from torch.ao.nn.intrinsic.modules.fused import _FusedModule, type_before_parametrizations


class LinearBnReLU1d(_FusedModule):
    def __init__(self, linear, bn, relu):
        assert type_before_parametrizations(linear) == nn.Linear and type_before_parametrizations(bn) == nn.BatchNorm1d and \
            type_before_parametrizations(relu) == nn.ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type_before_parametrizations(linear), type_before_parametrizations(bn), type_before_parametrizations(relu))
        super().__init__(linear, bn, relu)

    @classmethod
    def fuser_method(cls, is_qat, linear, bn, relu):
        return cls(linear, bn, relu)
