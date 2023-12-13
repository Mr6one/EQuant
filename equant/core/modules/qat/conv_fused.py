import torch.nn as nn
import torch.nn.functional as F

import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni

from equant.core.modules.fuse import conv_fused


class ConvReLU62d(nnqat.Conv2d, nni._FusedModule):

    _FLOAT_MODULE = conv_fused.ConvReLU62d
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU6

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode,
                         qconfig=qconfig)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        return F.relu6(self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(cls, mod):
        return super().from_float(mod)
    

class ConvBnReLU62d(nni.qat.modules.conv_fused.ConvBn2d):

    _FLOAT_MODULE = conv_fused.ConvBnReLU62d 
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = nn.ReLU6 
    _FUSED_FLOAT_MODULE = conv_fused.ConvReLU62d

    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias,
                         padding_mode, eps, momentum,
                         freeze_bn,
                         qconfig)

    def forward(self, input):
        return F.relu6(nni.qat.modules.conv_fused.ConvBn2d._forward(self, input))
    