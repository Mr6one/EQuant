import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
from torch.nn.utils.fusion import fuse_linear_bn_weights

from equant.core.modules.fuse import linear_fused
    

class LinearBnReLU1d(nniqat.LinearBn1d):

    _FLOAT_MODULE = linear_fused.LinearBnReLU1d 
    _FLOAT_CONV_MODULE = nn.Linear
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = nn.ReLU
    _FUSED_FLOAT_MODULE = nni.LinearReLU

    def __init__(self,
                 # Linear args
                 in_features, out_features, bias=True,
                 # BatchNorm1d args
                 # num_features: out_features
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super().__init__(in_features, out_features, bias,
                         eps, momentum, freeze_bn, qconfig)

    def forward(self, input):
        return F.relu(super().forward(self, input))
    
    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod' a float module, either produced by torch.ao.quantization
            utilities or directly from user
        """
        assert type(mod) == linear_fused.LinearBnReLU1d, 'qat.' + cls.__name__ + \
            '.from_float only works for ' + linear_fused.LinearBnReLU1d.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid config'
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(linear.in_features, linear.out_features, linear.bias is not None,
                           bn.eps, bn.momentum,
                           False, qconfig)
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_linearbn

    def to_float(self):
        linear = nn.Linear(self.in_features, self.out_features)
        assert self.bn.running_var is not None and self.bn.running_mean is not None
        linear.weight, linear.bias = fuse_linear_bn_weights(
            self.weight,
            self.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias)
        
        relu = nn.ReLU()
        return nni.LinearReLU(linear, relu)
