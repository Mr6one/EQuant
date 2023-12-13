import torch
import torch.nn.functional as F

import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq
from torch.nn.utils import fuse_conv_bn_weights

from equant.core.modules import fuse, qat


_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

class ConvReLU62d(nniq.ConvReLU2d):

    _FLOAT_MODULE = fuse.ConvReLU62d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, input):
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
            
        '''
        NOTE: We will remove qdq between conv and relu6 during export to ONNX

        NOTE: Actually, we can just replace relu6 with relu after PTQ or QAT is done
        '''
        
        x = torch.ops.quantized.conv2d(input, self._packed_params, self.scale, self.zero_point)
        scale = x.q_scale() 
        zero_point = x.q_zero_point()
        dtype = x.dtype

        x = x.dequantize()
        x = torch.quantize_per_tensor(F.relu6(x), scale, zero_point, dtype)

        return  x

    def _get_name(self):
        return 'QuantizedConvReLU62d'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == qat.ConvBnReLU62d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        return super().from_float(mod)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) != fuse.ConvBnReLU62d, \
            "BatchNorm2d should be fused into Conv2d before converting to reference module"
        return super().from_reference(ref_qconv, output_scale, output_zero_point)
