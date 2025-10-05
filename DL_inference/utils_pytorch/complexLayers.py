"""
Created on Tue Mar 19 10:30:02 2019

@author: Sebastien M. Popoff


Based on https://openreview.net/forum?id=H1T2hmZAb
"""
from typing import Optional

import torch
from torch.nn import (
    Module, Parameter, init,
    Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d, Linear, LSTM, GRU,
    BatchNorm1d, BatchNorm2d,
    InstanceNorm2d, InstanceNorm3d,
    LayerNorm,
    PReLU,
    Sequential
)

from complexFunctions import (
    complex_relu,
    complex_leaky_relu,
    complex_tanh,
    complex_sigmoid,
    complex_max_pool2d,
    complex_avg_pool2d,
    complex_max_pool3d,
    complex_dropout,
    complex_dropout2d,
    complex_opposite,
)


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
        + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


class ComplexDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


class ComplexDropout2d(Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complex_dropout2d(inp, self.p)
        else:
            return inp


class ComplexMaxPool2d(Module):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
        
        
class ComplexMaxPool3d(Module):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
    ):
        super(ComplexMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool3d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class ComplexAvgPool2d(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool2d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)


class ComplexReLU(Module):
    @staticmethod
    def forward(inp):
        return complex_relu(inp)
    
class ComplexLeakyReLU(Module):
    @staticmethod
    def forward(inp):
        return complex_leaky_relu(inp)


class ComplexSigmoid(Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid(inp)


class ComplexPReLU(Module):
    def __init__(self):
        super().__init__()
        self.r_prelu = PReLU()
        self.i_prelu = PReLU()

    @staticmethod
    def forward(self, inp):
        return self.r_prelu(inp.real) + 1j * self.i_prelu(inp.imag)


class ComplexTanh(Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)


class ComplexConvTranspose2d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
            dilation=1,
            padding_mode="zeros",
    ):
        super().__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, inp):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, inp)


class ComplexConvTranspose3d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
            dilation=1,
            padding_mode="zeros",
    ):
        super().__init__()

        self.conv_tran_r = ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, inp):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, inp)


class ComplexConv2d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, inp):
        return apply_complex(self.conv_r, self.conv_i, inp)


class ComplexConv3d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(ComplexConv3d, self).__init__()
        self.conv_r = Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        init.uniform_(self.conv_r.weight, 0, 1)
        init.uniform_(self.conv_i.weight, 0, 1)


    def forward(self, inp):
        return apply_complex(self.conv_r, self.conv_i, inp)


class ComplexLinear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, inp):
        return apply_complex(self.fc_r, self.fc_i, inp)
    
    
class ComplexInstanceNorm2d(Module):
    def __init__(self,
                 feature_dim):
        super().__init__()
        self.instance_norm_real = InstanceNorm2d(feature_dim)
        self.instance_norm_imag = InstanceNorm2d(feature_dim)

    def forward(self, input):
        return (self.instance_norm_real(input.real)).type(torch.complex64) \
            + 1j * (self.instance_norm_imag(input.imag)).type(torch.complex64)

class ComplexInstanceNorm3d(Module):
    def __init__(self,
                 feature_dim):
        super().__init__()
        self.instance_norm_real = InstanceNorm3d(feature_dim)
        self.instance_norm_imag = InstanceNorm3d(feature_dim)

    def forward(self, input):
        return (self.instance_norm_real(input.real)).type(torch.complex64) \
            + 1j * (self.instance_norm_imag(input.imag)).type(torch.complex64)
            
            
class NaiveComplexLayerNorm(Module):
    def __init__(self,
                 shape,
                 eps=1e-5,
                 elementwise_affine=True,
                 bias=True):
        super().__init__()
        self.layer_norm_real = LayerNorm(shape, eps, elementwise_affine, bias=bias)
        self.layer_norm_imag = LayerNorm(shape, eps, elementwise_affine, bias=bias)

    def forward(self, input):
        return self.layer_norm_real(input.real).type(torch.complex64) \
            + 1j * self.layer_norm_imag(input.imag).type(torch.complex64)


class NaiveComplexBatchNorm1d(Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


class _ComplexBatchNorm(Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm3d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3, 4]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3, 4]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3, 4]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3, 4]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3, 4])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st
        inp = (
              Rrr[None, :, None, None, None] * inp.real + Rri[None, :, None, None, None] * inp.imag
          ).type(torch.complex64) + 1j * (
              Rii[None, :, None, None, None] * inp.imag + Rri[None, :, None, None, None] * inp.real
          ).type(
        torch.complex64
        )

        if self.affine:
            inp = (
                    self.weight[None, :, None, None, None] * inp.real
                    + self.weight[None, :, None, None, None] * inp.imag
                    + self.bias[None, :, None, None, None]
                ).type(torch.complex64) + 1j * (
                    self.weight[None, :, None, None, None] * inp.real
                    + self.weight[None, :, None, None, None] * inp.imag
                    + self.bias[None, :, None, None, None]
                ).type(
                torch.complex64
            )
        return inp


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                                                 float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                        exponential_average_factor * Crr * n / (n - 1)  #
                        + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                        exponential_average_factor * Cii * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                        exponential_average_factor * Cri * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
                      Rrr[None, :, None, None] * inp.real + Rri[None, :, None, None] * inp.imag
              ).type(torch.complex64) + 1j * (
                      Rii[None, :, None, None] * inp.imag + Rri[None, :, None, None] * inp.real
              ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                          self.weight[None, :, 0, None, None] * inp.real
                          + self.weight[None, :, 2, None, None] * inp.imag
                          + self.bias[None, :, 0, None, None]
                  ).type(torch.complex64) + 1j * (
                          self.weight[None, :, 2, None, None] * inp.real
                          + self.weight[None, :, 1, None, None] * inp.imag
                          + self.bias[None, :, 1, None, None]
                  ).type(
                torch.complex64
            )
        return inp


class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, inp):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                                                 float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = inp.real.mean(dim=0).type(torch.complex64)
            mean_i = inp.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, ...]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = inp.real.var(dim=0, unbiased=False) + self.eps
            Cii = inp.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                        exponential_average_factor * Crr * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                        exponential_average_factor * Cii * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                        exponential_average_factor * Cri * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (Rrr[None, :] * inp.real + Rri[None, :] * inp.imag).type(
            torch.complex64
        ) + 1j * (Rii[None, :] * inp.imag + Rri[None, :] * inp.real).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                          self.weight[None, :, 0] * inp.real
                          + self.weight[None, :, 2] * inp.imag
                          + self.bias[None, :, 0]
                  ).type(torch.complex64) + 1j * (
                          self.weight[None, :, 2] * inp.real
                          + self.weight[None, :, 1] * inp.imag
                          + self.bias[None, :, 1]
                  ).type(
                torch.complex64
            )

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inp

# def make_actv(actv):
#     if actv == 'relu':
#         return ComplexReLU(inplace=True)
#     elif actv == 'leaky_relu':
#         return nn.LeakyReLU(0.2, inplace=True)
#     elif actv == 'exp':
#         return lambda x: torch.exp(x)
#     elif actv == 'sigmoid':
#         return lambda x: torch.sigmoid(x)
#     elif actv == 'tanh':
#         return lambda x: torch.tanh(x)
#     elif actv == 'softplus':
#         return lambda x: torch.log(1 + torch.exp(x - 1))
#     elif actv == 'linear':
#         return nn.Identity()
#     else:
#         raise NotImplementedError(
#             'invalid activation function: {:s}'.format(actv)
#         )
    
class ComplexResConv3d(Module):
    """ Complex Residual block with 3D conv layers """
    def __init__(
        self, 
        in_plane,           # number of input planes
        plane,              # number of intermediate and output planes
        stride=1,           # stride of first conv layer
        actv='leaky_relu',  # activation function
        norm='none',        # normalization function
        affine=False,        # if True, apply learnable affine transform in norm
    ):
        super(ComplexResConv3d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == 'none' or not affine else False

        self.conv1 = ComplexConv3d(
            in_plane, plane, 3, stride, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm1 = ComplexBatchNorm3d(num_features=plane, momentum=0.9, affine=affine)
        self.conv2 = ComplexConv3d(
            plane, plane, 3, 1, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm2 = ComplexBatchNorm3d(num_features=plane, momentum=0.9, affine=affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = ComplexConv3d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = ComplexBatchNorm3d(num_features=plane, momentum=0.9, affine=affine)

        self.actv = ComplexLeakyReLU()
    
    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x


class ComplexResBlock3d(Module):
    # Resnet Block complex version
    def __init__(
        self, 
        in_plane, 
        plane, 
        stride, 
        n_layers,  
        actv='relu', 
        norm='none', 
        affine=False,
    ):
        super(ComplexResBlock3d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ComplexResConv3d(in_plane, plane, stride, actv, norm, affine)
            )
            in_plane = plane
            stride = 1
        self.layers = Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x


class ComplexUnetConvBlock(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(ComplexUnetConvBlock, self).__init__()
        channels = out_channels // 2
        if in_channels > out_channels:
            channels = in_channels // 2

        self.double_conv = Sequential(
            # ComplexConv3d(in_channels, channels, kernel_size, stride, padding),
            # ComplexLeakyReLU(),

            # ComplexConv3d(channels, out_channels, kernel_size, stride, padding),
            # ComplexLeakyReLU(),
            
            ComplexConv3d(in_channels, out_channels, kernel_size, stride, padding),
            ComplexLeakyReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class ComplexDownSamplingBlock(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=3):
        super(ComplexDownSamplingBlock, self).__init__()
        self.maxpool_to_conv = Sequential(
            ComplexMaxPool3d(kernel_size=2, stride=2),
            ComplexUnetConvBlock(in_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)
    

class ComplexUpSamplingBlock(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super(ComplexUpSamplingBlock, self).__init__()
        # 采用反卷积进行上采样
        # self.up = ComplexConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2, output_padding=(1, 0, 0))
        self.up = ComplexConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ComplexUnetConvBlock(in_channels // 2 + in_channels // 2, out_channels)

    # inputs1：上采样的数据（对应黄色箭头传来的数据）
    # inputs2：特征融合的数据（对应绿色箭头传来的数据）
    def forward(self, inputs1, inputs2):
        # 进行一次up操作
        inputs1 = self.up(inputs1)

        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs