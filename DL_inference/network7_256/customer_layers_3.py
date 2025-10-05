

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/data1/sdsun/code/NLOS_adp_os/DL_inference/utils_pytorch')
from tfmodule import diffmodule as lct

from DL_inference.network7_256.ops import ResConv2D, ResConv3D, Interpsacle2d
import numpy as np

debugvalue = 1



#################################################################################
class Transient2volumn(nn.Module):
    
    def __init__(self, nf0, in_channels,
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn, self).__init__()
        
        ###############################################
        assert in_channels == 1
        
        weights = np.zeros((1, 1, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[2, 2, 2],
                      bias=True),
            
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )
    
    def forward(self, x0):
        
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights,
                           bias=None, stride=2, padding=1, dilation=1, groups=1)
        
        x1 = self.conv1(x0)
        
        re = torch.cat([x0_conv, x1], dim=1)

        return re
    

#################################################################################

def make_norm3d(name, plane, affine=True, per_channel=True):
    if name == 'batch':
        return nn.BatchNorm3d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm3d(plane, affine=affine)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )
    

def make_actv(actv):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'exp':
        return lambda x: torch.exp(x)
    elif actv == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif actv == 'tanh':
        return lambda x: torch.tanh(x)
    elif actv == 'softplus':
        return lambda x: torch.log(1 + torch.exp(x - 1))
    elif actv == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid activation function: {:s}'.format(actv)
        )


class ResConv3d(nn.Module):
    """ Residual block with 3D conv layers """
    def __init__(
        self, 
        in_plane,           # number of input planes
        plane,              # number of intermediate and output planes
        stride=1,           # stride of first conv layer
        actv='leaky_relu',  # activation function
        norm='none',        # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(ResConv3d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == 'none' or not affine else False

        self.conv1 = nn.Conv3d(
            in_plane, plane, 3, stride, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm1 = make_norm3d(norm, plane, affine)
        self.conv2 = nn.Conv3d(
            plane, plane, 3, 1, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm2 = make_norm3d(norm, plane, affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = nn.Conv3d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = make_norm3d(norm, plane, affine)

        self.actv = make_actv(actv)
    
    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x

class ResBlock3d(nn.Module):

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
        super(ResBlock3d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv3d(in_plane, plane, stride, actv, norm, affine)
            )
            in_plane = plane
            stride = 1
        self.layers = nn.Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x

class FRN(nn.Module):
    """
    Feature extraction and propagation network
    (NOTE: this implementation strictly follows Chen et al., SIGGRAPH Asia 2020
     for reproducibility)
    """
    def __init__(
        self,
        in_plane=1,         # number of input planes
        plane=6,            # number of planes prior to propagation
        actv="leaky_relu",  # activation function
        norm="none",        # normalization function
        affine=False,       # if True, apply learnable affine transform in norm
        **kwargs,
    ):
        super(FRN, self).__init__()

        bias = True if norm == "none" or not affine else False

        # conv1 is initialized as a mean filter
        conv1 = torch.zeros(1, in_plane, 3, 3, 3)
        conv1[..., 1:, 1:, 1:] = 0.125
        self.conv1 = nn.Parameter(conv1)

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_plane, plane - 1, 3, 2, 1),
            ResBlock3d(plane - 1, plane - 1, 1, 2, actv, norm, affine),
        )


    def forward(self, x):
        x1 = F.conv3d(x, self.conv1, bias=None, stride=2, padding=1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        return x


###########################################################################
class VisibleNet(nn.Module):
    
    def __init__(self, nf0, layernum=0, norm=nn.InstanceNorm3d):
        super(VisibleNet, self).__init__()
        
        self.layernum = layernum
    
    def forward(self, x):
        
        x5 = x
        
        ###############################################
        depdim = x5.shape[2]
        raw_pred_bxcxhxw, raw_dep_bxcxhxw = x5.max(dim=2)
        
        # -1 to 1
        # the nearer, the bigger ???
        raw_dep_bxcxhxw = depdim - 1 - raw_dep_bxcxhxw.float()
        raw_dep_bxcxhxw = raw_dep_bxcxhxw / (depdim - 1)
        
        xflatdep = torch.cat([raw_pred_bxcxhxw, raw_dep_bxcxhxw], dim=1)
        
        return xflatdep
    

# class VisbleNet(nn.Module):
#     """
#     A projection module that projects 3D feature volume into 2D feature maps.
#     """
#     def __init__(self):
#         super(VisbleNet, self).__init__()

#     def forward(self, x): 
#         ## x -> b c d h w
#         int, idx = torch.max(x,dim=2)
#         d = x.size(2)
#         # larger value for closer planes
#         depth = (idx.float() - 1) / (d - 1)
#         out = torch.cat([int, depth], dim=1)
#         return out


class Rendering(nn.Module):
    
    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()
        
        ######################################
        assert out_channels == 1
        
        weights = np.zeros((1, 2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, 1:, :, :] = 1.0
        else:
            weights[:, :1, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)
        
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),

            nn.Conv2d(nf0 * 1 + 1,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      nf0 * 4,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResConv2D(nf0 * 4, inplace=False),

            nn.Conv2d(nf0 * 4,
                      nf0 * 2,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
    
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:1, :, :]
        x0_dep = x0[:, dim:dim + 1, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + debugvalue * x2
        
        return re
