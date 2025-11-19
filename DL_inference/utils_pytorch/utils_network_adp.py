import torch
import torch.nn as nn

import cv2
import numpy as np
import scipy.io as sio
import os

from complexLayersAsReal import (
    ComplexConv2d,
    ComplexConv3d,
    ComplexConvTranspose2d,
    ComplexConvTranspose3d,
    ComplexInstanceNorm2d,
    ComplexInstanceNorm2d,
    ComplexInstanceNorm3d,
    ComplexBatchNorm3d,
    ComplexDownSamplingBlock,
    ComplexUpSamplingBlock,
    ComplexLeakyReLU,
)
from complexFunctions import complex_relu, complex_leaky_relu
from utils import (
    normalize_complex,
    un_normalize_complex,
    normalize_z_score,
    un_normalize_z_score,
)


def normalize(datapad_BDx2Tx2Hx2W):
    bd, t2, h2, w2 = datapad_BDx2Tx2Hx2W.shape
    data_bxtxk = datapad_BDx2Tx2Hx2W.reshape(bd, t2, -1)

    # most are 0
    data_max = torch.abs(data_bxtxk).max(2, keepdim=True)[0]
    data_norm = data_bxtxk / (data_max + 1e-15)

    return data_norm.view(bd, t2, h2, w2)


class ConvBlock3d_complex(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super(ConvBlock3d_complex, self).__init__()
        self.conv_a = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.conv_p = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.actv = nn.LeakyReLU(inplace=True)

    def forward(self, x_a, x_p):
        return self.actv(self.conv_a(x_a)), self.actv(self.conv_p(x_p))


class ConvBlock3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.actv = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.actv(self.conv(x))

def normalize(data_bxcxdxhxw):
    b, c, d, h, w = data_bxcxdxhxw.shape
    data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
    
    data_min = data_bxcxk.min(2, keepdim=True)[0]
    data_zmean = data_bxcxk - data_min
    
    # most are 0
    data_max = data_zmean.max(2, keepdim=True)[0]
    data_norm = data_zmean / (data_max + 1e-15)
    
    return data_norm.view(b, c, d, h, w)


class DCNet(nn.Module):
    # Distance Compensation Network
    def __init__(self):
        super(DCNet, self).__init__()

        self.init_denoise = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout(0.1),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout(0.1),
            nn.Conv3d(16, 2, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([256, 128, 128]),
        )

        self.avg_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv_stf = nn.Sequential(
            nn.Conv3d(2 * 3, 4 * 3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(4 * 3, 8 * 3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(8 * 3, 16 * 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(16 * 3, 16 * 3, kernel_size=5, stride=1, padding=2),
            # nn.Dropout(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(16 * 3, 6 * 3, kernel_size=1, stride=1, padding=0),
            # nn.Dropout(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(6 * 3, 3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True)
        )


        self.temporal_conv = nn.Conv3d(256, 1, kernel_size=3, stride=1, padding=1)

        # Fully connected layer to compress the T dimension
        self.norm_before_softmax = nn.LayerNorm([3, 128, 128], eps=1e-5)

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x, grid_z_1, grid_z_2, grid_z_4):
        bnum, dnum, tnum, hnum, wnum = x.shape
        x0 = nn.functional.leaky_relu(x + self.init_denoise(x), negative_slope=0.2, inplace=True)
        x0 = normalize(x0)
        # downsample first [b,c,t,h//2,w//2]
        x1 = self.avg_pool(x0)

        # compute initial distance compensation result
        # x1_dc_0 = x1
        x1_dc_1 = x1 * grid_z_1
        x1_dc_2 = x1 * grid_z_2
        x1_dc_4 = x1 * grid_z_4

        x_dc_0 = x0
        x_dc_1 = x0 * grid_z_1
        x_dc_2 = x0 * grid_z_2
        x_dc_4 = x0 * grid_z_4

        # concat all distance compensation result
        x1_dc_concat = torch.concat([x1_dc_1, x1_dc_2, x1_dc_4], dim=1)

        # conv and fusion all features
        x2 = self.conv_stf(x1_dc_concat)

        x2 = self.temporal_conv(x2.permute(0,2,1,3,4)).squeeze(1).view(3, hnum // 2, wnum // 2)  # [b,d,h,w]

        # nearest interpolate
        x3 = nn.functional.interpolate(x2.view(3, 1, hnum // 2, wnum // 2), size=(hnum, wnum), mode='nearest').squeeze(1)
        x3 = x3.view(bnum, 3, hnum, wnum)
        x3 = self.norm_before_softmax(x3)

        # sigmoid to norm
        x4 = self.softmax(x3)

        # hadamard product
        out = x_dc_1 * x4[:, 0:1] + x_dc_2 * x4[:, 1:2] + x_dc_4 * x4[:, 2:3]
        out = normalize(out)

        
        out_gpu = (grid_z_1 * x4[:, 0:1] + grid_z_2 * x4[:, 1:2] + grid_z_4 * x4[:, 2:3])
        out_gpu = torch.log(out_gpu.abs()) / torch.log(grid_z_1.abs())

        del x1_dc_concat

        return out, out_gpu.clone().detach()


def gaussianwin(L, alpha):
    N = L - 1
    Nhalf = N / 2.0
    n_k = torch.arange(N + 1).cuda() - Nhalf
    w_k = torch.exp(-0.5 * (alpha * n_k / Nhalf) ** 2)

    return w_k


def waveconvparam(bin_resolution, virtual_wavelength, cycles, window):
    c = 3e8
    s_z = bin_resolution * c
    samples = int(round(cycles * virtual_wavelength / (bin_resolution * c)))
    num_cycles = samples * s_z / virtual_wavelength

    grids_k = torch.arange(samples, dtype=torch.float32).cuda() + 1
    sin_wave_k = torch.sin(2 * torch.pi * (num_cycles * grids_k) / samples)
    cos_wave_k = torch.cos(2 * torch.pi * (num_cycles * grids_k) / samples)

    virtual_sin_wave_k = sin_wave_k * window
    virtual_cos_wave_k = cos_wave_k * window

    return virtual_cos_wave_k, virtual_sin_wave_k


class FrequencyPositionEncoder(nn.Module):
    def __init__(self, Fs):
        super(FrequencyPositionEncoder, self).__init__()
        self.Fs = Fs # sampling frequency
        # MLP to encode position
        self.position_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, fea):
        _, _, T = fea.size()
        Fs = self.Fs  # a simple normalization trick
        fre_pos = (torch.arange(T).cuda() * (Fs / T)).unsqueeze(0).unsqueeze(0)
        fre_pos = fre_pos / torch.max(fre_pos)
        return fea + self.position_encoder(fre_pos)


class AdaptiveVirtualWave(nn.Module):
    def __init__(self, bin_resolution, virtual_wavelength, cycles):
        super(AdaptiveVirtualWave, self).__init__()

        self.bin_resolution = bin_resolution
        self.virtual_wavelength = virtual_wavelength
        self.cycles = cycles
        self.c = 3e8

        # conv layer to move amptitude
        # 1x1 Convolution layer
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),  # D x T x 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),  # D x T x 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 4 x 4
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0)),  # D x T x 2 x 2
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 2048, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0)),  # D x T x 1 x 1
            nn.ReLU(inplace=True),
        )

        # 1x1 Convolution layer
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.norm = nn.LayerNorm([1, 257])

        # Fully connected layer to compress the T dimension
        self.fc = nn.Linear(in_features=257, out_features=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x_f):
        # B, D, _, _, _ = x_f.shape
        x_pad = torch.zeros_like(x_f)
        x_pad = x_pad.repeat(1, 1, 2, 2, 2)
        B, D, T2, H2, W2 = x_pad.shape
        center_T = T2 // 2
        center_H = H2 // 2
        center_W = W2 // 2
        # x_pos = x[:, :, : center_T + 1, :, :]  # get a half frequency position part

        x_pad[:, :, :center_T, :center_H, :center_W] = x_f
        x_pad = torch.fft.fft(x_pad, dim=-3, norm="ortho")  # ortho norm
        x_pad = x_pad[:, :, : center_T + 1, :, :]  # get a half frequency position part
        x_pad = x_pad.abs()

        # Summing across the H and W dimensions
        # x = x_pad.mean(dim=-1).mean(dim=-1)  # Shape: [B, D, T]
        x1 = self.spatial_conv(x_pad).squeeze(-1).squeeze(-1)

        # 1x1 Convolution
        x1 = self.norm(self.conv1x1(x1))  # Shape: [B, 1, T]

        # Fully connected layer
        x3 = self.fc(x1.squeeze(1))  # Shape: [B, 1]

        # Calculate the updated alpha value
        sigma_after = 1.0 / 2 * torch.pi * (x3 + 1e-6)

        sigma = (nn.functional.sigmoid(sigma_after) * 0.2) + 0.2
        alpha = 1.0 / (sigma + 1e-6)
        # Calculate the window using the gaussianwin function
        samples = int(
            round(
                self.cycles * self.virtual_wavelength / (self.bin_resolution * self.c)
            )
        )
        
        window = gaussianwin(
            samples, alpha
        )

        # Calculate the wave convolution parameters
        virtual_cos_wave_k, virtual_sin_wave_k = waveconvparam(
            self.bin_resolution, self.virtual_wavelength, self.cycles, window
        )

        virtual_cos_sin_wave_2xk = torch.stack(
            [virtual_cos_wave_k, virtual_sin_wave_k], dim=0
        )

        return virtual_cos_sin_wave_2xk, sigma


class AdaptiveVirtualWaveSingle(nn.Module):
    def __init__(self, bin_resolution, virtual_wavelength, cycles):
        super(AdaptiveVirtualWaveSingle, self).__init__()

        self.bin_resolution = bin_resolution
        self.virtual_wavelength = virtual_wavelength
        self.cycles = cycles
        self.c = 3e8

        # conv layer to move amptitude
        # 1x1 Convolution layer
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),  # D x T x 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),  # D x T x 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # D x T x 4 x 4
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0)),  # D x T x 2 x 2
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 2048, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0)),  # D x T x 1 x 1
            nn.ReLU(inplace=True),
        )

        # 1x1 Convolution layer
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.norm = nn.LayerNorm([1, 256])

        # Fully connected layer to compress the T dimension
        self.fc = nn.Linear(in_features=256, out_features=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x_f):
        # B, D, _, _, _ = x_f.shape
        x_pad = torch.zeros_like(x_f)
        x_pad = x_pad.repeat(1, 1, 2, 2, 2)
        B, D, T2, H2, W2 = x_pad.shape

        x_pad = torch.fft.fft(x_f, dim=-3, norm="forward")  # ortho norm
        x_pad = x_pad.abs()

        # Summing across the H and W dimensions
        x1 = self.spatial_conv(x_pad).squeeze(-1).squeeze(-1)

        # 1x1 Convolution
        x1 = self.norm(self.conv1x1(x1))  # Shape: [B, 1, T]

        # Fully connected layer
        x3 = self.fc(x1.squeeze(1))  # Shape: [B, 1]

        # Calculate the updated alpha value
        sigma_after = 1.0 / 2 * torch.pi * (x3 + 1e-6)

        sigma = (nn.functional.sigmoid(sigma_after) * 0.1) + 0.2
        alpha = 1.0 / (sigma + 1e-6)
        # Calculate the window using the gaussianwin function
        samples = int(
            round(
                self.cycles * self.virtual_wavelength / (self.bin_resolution * self.c)
            )
        )
        
        window = gaussianwin(
            samples, alpha
        )

        # Calculate the wave convolution parameters
        virtual_cos_wave_k, virtual_sin_wave_k = waveconvparam(
            self.bin_resolution, self.virtual_wavelength, self.cycles, window
        )

        virtual_cos_sin_wave_2xk = torch.stack(
            [virtual_cos_wave_k, virtual_sin_wave_k], dim=0
        )

        return virtual_cos_sin_wave_2xk, sigma


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Downsampling path
        self.conv0 = ConvBlock3d_complex(2, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = ConvBlock3d_complex(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = ConvBlock3d_complex(32, 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        x_pos, p1, p2 = normalize_complex(x, normalize_z_score)
        
        x0_a, x0_p = self.conv0(torch.abs(x_pos), torch.angle(x_pos))
        x1_a, x1_p = self.conv1(x0_a, x0_p)
        
        x2_a, x2_p = self.conv2(x1_a, x1_p)
        x2_a = x2_a + torch.abs(x_pos)
        x2_p = x2_p + torch.angle(x_pos)
        
        out = torch.complex(x2_a * torch.cos(x2_p), x2_a * torch.sin(x2_p))
        
        out = un_normalize_complex(out, un_normalize_z_score, p1, p2)
        
        return out