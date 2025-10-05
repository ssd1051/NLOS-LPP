
import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
sys.path.append('/data1/sdsun/sdsun/code/NLOS_adp/DL_inference/utils')
from helper import definePsf_single, resamplingOperator, waveconvparam, waveconv
from utils_network_adp import AdaptiveVirtualWaveSingle, UNet, DCNet


class phasor(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5):
        super(phasor, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len  # this is route length (m)
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
        
        self.adp_virtual_wave = AdaptiveVirtualWaveSingle(self.bin_resolution, self.virtual_wavelength, self.cycles)
        self.unet = UNet()
        self.dcnet = DCNet()
    
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0
        self.bin_resolution = self.bin_len / self.c  # s / v = t
        self.trange = self.crop * self.c * self.bin_resolution

        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles

        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))
        
        ######################################################
        s_lamda_limit = wall_size / (sptial_grid - 1)
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2)
        self.virtual_wavelength = virtual_wavelength
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf_single(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
    def todev(self, dev, dnum):
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)

        self.gridz_1xMx1x1_todev = self.gridz_1xMx1x1.to(dev)
        self.gridz_1xMx1x1_todev_1 = self.gridz_1xMx1x1_todev.clone() ** 1
        self.gridz_1xMx1x1_todev_2 = self.gridz_1xMx1x1_todev.clone() ** 2
        self.gridz_1xMx1x1_todev_4 = self.gridz_1xMx1x1_todev.clone() ** 4
        
    def forward(self, feture_bxdxtxhxw, tbes, tens):
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)

        gridz_1xMx1x1_1 = self.gridz_1xMx1x1_todev_1
        gridz_1xMx1x1_2 = self.gridz_1xMx1x1_todev_2
        gridz_1xMx1x1_4 = self.gridz_1xMx1x1_todev_4
        data_BDxTxHxW, LPC_mat = self.dcnet(data_BDxTxHxW.view(bnum, dnum, tnum, hnum, wnum), 
                                          gridz_1xMx1x1_1.reshape(1, 1, -1, 1, 1),
                                          gridz_1xMx1x1_2.reshape(1, 1, -1, 1, 1),
                                          gridz_1xMx1x1_4.reshape(1, 1, -1, 1, 1),)
        data_BDxTxHxW = data_BDxTxHxW.view(bnum * dnum, tnum, hnum, wnum)

        torch.cuda.empty_cache()
        
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        virtual_cos_sin_wave_inv_2x1xk_todev, alpha=self.adp_virtual_wave(data_BDHWx1xT.reshape(bnum, dnum, hnum, wnum, tnum).permute(0, 1, 4, 2, 3))
        knum = virtual_cos_sin_wave_inv_2x1xk_todev.shape[2]
        phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum // 2)
        torch.cuda.empty_cache()
        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        
        #############################################################    
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_2BDx2Tx2Hx2W = tmp2
        
        ###########################################################3
        datafre = torch.fft.fftn(datapad_2BDx2Tx2Hx2W, dim=(-3, -2, -1), norm="ortho")
        
        datafre = self.unet(datafre.view(2 * bnum, dnum, temprol_grid, sptial_grid, sptial_grid))
        datafre = torch.view_as_real(datafre.view(2 * bnum * dnum, self.crop, hnum, wnum))
        
        torch.cuda.empty_cache()
        
        re_real = datafre[:, :, :, :, 0] * self.invpsf_real_todev - datafre[:, :, :, :, 1] * self.invpsf_imag_todev
        re_imag = datafre[:, :, :, :, 0] * self.invpsf_imag_todev + datafre[:, :, :, :, 1] * self.invpsf_real_todev
        refre = torch.stack([re_real, re_imag], dim=4)
        refre = torch.view_as_complex(refre)
        re = torch.view_as_real(torch.fft.ifftn(refre, dim=(-3, -2, -1), norm="ortho"))
        volumn_2BDxTxHxWx2 = re
        
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 0]
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 1]
        
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 0]
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 1]
        
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)

        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1).contiguous()
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid).contiguous()

        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)

        return volumn_BxDxTxHxW, LPC_mat, 0, alpha
