

import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
sys.path.append('/data1/sdsun/sdsun/code/NLOS_adp_os/DL_inference/utils_pytorch')
from utils import init_mats

from tffk import lct_fk
from tflct import lct
from tfphasor_single import phasor
from tfphasor2 import phasor2
from cv2 import imshow
from tffkfast import lct_fk_fast


################################################################
class diffmodule(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 mview=False, mode='fk', \
                 align_corners=False, \
                 material='diffuse', \
                 sampling_coeff=2.0, cycles=5):
        super(diffmodule, self).__init__()
        
        self.mode = mode
        if mode == 'fk':
            self.model = lct_fk(spatial, crop, bin_len, wall_size, align_corners)
        
        elif mode == 'fkfast':
            self.model = lct_fk_fast(spatial, crop, bin_len, wall_size, align_corners)
        
        elif mode == 'lct' or mode == 'bp':
            self.model = lct(spatial, crop, bin_len, wall_size, \
                             method=mode, material=material)
        
        elif mode == 'phasor':
            self.model = phasor(spatial, crop, bin_len, wall_size, \
                                sampling_coeff, cycles)
        
        else:
            print('error!')
            
        self.align_corners = align_corners

        ################################################
        # multiple view
        self.mview = mview
        if mview:
            
            # resample
            self.grids = self.initgrid(1, sdim=spatial, tdim=crop)
            
            # crop z
            trange = self.model.trange
            gridslonger = self.grids[0]
            gridzlonger = (gridslonger[:, :, :, :, 2:] + 1) / 2
            gridzshorter = gridzlonger / ((trange / 2) / wall_size)
            gridzshorter = gridzshorter * 2 - 1
            self.gridsshorter = torch.cat([gridslonger[:, :, :, :, :2], gridzshorter], dim=4)
    
    def change_bin_len(self, bin_len):
        assert self.mode == 'fk' or  self.mode == 'fkfast'
        self.model.change_bin_len(bin_len)
        
    def todev(self, dev, dnum):
        self.model.todev(dev, dnum)
        
        if self.mview:
            self.grids_todev = [d.to(dev) for d in self.grids]
            self.gridsshorter_todev = self.gridsshorter.to(dev)
    
    def initgrid(self, channel, tdim, sdim):
        
        self.mats = init_mats(False)
        
        tgt_size = torch.Size((1, channel, tdim, sdim, sdim))
        p_in_bx3x4 = self.mats[0:1]
        grid = F.affine_grid(p_in_bx3x4, tgt_size, align_corners=self.align_corners)
        
        def rev(grid):
            grid[:, :, :, :, 1] = -grid[:, :, :, :, 1]
            grid[:, :, :, :, 2] = -grid[:, :, :, :, 2]
            return grid
        
        grid = rev(grid)
        grids = []
        for i in range(26):
            j = i
            mat = self.mats[j][:3, :3]
            gridnew = torch.matmul(grid, mat)
            gridnew = rev(gridnew)
            grids.append(gridnew)
        
        return grids
    
    def forward(self, feture_bxdxtxhxw, tbes, tens, views=None):
        
        volumn_BxDxTxHxW, LPC_mat, tv_res, alpha = self.model(feture_bxdxtxhxw, tbes, tens)
        
        if views is None:
            return volumn_BxDxTxHxW, [LPC_mat, tv_res, alpha]
        else:
            assert self.mview
            
            volumnz_BxDxTxHxW = F.grid_sample(volumn_BxDxTxHxW, self.gridsshorter_todev.repeat(bnum, 1, 1, 1, 1), \
                                              mode='bilinear', padding_mode='zeros', \
                                              align_corners=self.align_corners)
            
            grid_bxdxhxwx3 = torch.cat([self.grids_todev[vid] for vid in views])
            volumnz_BxDxTxHxW = F.grid_sample(volumnz_BxDxTxHxW, grid_bxdxhxwx3, \
                                              mode='bilinear', padding_mode='zeros', \
                                              align_corners=self.align_corners)
            
            return volumnz_BxDxTxHxW, alpha


######################################3333
def gather_numpy2(data, dim, index):
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = data.shape[:dim] + data.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) + 
                         ", all dimensions of index and data should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(data, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


def gather_numpy(a, dim, index):
    expanded_index = [index if dim == i else np.arange(a.shape[i]).reshape([-1 if i == j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
    
    re = []
    
    a_dxhxw = a
    index_1xhxw = index
    for i in range(a_dxhxw.ndim):
        if dim == i:
            tmp = index_1xhxw
        else:
            tmp = np.arange(a.shape[i])
            reshapdim = [-1 if i == j else 1 for j in range(a_dxhxw.ndim)]
            tmp = tmp.reshape(reshapdim)
        re.append(tmp)
    
    re1 = a[re]
    re2 = a[expanded_index]
    return re1