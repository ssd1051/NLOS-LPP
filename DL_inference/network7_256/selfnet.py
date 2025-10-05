import torch
import torch.nn as nn
import numpy as np
from DL_inference.network7_256.customer_layers_3 import Transient2volumn, VisibleNet, Rendering

import sys
from DL_inference.utils_pytorch.tfmodule import diffmodule as lct


###########################################################################
def normalize(data_bxcxdxhxw):
    b, c, d, h, w = data_bxcxdxhxw.shape
    data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
    
    data_min = data_bxcxk.min(2, keepdim=True)[0]
    data_zmean = data_bxcxk - data_min
    
    # most are 0
    data_max = data_zmean.max(2, keepdim=True)[0]
    data_norm = data_zmean / (data_max + 1e-15)
    
    return data_norm.view(b, c, d, h, w)


################################################################
class DeepVoxels(nn.Module):

    def __init__(self,
                 nf0=16,
                 in_channels=3,
                 out_channels=3,
                 img_sidelength=256,
                 grid_dim=32,
                 bin_len=0.01,
                 wall_size=2.0,
                 mode='fk',
                 spatial=128,
                 tlen=256):
        
        super(DeepVoxels, self).__init__()
        
        self.spatial = spatial
        self.tlen = tlen
        # assert sres == tres
        
        ########################################################
        basedim = nf0
        self.basedim = basedim
        # assert not raytracing
        
        self.downnet = Transient2volumn(nf0=basedim, in_channels=in_channels)
        
        print('bin_len %.7f' % bin_len)
        self.lct = lct(spatial=self.spatial, crop=self.tlen, bin_len=bin_len * 2, \
                       mode=mode, wall_size=wall_size)
        
        layernum = 0
        self.visnet = VisibleNet(nf0=basedim * 1 + 1, layernum=layernum)
        
        self.depth = True
        assert out_channels == 6 or out_channels == 2
        self.rendernet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=1)
        self.depnet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=1, isdep=True)
    
    def todev(self, dev):
        self.lct.todev(dev, self.basedim * 1 + 1)
                
    def noise(self, data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi

    def forward(self, input_voxel, tbes, tens):
        # tbes: bins count
        if False:
            noisedata = self.noise(input_voxel)
        else:
            noisedata = input_voxel
        
        ###############################
        data_norm = normalize(noisedata)
        
        tfre = self.downnet(data_norm)  # feature extraction
        
        tfre2, rest = self.lct(tfre, tbes, tens)
        
        # resize
        x = tfre2
        zdim = x.shape[2]
        # zdimnew = zdim * 100 // 128
        zdimnew = zdim
        x = x[:, :, :zdimnew]
        tfre2 = x
        
        tfre2 = nn.ReLU()(tfre2)
        tfre2 = normalize(tfre2)
        
        ######################################
        # unet 2 voxel
        tfflat = self.visnet(tfre2)
        
        # render
        rendered_img = self.rendernet(tfflat)
        
        dep_img = self.depnet(tfflat)
        
        rendered_img = torch.cat([rendered_img, dep_img], dim=1)

        rendered_img = torch.clamp(rendered_img, 0, 1)
        rendered_img = rendered_img * 2 - 1
        
        return rendered_img, rest
