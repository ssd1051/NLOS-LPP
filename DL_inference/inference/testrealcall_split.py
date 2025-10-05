from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import numpy as np
import cv2

from scipy.ndimage import zoom


########################################################
def crop_video(video, framenum, res):
    if framenum == 512:
        return video, 0, 512
    
    sumdata = np.zeros((len(video),), dtype=np.float32)
    for i, frame in enumerate(video):
        sumdata[i] = np.sum(frame)
    
    idxs = []
    vals = []
    for i in range((len(video) - framenum) // res):
        idxs.append([i * res, i * res + framenum])
        vals.append(np.sum(sumdata[i * res:i * res + framenum]))
    # find max
    valmaxid = np.argmax(vals)
    tbe, ten = idxs[valmaxid]
    return video[tbe:ten], tbe, ten


def bilinear_upsample(input_array, target_shape):
    zoom_factors = [n / i for n, i in zip(target_shape, input_array.shape)]

    return zoom(input_array, zoom_factors, order=1)

def nearest_upsample(input_array, target_shape):
    zoom_factors = [n / i for n, i in zip(target_shape, input_array.shape)]

    return zoom(input_array, zoom_factors, order=0)


##############################################
import scipy.io as sio

def loadrealdata(datafolder, name, scale, tres=2, isdebug=False):
   
    matmaxref = 252.0
   
    tlen = 512 
    tbe2 = 0
    tbe = 0
    
    folders = ['%s/%s.mat' % (datafolder, name)]

    mats = []
    tbes = []
    tens = []
    name1 = ['bike0', 'dragon0', 'statue0', 'resolution0', 'discoball0', 'teaser0']
    name2 = ['bike', 'dragon', 'statue', 'resolution', 'discoball', 'teaser', 'outdoor_256', 'resolution_10min', 'teaser_10min', 'outdoor_10min_256', 'bike_10min', 'dragon_10min', 'statue_10min']
    name3 = ['letter', 'res256', 'letter_tu_10ms', 'letter_tu_1ms', 'letter_10ms', 'letter_1ms', 'ladder', 'ladder_01ms', 'ladder_1ms', 'ladder_10ms', 'res256_01ms', 'res256_1ms', 'res256_10ms', 'res256_100ms', '918_12_54resize1_9_128_']
    name4 = ['complicate4_256_1e10', 'complicate4_128_1e11', 'complicate7_128_1e11', 'complicate8_128_1e11', 'complicate9_128_1e11', 'complicate9_256_1e10', 'complicate10_128_1e11', 'complicate11_128_1e11', 'complicate11_128_1e10']
    for df in folders:
        mat = sio.loadmat(df)
        if name in name1:
            mat = np.array(mat['measlr'], dtype=np.float32)
            mat = np.transpose(mat, [2, 0, 1])
        elif name in name2:
            mat = np.array(mat['final_meas'], dtype=np.float32)
            mat = np.transpose(mat, [2, 0, 1])
        elif name in name3:
            mat = np.array(mat['data'], dtype=np.float32)
        elif name in name4:
            mat = np.array(mat['data'], dtype=np.float32)
            mat = mat[0]
            print(mat.shape)
        else:
            mat = np.array(mat['data'], dtype=np.float32)
            mat = np.transpose(mat, [2, 0, 1])
        

        if (mat.shape[-1] < 256):
            target_shape = (512, 256, 256)
            mat = nearest_upsample(mat, target_shape)
        
        matmax = np.max(mat)

        print('maxmat%s %.2f, maxmatbike %.2f, scale%.2f' % \
              (name, matmax, matmaxref, matmax / matmaxref))
        
        mat = mat / matmax
        
        matcrop, tbe, ten = crop_video(mat, tlen, tres)
        if tbe2 > tbe:
            matcrop = matcrop[tbe2 - tbe:]
            tbe = tbe2
        print('tbe %d, ten %d' % (tbe, ten))
        mats.append(matcrop)
        tbes.append(tbe)
        tens.append(ten)
        
        return mats, tbes, tens


def testreal(model, epoch, tres=2, in_dim=1, dev='cuda', datafolder='.', svdir='.'):
    model.eval()
    
    ims = []
    
    for ni, name in enumerate(['resolution_10min', 'teaser_10min', 'outdoor_10min_256', 'bike_10min', 'dragon_10min', 'statue_10min', 'resolution', 'teaser', 'outdoor_256', 'man_re', 'deer_re_16s', 'mandeer2_re_160s', 'new_manletter_re', 'res256', 'ladder']):
        
    
        ims = []
        mats, tbes, tens = loadrealdata(datafolder, name, scale=-1, tres=tres, isdebug=False)
        mat0 = mats[0]

        if True:
            mat = mat0
            datanp_txhxw = mat

            tbe = tbes[0]
            ten = tens[0]
            
            tbe = [tbe // tres]
            ten = [ten // tres]
    
            data_bxcxdxhxw = torch.from_numpy(datanp_txhxw).unsqueeze(0).unsqueeze(0).repeat(1, in_dim, 1, 1, 1)
            
            data_bxcxdxhxw = data_bxcxdxhxw.to(dev)
            
            if torch.min(data_bxcxdxhxw) >= 0:
                print('data, max %.5f, min %.5f' % (torch.max(data_bxcxdxhxw), torch.min(data_bxcxdxhxw)))
            
            output, _ = model(data_bxcxdxhxw, tbe, ten)
            w = output.shape[-1]
            img = ((output[:, :1, ...]+1)/2)
            depth = (output[:, 1:, ...])
            
            tmpmax = img.max()
            img = img / (tmpmax.view(-1, 1, 1, 1) + 1e-8)
            data = np.concatenate([img.detach().cpu().numpy(), depth.detach().cpu().numpy()], axis=3)
            ims.append(data)
    
            ims = np.concatenate(ims, axis=0)
            datanum = len(ims)
            row = 1
            a = np.zeros([0, w * 2 * row, in_dim])
            for i in range(datanum // row):
                imslice = [ims[d:d + 1] for d in range(i * row, i * row + row)]  # ims[i * 4:i * 4 + 4]
                imslice = np.concatenate(imslice, axis=3)
                a = np.concatenate((a, np.transpose(imslice[0], [1, 2, 0])), axis=0)
            im = a * 255
            cv2.imwrite('%s/testreal-%s-%d.png' % (svdir, str(name), epoch), im)