import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import exp


class RMSE(nn.Module):
    """ Root mean square error """

    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, im1, im2, mask=None):
        assert im1.shape == im2.shape, 'input shape mismatch'

        rmse = (im1 - im2).pow(2)
        if mask is not None:
            rmse = (rmse * mask).flatten(1).sum(-1)
            rmse = rmse / mask.flatten(1).sum(-1)
        rmse = rmse.mean().sqrt()
        return rmse


class PSNR(nn.Module):
    """ Peak signal-to-noise ratio """

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, im1, im2):
        assert im1.shape == im2.shape, 'input shape mismatch'

        B = im1.size(0)
        mse = (im1 - im2).pow(2).mean() + 1e-8
        psnr = -10 * mse.log10()
        return psnr

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, prediction, target):
        distance = torch.cdist(prediction, target, p=2)
        return torch.mean(distance)
    
    
def cal_psnr(im1, im2):
    assert im1.shape == im2.shape, 'input shape mismatch'

    B = im1.size(0)
    mse = (im1 - im2).pow(2).mean() + 1e-8
    psnr = -10 * mse.log10()
    return psnr
    
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def cal_mad(x):
    median_val = torch.median(x)
    mad_val = torch.median(torch.abs(x - median_val))
    return mad_val


class MAD(nn.Module):
    def __init__(self):
        super(MAD, self).__init__()

    def forward(self, im1, im2, mask=None):
        assert im1.shape == im2.shape, 'input shape mismatch'

        rmse = torch.abs(im1 - im2)
        if mask is not None:
            rmse = (rmse * mask).flatten(1).sum(-1)
            rmse = rmse / mask.flatten(1).sum(-1)
        rmse = rmse.mean()
        return rmse


def crop_to_cal(img_gt,pred):
    int_mask = img_gt >0
    int_mask = int_mask.float()
    box = int_mask[0,0,:,:]  
    H = box.shape[0]
    x1=y1=x2=y2=0
    for i in range(H):
        if box[i,:].mean().item()!=0:
            y1=i
            break
    for i in range(H):
        # print(i)
        if box[H-1-i,:].mean().item()!=0:
            y2=H -1 -i
            break
    for i in range(H):
        if box[:,i].mean().item()!=0:
            x1=i
            break
    for i in range(H):
        if box[:,H-1-i].mean().item()!=0:
            x2= H -1 -i
            break
    padding = 40 
    if y1-padding>=0: y1-=padding
    if y2+padding<=H: y2+=padding
    if x1-padding>=0: x1-=padding
    if x2+padding<=H: x2+=padding

    eval_mask = np.zeros((H,H))
    eval_mask[y1:y2, x1:x2] = 1
    eval_mask = eval_mask.astype(np.bool_)
    eval_h = y2-y1 
    eval_w = x2-x1
    img_gt_np = img_gt.detach().clone().cpu().numpy()   
    pred_np = pred.detach().clone().cpu().numpy() 
    disp_np = np.reshape(pred_np[:,:,eval_mask], [pred_np.shape[0],pred_np.shape[1],eval_h, eval_w])
    img_gt_np = np.reshape(img_gt_np[:,:,eval_mask], [img_gt_np.shape[0],img_gt_np.shape[1],eval_h, eval_w])
    
    disp_tensor = torch.from_numpy(disp_np).cuda()
    gt_tensor = torch.from_numpy(img_gt_np).cuda()

    return gt_tensor, disp_tensor