# multi gpus version
from __future__ import print_function
from __future__ import division
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from pathlib import Path

import numpy as np
import cv2
import argparse
import logging
import datetime
import random
from tqdm import tqdm
import scipy.io as io
from torch.utils.tensorboard import SummaryWriter

from DL_inference.inference.config import get_args
import DL_inference.utils.DatasetUtils as utils
import DL_inference.utils.metric as metric
from DL_inference.inference.testrealcall_split import testreal

args = get_args()

import sys
from DL_inference.network7_256.selfnet import DeepVoxels as Net

from DL_inference.utils.LFEDataset_complicate import LFEDataset



def save_checkpoint(n_iter, epoch, model, optimer, file_path):
    state = {}
    state["n_iter"] = n_iter
    state["epoch"] = epoch
    state["lr"] = optimer.param_groups[0]["lr"]
    state["state_dict"] = model.state_dict()
    state["optimizer"] = optimer.state_dict()

    if utils.is_main_process():
        torch.save(state, file_path)
        
def add_poisson_noise(signal, desired_snr_db):
    signal_power = torch.mean(torch.mul(signal, signal))
    desired_snr = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / desired_snr
    noise_lambda = noise_power / 2
    noise = torch.distributions.Poisson(noise_lambda)
    noisy_signal = signal + noise.sample(signal.shape)

    return noisy_signal


def set_test_model_with_ckp(model, epoch, ckp_dir):
    ckp_dir = ckp_dir.joinpath(str(epoch) + '.pth')
    model.load_state_dict(torch.load('%s' % (ckp_dir))["state_dict"])
    return model


def cross_tv_loss(x, x_prime):
    vertical_diff = x_prime[:, :, :, :-1, :] - x[:, :, :, 1:, :]
    horizontal_diff = x_prime[:, :, :, :, :-1] - x[:, :, :, :, 1:]
    loss = torch.sum(torch.sqrt(vertical_diff ** 2 + horizontal_diff ** 2 + 1e-6))
    return loss

print("Start setting dir and logger ...")
base_dir = Path(args.basefolder)
log_dir = base_dir.joinpath('log')
log_dir.mkdir(exist_ok=True)
log_dir = log_dir.joinpath('LFE')
log_dir.mkdir(exist_ok=True)
timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
log_dir = log_dir.joinpath(timestr)
log_dir.mkdir(exist_ok=True)

ckp_dir = base_dir.joinpath('checkpoint')
ckp_dir.mkdir(exist_ok=True)
ckp_dir = ckp_dir.joinpath('LFE')
ckp_dir.mkdir(exist_ok=True)
timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
ckp_dir = ckp_dir.joinpath(timestr)
ckp_dir.mkdir(exist_ok=True)

writer_loc = os.path.join(log_dir, 'tensorboard_logs_%s_final/%s' % (args.dataset_name , args.exp_name))
writer = SummaryWriter(writer_loc)

def log_string(str):
    logger.info(str)
    print(str)

def log_string_without_print(str):
    logger.info(str)


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, timestr))  # random time file name
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_string('PARAMETER ...')
log_string(args)
print("Set logger success!")

# Make experiments reproducible
print("Start loading model...")
# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1234569527)

mode = args.mode
in_dim = args.in_dim
out_dim = in_dim * 2
model = Net(img_sidelength=256,
            in_channels=in_dim,
            out_channels=out_dim,
            nf0=args.dim,
            grid_dim=args.grid,
            mode=mode,
            bin_len=0.01,
            )
model_test = Net(img_sidelength=256,
            in_channels=in_dim,
            out_channels=out_dim,
            nf0=args.dim,
            grid_dim=args.grid,
            mode=mode,
            bin_len=0.0096
            )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.todev(device)

model_test.to(device)
model_test.todev(device)
print("Load model and move to multi GPUs success!")

print("Start load optimizer")
params = filter(lambda p: p.requires_grad, model.parameters())
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(params, lr=args.lr_rate)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
print("Load optimizer success!")

# modelsvdir = '/data1/sdsun/sdsun/code/NLOS_adp/results/LDC_8/checkpoint/LFE/2024-11-17_12-54/49.pth'
# model.load_state_dict(torch.load('%s' % (modelsvdir))["state_dict"])
# optimizer.load_state_dict(torch.load('%s' % (modelsvdir))["optimizer"])

# load data
print("Loading training and validation data...")
if args.dataset_name == 'bike':
    folder_path = [args.datafolder]
    shineness = [0]
    train_data = LFEDataset(root=folder_path,  # dataset root directory
                            shineness=shineness,
                            for_train=True,
                            ds=1,  # temporal down-sampling factor
                            clip=512,  # time range of histograms
                            size=256,  # measurement size (unit: px)
                            scale=1,  # scaling factor (float or float tuple)
                            background=[0.05, 2],  # background noise rate (float or float tuple)
                            target_size=256,  # target image size (unit: px)
                            target_noise=0.00,  # standard deviation of target image noise
                            color='gray',  # color channel(s) of target image
                            sp_ds=args.sp_ds_scale,  # spatial resolution downsample
                            mask=args.mask)

    val_data = LFEDataset(root=folder_path,  # dataset root directory
                          shineness=shineness,
                          for_train=False,
                          ds=1,  # temporal down-sampling factor
                          clip=512,  # time range of histograms
                          size=256,  # measurement size (unit: px)
                          scale=1,  # scaling factor (float or float tuple)
                          background=[0.05, 2],  # background noise rate (float or float tuple)
                          target_size=256,  # target image size (unit: px)
                          target_noise=0.00,  # standard deviation of target image noise
                          color='gray',  # color channel(s) of target image
                          sp_ds=args.sp_ds_scale,  # spatial resolution downsample
                          mask=args.mask)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = DistributedSampler(train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = SequentialSampler(val_data)
    else:
        train_sampler = RandomSampler(train_data)
        val_sampler = SequentialSampler(val_data)
    
    # set worker
    def worker_init(worker_id):
        worker_seed = torch.initial_seed() % 2**32  # get current setting seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_loader = DataLoader(train_data,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              drop_last=True,
                              worker_init_fn=worker_init)
    valid_loader = DataLoader(val_data,
                              sampler=val_sampler,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    print(len(train_data))
    print(len(val_data))
    print("Set dataloader success!")


def rmnan(tensor):
    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0
    return tensor


for p in model.parameters():
    if p.requires_grad:
        p.register_hook(lambda grad: rmnan(grad))

log_string("Start training...")
start_epoch = 0
n_iter = 0
items = ["LOSS_ALL_L2", "LOSS_IMG_L2", "LOSS_DEPTH_L2", "PSNR", "SSIM", "RMSE", "MAD"]
train_loss = {items[0]: [], items[1]: [], items[2]: []}
val_loss = {items[0]: [], items[1]: [], items[2]: [], items[3]: []}
dtype = torch.cuda.FloatTensor
mse = nn.MSELoss()
datafolder = "/data1/sdsun/sdsun/dataset/NLOS/real_world" # real world data folder
log_string_without_print("complicate training version")
cal_ssim = metric.SSIM().cuda()
cal_mad = metric.MAD().cuda()
for epoch in range(start_epoch, args.num_epoch):
    print("Epoch: {}, LR: {}".format(epoch, optimizer.param_groups[0]["lr"]))
    train_loss = {items[0]: [], items[1]: [], items[2]: []}
    val_loss = {items[0]: [], items[1]: [], items[2]: [], items[3]: [], items[4]: [], items[5]: [], items[6]: []}
    model.train()
    n_iter = 0

    for sample in tqdm(train_loader):
        M_mea = sample["ds_meas"].type(dtype).contiguous()
        dep_gt = sample["dep_gt"].type(dtype).squeeze(1).squeeze(1).contiguous()
        img_gt = sample["img_gt"].type(dtype).squeeze(1).squeeze(1).contiguous()
        
        M_mea = add_poisson_noise(M_mea, 30)
        
        IntandDep_re, alpha = model(M_mea, [0], [256])
        img = (IntandDep_re[:, 0, ...]+ 1) / 2
        depth = IntandDep_re[:, 1, ...]

        loss_img = mse(img, img_gt)
        loss_depth = mse(depth, dep_gt)
        loss = loss_img + loss_depth
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()
        n_iter += 1

        if n_iter % 30 == 0:
        # write loss log
            log_string_without_print("Epoch:%d, loss_all: %f, loss_img:%f, loss_depth:%f, alpha:%f" % (epoch, loss.item(),
                                                                                             loss_img.item(),
                                                                                             loss_depth.item(),
                                                                                             alpha[2].item()))
        train_loss[items[0]].append(loss.item())
        train_loss[items[1]].append(loss_img.item())
        train_loss[items[2]].append(loss_depth.item())

    log_string_without_print("Train: Epoch:%d, loss_all: %f, loss_img:%f" % (epoch, float(sum(train_loss[items[0]]))/float(len(train_loss[items[0]]))
                                                         , float(sum(train_loss[items[1]]))/float(len(train_loss[items[1]]))))
    
    savepath = str(ckp_dir) + '/' + str(epoch) + '.pth'
    save_checkpoint(n_iter, epoch, model, optimizer, file_path=savepath)
    print("End of epoch: {}. Checkpoint saved!".format(epoch))
    if (epoch + 1) % args.save_per_epochs == 0:
        print("Start dataset validation...")
        with torch.no_grad():
            model.eval()
            l_all = []
            l_kl = []
            l_l2 = []
            for sample in tqdm(valid_loader):
                M_mea = sample["ds_meas"].type(dtype)
                dep_gt = sample["dep_gt"].type(dtype).squeeze(1).squeeze(1)
                img_gt = sample["img_gt"].type(dtype).squeeze(1).squeeze(1)
                
                M_mea = add_poisson_noise(M_mea, 30)
                
                IntandDep_re, alpha = model(M_mea, [0], [256])
                img = (IntandDep_re[:, 0, ...]+ 1) / 2
                depth = IntandDep_re[:, 1, ...]

                loss_im = torch.sqrt(mse(img, img_gt))  # RMSE
                loss_depth = torch.sqrt(mse(depth, dep_gt))
                loss = loss_im + loss_depth

                img_gt, img = metric.crop_to_cal(img_gt.unsqueeze(1), img.unsqueeze(1))
                dep_gt, depth = metric.crop_to_cal(dep_gt.unsqueeze(1), depth.unsqueeze(1))
                im_psnr = metric.cal_psnr(img, img_gt)
                im_ssim = cal_ssim(img, img_gt)
                dep_mad = cal_mad(depth, dep_gt)
                dep_rmse = torch.sqrt(mse(depth, dep_gt))

                optimizer.zero_grad()
                val_loss[items[0]].append(loss.item())
                val_loss[items[1]].append(loss_im.item())
                val_loss[items[2]].append(loss_depth.item())
                val_loss[items[3]].append(im_psnr.item())
                val_loss[items[4]].append(im_ssim.item())
                val_loss[items[5]].append(dep_rmse.item())
                val_loss[items[6]].append(dep_mad.item())

            print("Validation complete!")
            log_string_without_print("Train: Epoch:%d, loss_all: %f, loss_img:%f, loss_depth:%f, psnr:%f, ssim:%f, rmse:%f, mad:%f" % (epoch, 
                                                                                                                                       float(sum(val_loss[items[0]]))/float(len(val_loss[items[0]])), 
                                                                                                                                       float(sum(val_loss[items[1]]))/float(len(val_loss[items[1]])),
                                                                                                                                       float(sum(val_loss[items[2]]))/float(len(val_loss[items[2]])), 
                                                                                                                                       float(sum(val_loss[items[3]]))/float(len(val_loss[items[3]])), 
                                                                                                                                       float(sum(val_loss[items[4]]))/float(len(val_loss[items[4]])), 
                                                                                                                                       float(sum(val_loss[items[5]]))/float(len(val_loss[items[5]])), 
                                                                                                                                       float(sum(val_loss[items[6]]))/float(len(val_loss[items[6]]))))
            
            print("Start real world data validation...")
            model_test = set_test_model_with_ckp(model_test, epoch, ckp_dir) # change bin width to 0.0096 for real data test
            testreal(model_test, epoch, args.tres, args.in_dim, device, datafolder, '/data1/sdsun/sdsun/code/NLOS_adp/results/LDC_8_resume/real_world')
            print("Validate in real world data success!")
