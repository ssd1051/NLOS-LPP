# -*- coding: utf-8 -*-
import argparse

eps = 1e-8


def get_args():
    parser = argparse.ArgumentParser(description='deblur')
    
    # dataset
    parser.add_argument('--datafolder', type=str, default='/data1/yueli/dataset/bike',
                        help='dataset folder')
    parser.add_argument('--dataset_name', type=str, default='bike',
                        help='num of  dataset')
    parser.add_argument('--datanum', type=int, default=10000,
                        help='num of  dataset')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers')
    parser.add_argument('--svfolder', type=str, default='.',
                        help='save folder')
    parser.add_argument('--basefolder', type=str, default='/data1/sdsun/sdsun/code/NLOS_adp/results/wwl/0.1',
                        help='base output folder')
    
    parser.add_argument('--h', type=int, default=64,
                        help='save folder')
    parser.add_argument('--w', type=int, default=64,
                        help='save folder')

    parser.add_argument('--sp_ds_scale', type=int, default=0,
                        help='spatial resolution downsample')
    parser.add_argument('--mask', type=bool, default=False,
                        help='mea * mask or not')
    parser.add_argument('--distributed', type=bool, default=False,
                        help='Turn on the distributed mode?')
    
    # training setting
    parser.add_argument('--num_epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('--epochsv', type=int, default=1,
                        help='epoch per model saving')
    parser.add_argument('--epochbe', type=int, default=-1,
                        help='initialize from existing model')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimizer name')
    parser.add_argument('--save_per_epochs', type=int, default=1,
                        help='how many epochs to save ckp')
    
    parser.add_argument('--iter_log', type=int, default=1,
                        help='iterations per log')
    parser.add_argument('--mode', type=str, default='phasor',
                         help='which model')
    parser.add_argument('--bs', type=int, default=1,
                        help='train bs')
    parser.add_argument('--dim', type=int, default=1,
                        help='feature dim')
    parser.add_argument('--in_dim', type=int, default=1,
                        help='input dim')
    parser.add_argument('--frame', type=int, default=512,
                        help='training frame')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='learning rate')
    
    
    #############################################3
    parser.add_argument('--tem', type=float, default=1.0,
                        help='temperaturer')
    parser.add_argument('--norm', type=str, default='in',
                        help='normalization')
    parser.add_argument('--normlct', type=int, default=0,
                        help='norm lct output or not')
    parser.add_argument('--addraw', type=int, default=0,
                        help='add raw data or not')
    parser.add_argument('--raytracing', type=int, default=0,
                        help='do we use raytracining')
    
    #############################
    parser.add_argument('--netfolder', type=str, default='utils_pytorch',
                        help='network structure')
    parser.add_argument('--netsvfolder', type=str, default='model6_bike',
                        help='network svfolder')
    parser.add_argument('--grid', type=int, default=128,
                        help='grid')
    parser.add_argument('--tres', type=int, default=2,
                        help='tres')
    parser.add_argument('--confocal', type=int, default=2,
                        help='confocal')
    parser.add_argument('--res0', type=int, default=0,
                        help='do we use res0')
    
    # distributed training setting
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--world_size", type=int, default=1, help="number of total machines")
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()

    return args

