from enum import EnumMeta
import os
import glob
from pickle import TRUE
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import random
import torch
import torch.nn.functional as F


def histogram_equalization_ignore_bg(image):
    non_bg_pixels = image[image > 0].flatten()
    
    hist, bins = np.histogram(non_bg_pixels, bins=np.linspace(0, 1, 257), range=(0, 1))
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    image_equalized = image_equalized.reshape(image.shape)
    
    image_equalized[image == 0] = 0
    
    return image_equalized


def list_file_path(root_path, fortrain=True, shineness=[0]):
    modeldirs = []
    testmodeldirs = []
    mea = []
    im = []
    de = []
    test_mea = []
    test_im = []
    test_de = []
    root_path = [root_path]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))

            for modeldir in modeldir_all[:2]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspadPDPS*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    de.append(path)
            for modeldir in modeldir_all[250:]:
                print(modeldir)
                rotdirs = glob.glob('%s/shin*' % (modeldir))[:1]
                testmodeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspadPDPS*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    test_de.append(path)

    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': testmodeldirs}
    if fortrain:
        return train_sample
    else:
        return test_sample


def list_file_path_bike(root_path, fortrain=TRUE, shineness=[0]):
    modeldirs = []
    test_modeldirs = []
    mea = []
    im = []
    de = []
    test_mea = []
    test_im = []
    test_de = []
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))
            modeldir_all = sorted(modeldir_all)
            modeldir_all.reverse()
            
            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))
                rotdirs = sorted(rotdirs)
                rotdirs.reverse()
                
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    de.append(path)

            for modeldir in modeldir_all[250:]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))
                rotdirs = sorted(rotdirs)
                rotdirs.reverse()
                test_modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    test_de.append(path)

    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': test_modeldirs}
    if fortrain:
        return train_sample
    else:
        return test_sample


def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)



class NLOSPoissonNoise:

    def __init__(self, background=0.1529):
        self.rate = background

    def __call__(self, x):
        if self.rate == 0:
            return x
        else:
            rate = self.rate
            poisson = torch.distributions.Poisson(rate)
            x += poisson.sample(x.shape)
            return x

    def __repr__(self):
        return 'Introduce shot noise and background noise to raw histograms'


class NLOSRandomScale:

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, x):
        if isinstance(self.scale, (int, float)):
            x *= self.scale
        elif isinstance(self.scale, (list, tuple)):
            scale = random.random() * (self.scale[1] - self.scale[0])
            scale += self.scale[0]
            x *= scale
        return x

    def __repr__(self):
        return 'Randomly scale raw histograms'


class NLOSCompose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transforms(x)
        return x

    def __repr__(self):
        repr_str = ''
        for t in self.transforms:
            repr_str += t.__repr__() + '\n'
        return repr_str


def get_transform(background=0.1529):
    transform = NLOSPoissonNoise(background)
    transform = NLOSCompose(transform)
    return transform


class LFEDataset(Dataset):

    def __init__(
            self,
            root,  # dataset root directory
            shineness,  # data split ('train', 'val')
            for_train=True,
            ds=1,  # temporal down-sampling factor
            clip=512,  # time range of histograms
            size=128,  # measurement size (unit: px)
            scale=1,  # scaling factor (float or float tuple)
            background=0,  # background noise rate (float or float tuple)
            target_size=128,  # target image size (unit: px)
            target_noise=0,  # standard deviation of target image noise
            color='gray',  # color channel(s) of target image
            sp_ds=8,  # spatial resolution downsample,
            mask=False
    ):
        super(LFEDataset, self).__init__()
        self.root = root
        self.ds = ds
        self.clip = clip
        self.size = size
        self.target_size = target_size
        self.target_noise = target_noise
        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.data_list = list_file_path_bike(root, for_train, shineness)
        self.color = color
        self.sp_ds = sp_ds
        self.mask = mask

    
    def add_poisson_noise_per_slice(self, signal, desired_snr_db):
        signal = np.squeeze(signal)
        signal_power_per_slice = np.mean(signal ** 2, axis=(1, 2))

        desired_snr = 10 ** (desired_snr_db / 10)

        noise_power_per_slice = signal_power_per_slice / desired_snr

        noisy_signal = np.zeros_like(signal)

        for t in range(signal.shape[0]):
            noisy_signal[t] = signal[t] + np.random.poisson(noise_power_per_slice[t], signal[t].shape)

        return np.expand_dims(noisy_signal, axis=0)
    
    
    def add_poisson_noise(self, signal, desired_snr_db):
        signal_power = torch.mean(signal ** 2)
        desired_snr = 10 ** (desired_snr_db / 10)
        noise_power = signal_power / desired_snr

        noise_lambda = noise_power / 2 

        noise = torch.distributions.Poisson(noise_lambda)

        noisy_signal = signal + noise.sample(signal.shape)

        return noisy_signal


    def add_gaussian_noise_per_slice(self, signal, desired_snr_db):
        signal = np.squeeze(signal)
        signal_power_per_slice = np.mean(signal ** 2, axis=(1, 2))
        desired_snr = 10 ** (desired_snr_db / 10)

        noise_power_per_slice = signal_power_per_slice / desired_snr

        noise_sigma_per_slice = np.sqrt(noise_power_per_slice)

        noisy_signal = np.zeros_like(signal)

        for t in range(signal.shape[0]):
            noisy_signal[t] = signal[t] + np.random.normal(0, noise_sigma_per_slice[t], signal[t].shape)

        return np.expand_dims(noisy_signal, axis=0)
    
    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'mp4')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.reshape(-1, x.shape[1], x.shape[1], 1)
                x = x.transpose(3, 0, 1, 2)  # 1 600 256 256
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened()
                ims = []
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        ims.append(frame)
                    else:
                        break
                cap.release()
                raw = np.array(ims, dtype=np.float32) / 255.0
                raw = raw.transpose(3, 0, 1, 2)

                # raw = raw[:, :, ::2, :] + raw[:, :, 1::2, :]
                # raw = raw[:, :, :, ::2] + raw[:, :, :, 1::2]
                # raw = raw[:, :, ::2, :] + raw[:, :, 1::2, :]
                # raw = raw[:, :, :, ::2] + raw[:, :, :, 1::2]
                
            if raw.ndim == 3:
                raw = raw[None]

            # temporal down-sampling
            if self.ds > 1:
                c, t, h, w = raw.shape
                assert t % self.ds == 0
                raw = raw.reshape(c, t // self.ds, self.ds, h, w)
                raw = raw.sum(axis=2)

            # clip temporal range
            raw = raw[:, :self.clip]
            if raw.shape[0] == 3:
                if self.color == 'gray':
                    raw = 0.299 * raw[0:1] + 0.587 * raw[1:2] + 0.114 * raw[2:3]
                elif self.color == 'r':
                    raw = raw[0:1]
                elif self.color == 'g':
                    raw = raw[1:2]
                elif self.color == 'b':
                    raw = raw[2:3]
            else:
                if self.color == 'rgb':
                    raw = np.tile(raw, [3, 1, 1, 1])
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))

        # temporal down-sampling
        if self.ds > 1:
            c, t, h, w = raw.shape
            assert t % self.ds == 0
            raw = raw.reshape(c, t // self.ds, self.ds, h, w)
            raw = raw.sum(axis=2)

        c, t, h, w = raw.shape
        x = torch.from_numpy(raw.astype(np.float32))  # (1/3, t, h, w)
        raw = x
        return x, raw

    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.astype(np.float32)
                if ext == 'hdr':
                    x = x.reshape(x.shape[1], x.shape[1], 1)  # 256 256 1 1
                    x = cv2.resize(x, (self.target_size, self.target_size))
                    x = x.reshape(1, x.shape[1], x.shape[1])  # 128 128 1 1
                    x = x / np.max(x)
                else:
                    x = x / 255
        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))  # (h, w, v, 1/3)       #  256 256 1 1
        x = x.unsqueeze(0)
        return x

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # meters  fe 0.597m
                x = x[..., 0]
                # x = 1 - x.clip(0,1)
                x = x.clip(0, 1)
            x = cv2.resize(x, (self.target_size, self.target_size))  # (h, w)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        
        x = x.transpose(2, 3, 0, 1)
        x = torch.from_numpy(x.astype(np.float32))  # (v, 1, h, w)
        return x

    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas, raw_meas = self._load_meas(idx)
        images = self._load_image(idx)  
        depths = self._load_depth(idx)
        sample = {'ds_meas': meas, 'raw_meas': raw_meas, 'dep_gt': depths, 'img_gt': images}
        return sample