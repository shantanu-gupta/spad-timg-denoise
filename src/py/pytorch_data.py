""" src.py.pytorch_data
"""

import numpy as np
from scipy.signal import gaussian

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from src.py.pytorch_ops import GradientXY

import src.py.io as io_utils 

class TimgDataset(Dataset):
    """ TODO: rescaling, smoothed versions, averaging multiple timgs/logtimgs
    """
    def __init__(self, metadata, crop_size=256, crops_per_img=1,
                var_thresh=None, img_type='logtimg'):
        super(TimgDataset, self).__init__()
        self.metadata = metadata
        self.num_timgs_per_img = self.metadata['num-timgs']
        assert img_type in ['logtimg', 'timg']
        self.data_key = img_type
        self.gt_key = 'true-{}-path'.format(img_type)
        self.crop_size = crop_size
        self.crops_per_img = crops_per_img
        self.var_thresh = var_thresh
        self.max_var_attempts = 10
        self.img_metadata = self.metadata['image-metadata']
        # TODO: don't hardcode this!
        if img_type == 'timg':
            self.min_val = np.exp(-6)
            self.max_val = np.exp(20)
        elif img_type == 'logtimg':
            self.min_val = -6
            self.max_val = 20

    def __len__(self):
        return len(self.img_metadata) * self.num_timgs_per_img
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_timgs_per_img
        gt = io_utils.load_float_img(self.img_metadata[img_idx][self.gt_key])
        data_idx = idx % self.num_timgs_per_img
        data_md = self.img_metadata[img_idx]['timgs'][data_idx]
        data = np.clip(np.load(data_md[self.data_key]),
                    self.min_val, self.max_val)

        gt_input = np.empty((self.crops_per_img, 1,
                            self.crop_size, self.crop_size),
                        dtype=float)
        data_input = np.empty((self.crops_per_img, 1,
                                self.crop_size, self.crop_size),
                            dtype=float)
        H, W = gt.shape
        assert self.crop_size <= min(H, W), \
                '{} < ({}, {})'.format(self.crop_size, H, W)
        for c in range(self.crops_per_img):
            selected = False
            num_attempts = 0
            while not selected:
                i0 = np.random.randint(H - self.crop_size)
                j0 = np.random.randint(W - self.crop_size)
                gt_input[c,0,:,:] = gt[i0:i0+self.crop_size,
                                       j0:j0+self.crop_size]
                data_input[c,:,:,:] = data[i0:i0+self.crop_size,
                                           j0:j0+self.crop_size]
                num_attempts += 1
                if self.var_thresh is None \
                        or self.var_thresh == 0 \
                        or num_attempts == self.max_var_attempts:
                    selected = True
                else:
                    var_gt = np.var(gt_input[c,:,:])
                    selected = var_gt > self.var_thresh
        gt_tensor = torch.from_numpy(gt_input).type(torch.float)
        data_tensor = torch.from_numpy(data_input).type(torch.float)
        return {
                'data': data_tensor,
                'gt': gt_tensor}

class TimgDataset_2K(Dataset):
    """ Timgs with their corresponding radiance image.
    """
    def _gauss_kernel(self, sigma):
        w = gaussian(4*np.ceil(sigma) + 1, sigma, sym=True)
        w = np.outer(w, w)
        H, W = w.shape
        return torch.from_numpy(w).type(torch.float).view(1,1,H,W)

    def __init__(self, metadata, crop_size=256, use_logt=False,
                rescale_range=True, crops_per_img=1, num_timgs_per_img=1,
                var_thresh=None, add_grad_mag=False, add_smoothed_timg=False,
                num_avg=1):
        self.metadata = metadata
        self.num_timgs_per_img = num_timgs_per_img
        self.use_logt = use_logt 
        if self.use_logt:
            self.actual_t_key = 'log_mean_timg_path'
            self.timgs_key = 'logtimgs'
        else:
            self.actual_t_key = 'mean_timg_path'
            self.timgs_key = 'timgs'
        self.rescale_range = rescale_range
        self.crops_per_img = crops_per_img
        self.crop_size = crop_size
        assert isinstance(self.crop_size, int)

        self.var_thresh = var_thresh
        self.max_var_attempts = 10
        self.num_avg = num_avg
        assert self.num_avg <= self.num_timgs_per_img
        self.add_grad_mag = add_grad_mag
        self.add_smoothed_timg = add_smoothed_timg
        if self.add_smoothed_timg or self.add_grad_mag:
            self.grad_filt_sigma = 3
            self.grad_filt_kernel = self._gauss_kernel(self.grad_filt_sigma)
            if self.add_grad_mag:
                self.grad_module = GradientXY()

        self.Tmin_true = 1.0    # from using 8-bit [0, 1] imgs in sim
        self.Tmax_true = 255.0  # 
        self.Tmin_data = 1e-3   # clipped this way in simulation
        self.Tmax_data = 1e3    # 
        self.rescale_mult = None
        self.origin = None
        if rescale_range:
            if self.use_logt:
                # we will make the values go to [-1, 1]
                self.origin = 0
                min_val = np.log(self.Tmin_data)
                max_val = np.log(self.Tmax_data)
                self.rescale_mult = 2.0 / (max_val - min_val)
            else:
                # we will make the values go to [0, 1], kind of
                # Low values are REALLY common, so it seems better to put them
                # all near 0 instead of -1
                # Highest values will overshoot 1, but that's okay -- they are
                # infrequent
                # Dividing by a huge value (Tmax_data) throws off the scale too
                # much anyway
                self.origin = 0
                self.rescale_mult = 1.0 / self.Tmax_true

    def __len__(self):
        return len(self.metadata) * (self.num_timgs_per_img // self.num_avg)

    def __getitem__(self, idx):
        timg_input = np.empty((self.crops_per_img, self.num_avg,
                            self.crop_size, self.crop_size))
        gt_input = np.empty((self.crops_per_img, 1,
                            self.crop_size, self.crop_size))
        img_idx = idx // (self.num_timgs_per_img // self.num_avg)
        actual_t = np.load(self.metadata[img_idx][self.actual_t_key])
        timgs = np.empty((self.num_avg,) + actual_t.shape)
        for k in range(self.num_avg):
            timg_idx = \
                k + self.num_avg * (idx%(self.num_timgs_per_img//self.num_avg))
            timgs[k,:,:] = \
                np.load(self.metadata[img_idx][self.timgs_key][timg_idx]['path'])

        H, W = actual_t.shape
        assert self.crop_size <= min(H, W), \
                '{} < ({}, {})'.format(self.crop_size, H, W)
        for c in range(self.crops_per_img):
            selected = False
            num_attempts = 0
            while not selected:
                i0 = np.random.randint(H - self.crop_size)
                j0 = np.random.randint(W - self.crop_size)
                gt_input[c,0,:,:] = actual_t[i0:i0+self.crop_size,
                                             j0:j0+self.crop_size]
                timg_input[c,:,:,:] = timgs[:,i0:i0+self.crop_size,
                                              j0:j0+self.crop_size]
                num_attempts += 1
                if self.var_thresh is None \
                        or self.var_thresh == 0 \
                        or num_attempts == self.max_var_attempts:
                    selected = True
                else:
                    var_gt = np.var(gt_input[c,:,:])
                    selected = var_gt > self.var_thresh

        gt_tensor = torch.from_numpy(gt_input).type(torch.float)
        timg_tensor = torch.from_numpy(timg_input).type(torch.float)
        with torch.no_grad():
            # remove inf, goes up to only Tmax_true instead
            if self.rescale_range:
                if self.origin != 0:
                    gt_tensor = gt_tensor - self.origin
                    timg_tensor = timg_tensor - self.origin
                gt_tensor = self.rescale_mult * gt_tensor
                timg_tensor = self.rescale_mult * timg_tensor
            if self.num_avg > 1:
                timg_tensor = timg_tensor.mean(dim=1, keepdim=True)
                timg_tensor = torch.cat((timg_tensor,
                                        torch.full_like(
                                                timg_tensor,
                                                np.sqrt(1.0 / self.num_avg))),
                                        dim=1)
            if self.add_smoothed_timg or self.add_grad_mag:
                timg_sm = F.conv2d(timg_tensor, self.grad_filt_kernel)
                if self.add_smoothed_timg:
                    timg_tensor = torch.cat((timg_tensor, timg_sm), dim=1)
                if self.add_grad_mag:
                    timg_grad = self.grad_module(timg_sm)
                    timg_grad_mag = sum(torch.pow(timg_grad, 2), dim=1,
                                        keepdim=True)
                    eps = 1e-6
                    maxg, imax = torch.max(timg_grad_mag, dim=(2,3),
                                        keepdim=True)
                    timg_grad_mag /= (maxg + eps)
                    timg_tensor = torch.cat((timg_tensor, timg_grad_mag), dim=1)
        return {
                'timg': timg_tensor,
                'gt': gt_tensor}
