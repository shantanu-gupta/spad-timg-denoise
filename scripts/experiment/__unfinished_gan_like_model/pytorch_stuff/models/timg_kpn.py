""" timg_kpn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import gaussian

class Timg_KPN(nn.Module):
    def _convf_block(self, l_idx, use_leaky_relu=False):
        if l_idx == 0:
            Cin = self.C0
        else:
            Cin = self.C[l_idx-1]
        Cout = self.C[l_idx]
        parts = []
        if l_idx > 0:
            parts.append(nn.AvgPool2d(2))
        parts.append(nn.Conv2d(Cin, Cout, self.K, padding=self.K//2))
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        parts.append(nn.GroupNorm(2**l_idx, Cout))
        if use_leaky_relu:
            parts.append(nn.LeakyReLU())
        else:
            parts.append(nn.ReLU())
        block = nn.Sequential(*parts)
        return block

    def _convb_block(self, l_idx, use_leaky_relu=False):
        Cin = self.C[l_idx+1] + self.C[l_idx]
        Cout = self.C[l_idx]
        parts = []
        parts.append(nn.Conv2d(Cin, Cout, self.K, padding=self.K//2))
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        parts.append(nn.GroupNorm(2**l_idx, Cout))
        if use_leaky_relu:
            parts.append(nn.LeakyReLU())
        else:
            parts.append(nn.ReLU())
        block = nn.Sequential(*parts)
        return block

    def __init__(self, C0=1, C1=128, K=3, Kout=7, layers=5, use_logt=False):
        super(Timg_KPN, self).__init__()
        self.C0 = C0
        self.C1 = C1
        self.layers = layers
        self.K = K
        self.Kout = Kout
        self.C = [(2 ** k) * C1 for k in range(layers)]
        self.use_logt = use_logt
        self.Tmin_true = 1
        self.Tmax_true = 255
        if self.use_logt:
            # assume logT is normalized [-1, 1]
            # original data is [-3*log(10), 3*log(10)]
            # bias correction is 0.577 for logT, so we rescale appropriately
            # self.bias_correction = 0.577 / (3*np.log(10))
            self.bias_correction = 0.5722 / (3*np.log(10))
            # Making this trainable is tempting, but needs some thinking
            # self.bias_correction = nn.Parameter(
            #                         torch.Tensor([0.5722 / (3*np.log(10))])\
            #                                 .type(torch.float))
            # just want t >= Tmin in the end
            self.threshold = np.log(self.Tmin_true) / (3*np.log(10))
        else:
            # assume t is normalized [0, 1]
            # this implements saturation to [0, Tmax/Tmin]
            self.threshold = self.Tmin_true / self.Tmax_true

        # self._use_leaky_relu = self.Kout == 21 # HACK!
        self._use_leaky_relu = False
        self.convf_blocks = nn.ModuleList([self._convf_block(
                                            k, 
                                            use_leaky_relu=self._use_leaky_relu)
                                        for k in range(self.layers)])
        self.upsamplers = nn.ModuleList([nn.Upsample(scale_factor=2,
                                                    mode='bicubic',
                                                    align_corners=False)
                                        for k in range(self.layers-1)])
        self.convb_blocks = nn.ModuleList([self._convb_block(
                                            k,
                                            use_leaky_relu=self._use_leaky_relu)
                                        for k in range(self.layers-1)])

        # self.comb = nn.Conv2d(self.C1, self.Kout ** 2, 1)
        self.comb = nn.Conv2d(self.C1, self.Kout ** 2, self.Kout,
                            padding=self.Kout//2)
        self.norm_comb = nn.GroupNorm(1, self.Kout**2, affine=True)
        # hack!!!
        self.wscale = np.pi * (self.Kout ** -2)
        # Making this trainable is tempting, but needs some thinking
        # self.wscale = nn.Parameter(
        #                 torch.Tensor([2.4 * (self.Kout ** -2)])\
        #                         .type(torch.float))
        self.makewnonneg = nn.ReLU()
        self.unfold = nn.Unfold(self.Kout, padding=self.Kout//2)
        self.fix_range_t = nn.Threshold(self.threshold, self.threshold) 

        # initialize comb to a Gaussian filter-like state
        with torch.no_grad():
            sigma = min(self.Kout/4, 2) # usually, any more is just too blurry
            wbmin = 0.05
            wb = np.maximum(gaussian(self.Kout, sigma), wbmin)
            wbias_scale = torch.from_numpy(np.outer(wb, wb)).type(torch.float)
            wbias_scale = F.unfold(wbias_scale.view(1,1,self.Kout,self.Kout),
                                self.Kout).squeeze()
            for k in range(self.Kout ** 2):
                self.comb.weight[k,:,:,:].mul_(wbias_scale[k])

    def forward(self, t):
        t_in = t[:,0:1,:,:]
        ts = []
        for k in range(self.layers):
            t = self.convf_blocks[k](t)
            if k != self.layers - 1: # don't need last downsampled t separately
                ts.append(t)
        for k in range(self.layers-2, -1, -1):
            t = self.upsamplers[k](t)
            t = torch.cat((t, ts[k]), dim=1)
            t = self.convb_blocks[k](t)

        w = self.comb(t)
        w = self.wscale * self.norm_comb(w)
        w = self.makewnonneg(w) # w is an N x K^2 x H x W set of weights

        tblocks = self.unfold(t_in).view_as(w)
        tsm = (w * tblocks).sum(dim=1, keepdim=True)
        if self.use_logt:
            # correction
            tsm = tsm + self.bias_correction
        tsm = self.fix_range_t(tsm)
        return tsm, w

