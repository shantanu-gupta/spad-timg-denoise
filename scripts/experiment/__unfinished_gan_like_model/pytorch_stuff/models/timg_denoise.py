""" timg_denoise.py
"""

import numpy as np
import torch
import torch.nn as nn

class Timg_DenoiseNet_LinT_1Layer(nn.Module):
    def __init__(self):
        super(Timg_DenoiseNet_LinT_1Layer, self).__init__()
        self.C = 64
        self.K = 13
        self.centre = 3/255.0
        self.scale = 2.0

        self.conv1 = nn.Conv2d(1, self.C, self.K, padding=self.K//2)
        self.norm1 = nn.BatchNorm2d(self.C)
        self.relu1 = nn.ReLU()

        self.comb = nn.Conv2d(self.C, 1, 1)
        # just need time to be above the minimum
        self.fix_range_t = nn.Threshold(1/255.0, 1/255.0)

        # nn.init.dirac_(self.conv1.weight)

    def forward(self, t):
        t = self.scale * (t - self.centre)
        t = self.conv1(t)
        t = self.relu1(t)

        t = self.comb(t)
        t = self.fix_range_t(t)

        return t

class Timg_DenoiseNet_LinT(nn.Module):
    def __init__(self, Tmin=1e-3, Tmax=1e3):
        super(Timg_DenoiseNet_LinT, self).__init__()

        self.C = 64
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Tmid = 1
        self.Tscale = self.Tmid - self.Tmin

        self.conv1 = nn.Conv2d(1, self.C, 5, padding=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(self.C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(self.C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(self.C)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn5 = nn.BatchNorm2d(self.C)
        self.relu5 = nn.ReLU()

        self.comb = nn.Conv2d(self.C, 1, 1)

        self.fix_range1 = nn.Hardtanh(min_val=self.Tmin, max_val=self.Tmax)
        self.fix_range2 = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, t):
        t = (1.0/self.Tscale) * (t - self.Tmid)

        t = self.conv1(t)
        t = self.relu1(t)

        t = self.conv2(t)
        t = self.bn2(t)
        t = self.relu2(t)

        t = self.conv3(t)
        t = self.bn3(t)
        t = self.relu3(t)

        t = self.conv4(t)
        t = self.bn4(t)
        t = self.relu4(t)

        t = self.conv5(t)
        t = self.bn5(t)
        t = self.relu5(t)

        t = self.comb(t)

        t = self.Tmid + (self.Tscale * t)
        t = self.fix_range1(t)
        y = torch.pow(t, -1)
        y = self.fix_range2(y)
        return y

class Timg_DenoiseNet(nn.Module):
    def __init__(self, Tmin=1e-3, Tmax=1e3):
        super(Timg_DenoiseNet, self).__init__()

        self.C = 64
        self.Tmin = np.log(Tmin)
        self.Tmax = np.log(Tmax)

        # self.conv1 = nn.Conv2d(1, self.C, 3, padding=1)
        self.conv1 = nn.Conv2d(1, self.C, 5, padding=2)
        # self.conv1 = nn.Conv2d(1, self.C, 7, padding=3)
        # self.conv1 = nn.Conv2d(1, self.C, 9, padding=4)
        # self.conv1 = nn.Conv2d(1, self.C, 11, padding=5)
        # self.conv1 = nn.Conv2d(1, self.C, 13, padding=6)
        # self.conv1 = nn.Conv2d(1, self.C, 15, padding=7)
        # self.conv1 = nn.Conv2d(1, self.C, 17, padding=8)
        # self.conv1 = nn.Conv2d(1, self.C, 19, padding=9)
        # self.conv1 = nn.Conv2d(1, self.C, 21, padding=10)
        self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(self.C, self.C, 3, padding=1)
        self.conv2 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(self.C)
        self.relu2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(self.C, self.C, 3, padding=1)
        self.conv3 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(self.C)
        self.relu3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(self.C, self.C, 3, padding=1)
        self.conv4 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(self.C)
        self.relu4 = nn.ReLU()

        # self.conv5 = nn.Conv2d(self.C, self.C, 3, padding=1)
        self.conv5 = nn.Conv2d(self.C, self.C, 5, padding=2)
        self.bn5 = nn.BatchNorm2d(self.C)
        self.relu5 = nn.ReLU()

        self.comb = nn.Conv2d(self.C, 1, 1)

        self.fix_range1 = nn.Hardtanh(min_val=self.Tmin, max_val=self.Tmax)
        self.fix_range2 = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, t):
        logt = torch.log(t)

        logt = self.conv1(logt)
        logt = self.relu1(logt)

        logt = self.conv2(logt)
        logt = self.bn2(logt)
        logt = self.relu2(logt)

        logt = self.conv3(logt)
        logt = self.bn3(logt)
        logt = self.relu3(logt)

        logt = self.conv4(logt)
        logt = self.bn4(logt)
        logt = self.relu4(logt)

        logt = self.conv5(logt)
        logt = self.bn5(logt)
        logt = self.relu5(logt)

        logt = self.comb(logt)

        logt = self.fix_range1(logt)
        t = torch.exp(logt)
        y = torch.pow(t, -1)
        y = self.fix_range2(y)
        return y
