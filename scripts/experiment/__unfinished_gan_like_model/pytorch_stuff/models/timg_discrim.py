""" timg_discrim.py
"""

import torch
import torch.nn as nn

class Timg_Discrim(nn.Module):
    def __init__(self):
        super(Timg_Discrim, self).__init__()
        self.C = 64

        # input is expected to be 512 x 512
        # 5 layers of downsampling gets it down to 16 x 16
        # self.conv1 = nn.Conv2d(1, self.C, 33, stride=16, padding=16)
        self.conv1 = nn.Conv2d(1, self.C, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(self.C, self.C, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(self.C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.C, self.C, 3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(self.C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(self.C, self.C, 3, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(self.C)
        self.relu4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(self.C, self.C, 3, stride=2, padding=1)
        self.norm5 = nn.BatchNorm2d(self.C)
        self.relu5 = nn.ReLU()

        self.comb6 = nn.Conv2d(self.C, 1, 1)
        self.pool6 = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.conv1(2 * (x - 0.5))
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)

        x = self.comb6(x)
        x = self.pool6(x)
        x = x.view(-1, 1)

        return x
