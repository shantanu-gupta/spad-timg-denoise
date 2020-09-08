""" src.py.pytorch_ops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

class GradientXY(nn.Module):
    def __init__(self):
        super(GradientXY, self).__init__()
        kh = torch.tensor([[-1, 1]], requires_grad=False,
                        dtype=torch.float).view((1, 1, 1, 2))
        kv = torch.tensor([[-1],
                           [ 1]], requires_grad=False,
                        dtype=torch.float).view((1, 1, 2, 1))
        self.register_buffer('kh', kh)
        self.register_buffer('kv', kv)

    def forward(self, Y):
        Gx = F.conv2d(Y, self.kh)
        Gy = F.conv2d(Y, self.kv)
        Gx = Gx[:,:,:-1,:] # lose last row as it won't have valid y-grad
        Gy = Gy[:,:,:,:-1] # lose last col as it won't have valid x-grad
        G = torch.cat((Gx, Gy), dim=1)
        return G

class LaplacianOperator(nn.Module):
    def __init__(self):
        super(LaplacianOperator, self).__init__()
        K = torch.tensor([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], requires_grad=False,
                        dtype=torch.float).view((1,1,3,3))
        self.register_buffer('K', K)

    def forward(self, X):
        X = F.conv2d(X, self.K)
        return X
