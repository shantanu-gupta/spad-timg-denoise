""" src.py.kpn
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import os
from src.py.pytorch_ops import GradientXY

class KPN_MildenhallEtAl(nn.Module):
    class _Upsampler(nn.Module):
        def __init__(self, Cin_lr, Cin_skip):
            super(KPN_MildenhallEtAl._Upsampler, self).__init__()
            self.Cin_lr = Cin_lr
            self.Cin_skip = Cin_skip
            self.upsample = nn.Upsample(scale_factor=2,
                                        mode='bilinear',
                                        align_corners=False)

        def forward(self, y_lr, y_skip):
            y_lr = self.upsample(y_lr)
            y = torch.cat((y_lr, y_skip), dim=1)
            return y

    def _conv_block(self, Cin, Cout, add_avg_pool=False, add_final_relu=True):
        parts = []
        if add_avg_pool:
            parts.append(nn.AvgPool2d(2))
        parts.append(nn.Conv2d(Cin, Cout, self.K, padding=self.K//2))
        parts.append(nn.ReLU())
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        parts.append(nn.ReLU())
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        if add_final_relu:
            parts.append(nn.ReLU())
        block = nn.Sequential(*parts)
        return block

    def __init__(self, Kout=5, burst_length=8, stdev=False):
        super(KPN_MildenhallEtAl, self).__init__()
        self.Kout = Kout
        self.N = burst_length
        self.stdev = stdev
        if self.stdev:
            self.N_in = self.N + 1
        else:
            self.N_in = self.N
        self.K = 3
        # self.channels_f = [64, 128, 256, 512, 512]
        # self.channels_b = [512, 256, self.N * (self.Kout ** 2)]

        self.convf_blocks = []
        self.convf_blocks.append(self._conv_block(self.N_in, 64,
                                                add_avg_pool=False))
        self.convf_blocks.append(self._conv_block(64, 128,
                                                add_avg_pool=True))
        self.convf_blocks.append(self._conv_block(128, 256,
                                                add_avg_pool=True))
        self.convf_blocks.append(self._conv_block(256, 512,
                                                add_avg_pool=True))
        self.convf_blocks.append(self._conv_block(512, 512,
                                                add_avg_pool=True))
        self.convf_blocks = nn.ModuleList(self.convf_blocks)

        self.upsamplers = []
        self.convb_blocks = []
        self.upsamplers.append(KPN_MildenhallEtAl._Upsampler(512, 512))
        self.convb_blocks.append(self._conv_block(1024, 512,
                                                add_avg_pool=False))
        self.upsamplers.append(KPN_MildenhallEtAl._Upsampler(512, 256))
        self.convb_blocks.append(self._conv_block(768, 256,
                                                add_avg_pool=False))
        self.upsamplers.append(KPN_MildenhallEtAl._Upsampler(256, 128))
        self.convb_blocks.append(self._conv_block(384, self.N*(self.Kout**2),
                                                add_avg_pool=False,
                                                add_final_relu=False))
        self.upsamplers.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                        align_corners=False))
        self.convb_blocks = nn.ModuleList(self.convb_blocks)
        self.upsamplers = nn.ModuleList(self.upsamplers)
        self.wscale = 6

        if self.Kout != 1:
            self.unfold = nn.Unfold(self.Kout, padding=self.Kout//2)

    def forward(self, y, stdev=None, return_framewise=True,
                return_kernels=True):
        y_in = y
        if self.stdev and stdev is not None:
            y = torch.cat((y, stdev), dim=1)
        ys = []
        y = self.convf_blocks[0](y)
        y = self.convf_blocks[1](y)
        ys.append(y)
        y = self.convf_blocks[2](y)
        ys.append(y)
        y = self.convf_blocks[3](y)
        ys.append(y)
        y = self.convf_blocks[4](y)
        y = self.upsamplers[0](y, ys[2])
        y = self.convb_blocks[0](y)
        y = self.upsamplers[1](y, ys[1])
        y = self.convb_blocks[1](y)
        y = self.upsamplers[2](y, ys[0])
        y = self.convb_blocks[2](y)
        w = self.upsamplers[3](y)
        w = self.wscale * w
        if self.Kout == 1:
            yblocks = y_in
        else:
            yblocks = self.unfold(y_in).view_as(w)
        yw = torch.cat([(w[:,k:k+(self.Kout**2),:,:]
                        * yblocks[:,k:k+(self.Kout**2),:,:]).sum(dim=1,
                                                                keepdim=True)
                        for k in range(0,self.N*(self.Kout**2),self.Kout**2)],
                    dim=1)
        ymerged = yw.mean(dim=1, keepdim=True)
        w = w.view(w.shape[0], self.Kout ** 2, self.N, w.shape[2], w.shape[3])
        ret = (ymerged,)
        if return_framewise:
            ret = ret + (yw,)
        if return_kernels:
            ret = ret + (w,)
        return ret
    
    def save_checkpoint(self, path, epoch):
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        torch.save({'epoch': epoch,
                    'Kout': self.Kout,
                    'stdev': self.stdev,
                    'model_state_dict': self.state_dict(),
                }, path)
        return

    @staticmethod
    @lru_cache(maxsize=1)
    def load_checkpoint(path):
        have_gpu = torch.cuda.is_available()
        device = None if have_gpu else torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)
        Kout = checkpoint['Kout']
        stdev = checkpoint['stdev'] if 'stdev' in checkpoint else False
        model = KPN_MildenhallEtAl(Kout=Kout, stdev=stdev)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return model
    
    @staticmethod
    def framewise_loss_weight(beta, alpha, t):
        return beta * (alpha ** t)

    @staticmethod
    def loss_fn(beta=100, alpha=0.9998, use_l1_vloss=False, fw_weight_thresh=0,
                use_gc=False, use_binomial_anscombe=False,
                ccrop_size=None):
        assert torch.cuda.is_available()
        if use_gc:
            assert not use_binomial_anscombe
        if use_binomial_anscombe:
            assert not use_gc
        grad_module = GradientXY().cuda()
        if use_l1_vloss:
            vloss_fn = nn.L1Loss()
        else:
            vloss_fn = nn.MSELoss()
        gloss_fn = nn.L1Loss()

        def binomial_anscombe_transform(y, num_shots):
            y = torch.clamp(y, 0, 1) * num_shots
            y = (y + (3/8)) * (1.0 / (num_shots + (3/4)))
            y = torch.sqrt(y)
            y = torch.asin(y)
            return y

        def gamma_correct(y):
            yhigh = F.threshold(y, 0.0031308, 0)
            ylow = -F.threshold(-y, -0.0031308, 0)
            a = 0.055
            p = (1/2.4)
            s = 12.92
            return ylow * s + (1 + a) * torch.pow(yhigh, p) - a

        def img_loss(yhat, y, num_shots=None):
            if use_gc:
                yhat, y = gamma_correct(yhat), gamma_correct(y)
            elif use_binomial_anscombe:
                yhat, y = (binomial_anscombe_transform(yhat, num_shots),
                            binomial_anscombe_transform(y, num_shots))
            yhatg = grad_module(yhat)
            yg = grad_module(y)
            if ccrop_size is not None:
                H, W = y.shape[2], y.shape[3]
                assert ccrop_size <= min(H, W)
                gap_H, gap_W = H - ccrop_size, W - ccrop_size
                i0, j0 = gap_H // 2, gap_W // 2
                i1, j1 = i0 + ccrop_size, j0 + ccrop_size
                y = y[:,:,i0:i1,j0:j1]
                yhat = yhat[:,:,i0:i1,j0:j1]
                yg = yg[:,:,i0:i1,j0:j1]
                yhatg = yhatg[:,:,i0:i1,j0:j1]
            vloss = vloss_fn(yhat, y)
            gloss = gloss_fn(yhatg, yg)
            return vloss, gloss

        def fn(y_merged, y_true, t=None, y_fw=None, num_shots=None):
            loss = {}
            merged_vloss, merged_gloss = img_loss(y_merged, y_true,
                                                num_shots=num_shots)
            loss['merged_vloss'] = {'weight': 0.5, 'value': merged_vloss}
            loss['merged_gloss'] = {'weight': 0.5, 'value': merged_gloss}
            if t is not None and y_fw is not None:
                fw_weight = KPN_MildenhallEtAl.framewise_loss_weight(beta, alpha, t)
                if fw_weight > fw_weight_thresh:
                    if num_shots is None:
                        num_shots_fw = None
                    else:
                        num_shots_fw = num_shots / y_fw.shape[1]
                    fw_losses = [img_loss(y_fw[:,k:k+1,:,:], y_true,
                                        num_shots=num_shots_fw)
                                for k in range(y_fw.shape[1])]
                    fw_vloss, fw_gloss = zip(*fw_losses)
                    fw_vloss = torch.stack(fw_vloss).sum()
                    fw_gloss = torch.stack(fw_gloss).sum()
                    loss['fw_vloss'] = {'weight': 0.5 * beta * (alpha ** t),
                                        'value': fw_vloss}
                    loss['fw_gloss'] = {'weight': 0.5 * beta * (alpha ** t),
                                        'value': fw_gloss}
            return loss
        return fn

    @staticmethod
    def merge_burst(burst, ckpt_path, stdev=False, shots_per_frame=None,
                    stdev_multiplier=1.0):
        model = KPN_MildenhallEtAl.load_checkpoint(ckpt_path)
        B = torch.from_numpy(burst).type(torch.float)
        B = B.view(1, B.shape[0], B.shape[1], B.shape[2])
        is_gpu = model.convf_blocks[0][0].weight.is_cuda
        if is_gpu:
            B = B.pin_memory().cuda()
        if stdev:
            assert shots_per_frame is not None
            s0 = (1 / shots_per_frame) * stdev_multiplier
            S = torch.sqrt((B * (1 - B)) * s0)
        else:
            S = None
        with torch.no_grad():
            gray_est, = model(B, stdev=S, return_framewise=False,
                            return_kernels=False)
        gray_est = gray_est.cpu().numpy()[0,0,:,:]
        return gray_est

class KPN_MildenhallEtAl_logtimg(nn.Module):
    """ A wrapper on top of KPN_MildenhallEtAl, specialized for logtimgs.
        TODO: add smoothed_timg input capability
        In that case, each channel means something different, so we can't
        just reuse the previous code like this.
    """
    def __init__(self, Kout=7):
        super(KPN_MildenhallEtAl_logtimg, self).__init__()
        self.kpn = KPN_MildenhallEtAl(Kout=Kout, burst_length=1)
        self.Kout = Kout
        # NOTE: assuming 16-bit ground truth images
        self.tmin_true, self.tmax_true = 1, 65535
        self.bias_correction = 0.5722
        self.threshold = nn.Threshold(np.log(self.tmin_true),
                                    np.log(self.tmin_true))

    def forward(self, t):
        tsm, w = self.kpn(t, return_framewise=False, return_kernels=True)
        tsm = tsm + self.bias_correction
        tsm = self.threshold(tsm)
        w = torch.squeeze(w, dim=2)
        return tsm, w

    def save_checkpoint(self, path, epoch):
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        torch.save({'epoch': epoch,
                    'Kout': self.Kout,
                    'tmin_true': self.tmin_true,
                    'tmax_true': self.tmax_true,
                    'bias_correction': self.bias_correction,
                    'model_state_dict': self.state_dict(),
                }, path)
        return

    @staticmethod
    @lru_cache(maxsize=1)
    def load_checkpoint(path):
        have_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0') if have_gpu else torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)
        Kout = checkpoint['Kout']
        model = KPN_MildenhallEtAl_logtimg(Kout=Kout)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return model

class TinyKPN(nn.Module):
    class _Upsampler(nn.Module):
        def __init__(self, Cin_lr, Cin_skip):
            super(TinyKPN._Upsampler, self).__init__()
            self.Cin_lr = Cin_lr
            self.Cin_skip = Cin_skip
            self.upsample = nn.Upsample(scale_factor=2,
                                        mode='bilinear',
                                        align_corners=False)

        def forward(self, y_lr, y_skip):
            y_lr = self.upsample(y_lr)
            y = torch.cat((y_lr, y_skip), dim=1)
            return y

    def _conv_block(self, Cin, Cout, add_avg_pool=False, add_final_relu=True):
        parts = []
        if add_avg_pool:
            parts.append(nn.AvgPool2d(2))
        parts.append(nn.Conv2d(Cin, Cout, self.K, padding=self.K//2))
        parts.append(nn.ReLU())
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        parts.append(nn.ReLU())
        parts.append(nn.Conv2d(Cout, Cout, self.K, padding=self.K//2))
        if add_final_relu:
            parts.append(nn.ReLU())
        block = nn.Sequential(*parts)
        return block

    def __init__(self, Kout=3, burst_length=8, stdev=False):
        super(TinyKPN, self).__init__()
        self.Kout = Kout
        self.N = burst_length
        self.stdev = stdev
        if self.stdev:
            self.N_in = self.N + 1
        else:
            self.N_in = self.N
        self.K = 3
        # self.channels_f = [64, 128, 256]
        # self.channels_b = [256, self.N * (self.Kout ** 2)]

        self.convf_blocks = []
        self.convf_blocks.append(self._conv_block(self.N_in, 64,
                                                add_avg_pool=False))
        self.convf_blocks.append(self._conv_block(64, 128,
                                                add_avg_pool=True))
        self.convf_blocks.append(self._conv_block(128, 128,
                                                add_avg_pool=True))
        self.convf_blocks = nn.ModuleList(self.convf_blocks)

        self.upsamplers = []
        self.convb_blocks = []
        self.upsamplers.append(KPN_MildenhallEtAl._Upsampler(128, 128))
        self.convb_blocks.append(self._conv_block(256, self.N*(self.Kout**2),
                                                add_avg_pool=False,
                                                add_final_relu=False))
        self.upsamplers.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                        align_corners=False))
        self.convb_blocks = nn.ModuleList(self.convb_blocks)
        self.upsamplers = nn.ModuleList(self.upsamplers)
        self.wscale = 6

        if self.Kout != 1:
            self.unfold = nn.Unfold(self.Kout, padding=self.Kout//2)

    def forward(self, y, stdev=None, return_framewise=True,
                return_kernels=True):
        y_in = y
        if self.stdev and stdev is not None:
            y = torch.cat((y, stdev), dim=1)
        ys = []
        y = self.convf_blocks[0](y)
        y = self.convf_blocks[1](y)
        ys.append(y)
        y = self.convf_blocks[2](y)
        y = self.upsamplers[0](y, ys[0])
        y = self.convb_blocks[0](y)
        w = self.upsamplers[1](y)
        w = self.wscale * w
        if self.Kout == 1:
            yblocks = y_in
        else:
            yblocks = self.unfold(y_in).view_as(w)
        yw = torch.cat([(w[:,k:k+(self.Kout**2),:,:]
                        * yblocks[:,k:k+(self.Kout**2),:,:]).sum(dim=1,
                                                                keepdim=True)
                        for k in range(0,self.N*(self.Kout**2),self.Kout**2)],
                    dim=1)
        ymerged = yw.mean(dim=1, keepdim=True)
        w = w.view(w.shape[0], self.Kout ** 2, self.N, w.shape[2], w.shape[3])
        ret = (ymerged,)
        if return_framewise:
            ret = ret + (yw,)
        if return_kernels:
            ret = ret + (w,)
        return ret
    
    def save_checkpoint(self, path, epoch):
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        torch.save({'epoch': epoch,
                    'Kout': self.Kout,
                    'stdev': self.stdev,
                    'model_state_dict': self.state_dict(),
                }, path)
        return

    @staticmethod
    @lru_cache(maxsize=1)
    def load_checkpoint(path):
        have_gpu = torch.cuda.is_available()
        device = None if have_gpu else torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)
        Kout = checkpoint['Kout']
        stdev = checkpoint['stdev'] if 'stdev' in checkpoint else False
        model = TinyKPN(Kout=Kout, stdev=stdev)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return model

    @staticmethod
    def merge_burst(burst, ckpt_path, stdev=False, shots_per_frame=None,
                    stdev_multiplier=1.0):
        model = TinyKPN.load_checkpoint(ckpt_path)
        B = torch.from_numpy(burst).type(torch.float)
        B = B.view(1, B.shape[0], B.shape[1], B.shape[2])
        is_gpu = model.convf_blocks[0][0].weight.is_cuda
        if is_gpu:
            B = B.pin_memory().cuda()
        if stdev:
            assert shots_per_frame is not None
            s0 = (1 / shots_per_frame) * stdev_multiplier
            S = torch.sqrt((B * (1 - B)) * s0)
        else:
            S = None
        with torch.no_grad():
            gray_est, = model(B, stdev=S, return_framewise=False,
                            return_kernels=False)
        gray_est = gray_est.cpu().numpy()[0,0,:,:]
        return gray_est

