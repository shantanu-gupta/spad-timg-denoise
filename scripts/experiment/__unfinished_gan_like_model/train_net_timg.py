""" train_net_timg.py
"""

import argparse
import os
import datetime
import json
import shutil

from itertools import chain

from utils import io as io_utils

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_stuff.datasets.timg import TimgDataset_2K
# from pytorch_stuff.models.timg_denoise import Timg_DenoiseNet
# from pytorch_stuff.models.timg_denoise import Timg_DenoiseNet_LinT
from pytorch_stuff.models.timg_denoise import Timg_DenoiseNet_LinT_1Layer
from pytorch_stuff.models.timg_kpn import Timg_KPN_1Layer
from pytorch_stuff.models.timg_kpn import Timg_KPN_2Layer
from pytorch_stuff.models.timg_discrim import Timg_Discrim
from pytorch_stuff.misc import GradientXY

def save_checkpoint(path, model, discrim, epoch):
    base_dir = os.path.dirname(path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discrim_state_dict': discrim.state_dict(),
            }, path)
    return

def gen_lossfn(crit_r, disc_module, disc_weight):
    crit_d = nn.BCEWithLogitsLoss()
    # avoid over-fitting to current state of discriminator
    clipper = nn.Hardtanh(min_val=0, max_val=0.693)
    def fn(that, t):
        rloss = crit_r(that, t)
        disc_output = disc_module(that)
        N = t.shape[0]
        disc_loss = crit_d(disc_output, torch.zeros(N,1).cuda())
        disc_loss = clipper(disc_loss)
        return (1 - disc_weight) * rloss + disc_weight * (0.693 - disc_loss)
    return fn

def disc_lossfn(disc_module):
    crit = nn.BCEWithLogitsLoss()
    def fn(that, t):
        N = t.shape[0]
        din = torch.cat((that, t), dim=0)
        dout = disc_module(din)
        labels = torch.cat((torch.zeros(N,1), torch.ones(N,1)), dim=0).cuda()
        return crit(dout, labels)
    return fn

def ntimg2img(ntimg):
    return 1.0 / (255.0 * torch.clamp(ntimg, 1/255.0, 1.0))

def rescale_pimg(pimg):
    return (1.0 / pimg.max()) * pimg

def reconstruction_lossfn(value_lossfn, grad_lossfn=None):
    if grad_lossfn is not None:
        grad_module = GradientXY().cuda()
    def fn(that, t):
        vloss = value_lossfn(that, t)
        if grad_lossfn is not None:
            thatg = grad_module(that)
            tg = grad_module(t)
            gloss = grad_lossfn(thatg, tg)
            return 0.5 * (vloss + gloss)
        else:
            return vloss
    return fn

def train_monly(model, optimizer, loss_fn, train_loader, epoch, tb_writer=None):
    model.train()
    num_batches = len(train_loader)
    for i_batch, batch in enumerate(train_loader):
        timgs = batch['timg']
        timgs_true = batch['gt']
        N, C, H, W = timgs.shape
        timgs = timgs.view(-1, 1, H, W).cuda()
        timgs_true = timgs_true.view(-1, 1, H, W).cuda()

        logmore = (20*i_batch) % num_batches == 0
        if logmore:
            timgs.requires_grad = True
            timgs_true.requires_grad = True
            if timgs.grad is not None:
                timgs.grad.zero_()
            if timgs_true.grad is not None:
                timgs_true.grad.zero_()

        timgs_est = model(timgs)
        loss = loss_fn(timgs_est, timgs_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb_writer is not None:
            tb_writer.add_scalar('time/monly_epoch', epoch, epoch)
            tb_writer.add_scalar('train_loss/monly_loss', loss,
                                (epoch-1)*num_batches + i_batch + 1)
            if logmore:
                timg_comparison = rescale_pimg(
                                        torch.cat((timgs[0,0,:,:],
                                                timgs_est[0,0,:,:],
                                                timgs_true[0,0,:,:]), dim=1))
                tb_writer.add_image('train_compare_timg_monly',
                                    timg_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')

                with torch.no_grad():
                    gray_init = ntimg2img(timgs[0,0,:,:])
                    gray_est = ntimg2img(timgs_est[0,0,:,:])
                    gray_true = ntimg2img(timgs_true[0,0,:,:])
                gray_comparison = torch.cat((gray_init, gray_est, gray_true),
                                            dim=1)
                tb_writer.add_image('train_compare_gray_monly',
                                    gray_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')

                attention_input = torch.abs(timgs.grad[0,0,:,:])
                attention_true = torch.abs(timgs_true.grad[0,0,:,:])
                attention_comparison = rescale_pimg(
                                            torch.cat((attention_input,
                                                        attention_true),
                                                    dim=1))
                tb_writer.add_image('train_attention_monly',
                                    attention_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')
    return

def train_donly(model, discrim, optimizer, loss_fn, train_loader, epoch,
                tb_writer=None):
    discrim.train()
    model.eval()
    num_batches = len(train_loader)
    for i_batch, batch in enumerate(train_loader):
        timgs = batch['timg']
        timgs_true = batch['gt']
        N, C, H, W = timgs.shape
        timgs = timgs.view(-1, 1, H, W).cuda()
        timgs_true = timgs_true.view(-1, 1, H, W).cuda()

        # don't need to worry about the generator end here
        with torch.no_grad():
            timgs_est = model(timgs)

        logmore = (20*i_batch) % num_batches == 0
        if logmore:
            timgs_est.requires_grad = True
            timgs_true.requires_grad = True
            if timgs_est.grad is not None:
                timgs_est.grad.zero_()
            if timgs_true.grad is not None:
                timgs_true.grad.zero_()
        loss = loss_fn(timgs_est, timgs_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb_writer is not None:
            tb_writer.add_scalar('time/donly_epoch', epoch, epoch)
            tb_writer.add_scalar('train_loss/donly_loss', loss,
                                (epoch-1)*num_batches + i_batch + 1)
            if logmore:
                timg_comparison = rescale_pimg(
                                        torch.cat((timgs[0,0,:,:],
                                                timgs_est[0,0,:,:],
                                                timgs_true[0,0,:,:]), dim=1))
                tb_writer.add_image('train_compare_timg_donly',
                                    timg_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')

                with torch.no_grad():
                    gray_init = ntimg2img(timgs[0,0,:,:])
                    gray_est = ntimg2img(timgs_est[0,0,:,:])
                    gray_true = ntimg2img(timgs_true[0,0,:,:])
                gray_comparison = torch.cat((gray_init, gray_est, gray_true),
                                            dim=1)
                tb_writer.add_image('train_compare_gray_donly',
                                    gray_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')
                attention_est = torch.abs(timgs_est.grad[0,0,:,:])
                attention_true = torch.abs(timgs_true.grad[0,0,:,:])
                attention_comparison = rescale_pimg(
                                            torch.cat((attention_est,
                                                        attention_true),
                                                    dim=1))
                tb_writer.add_image('train_attention_donly',
                                    attention_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')
    return

def train_combined(model, m_opt, m_lrs, m_lf, discrim, d_opt, d_lrs, d_lf,
                train_loader, epoch, tb_writer=None):
    num_batches = len(train_loader)
    for i_batch, batch in enumerate(train_loader):
        timgs = batch['timg']
        timgs_true = batch['gt']
        N, C, H, W = timgs.shape
        timgs = timgs.view(-1, 1, H, W).cuda()
        timgs_true = timgs_true.view(-1, 1, H, W).cuda()

        logmore = (20*i_batch) % num_batches == 0
        if logmore:
            timgs.requires_grad = True
            timgs_true.requires_grad = True
            if timgs.grad is not None:
                timgs.grad.zero_()
            if timgs_true.grad is not None:
                timgs_true.grad.zero_()

        model.train()
        discrim.eval()
        timgs_est = model(timgs)
        m_loss = m_lf(timgs_est, timgs_true)
        m_opt.zero_grad()
        m_loss.backward()
        m_opt.step()
        m_lrs.step()

        if logmore:
            attention_true_m = torch.abs(timgs_true.grad[0,0,:,:])
            timgs_true.detach_()
            timgs_true.requires_grad = True
            if timgs_true.grad is not None:
                timgs_true.grad.zero_()

        discrim.train()
        model.eval()
        timgs_est.detach_()
        if logmore:
            timgs_est.requires_grad = True
            if timgs_est.grad is not None:
                timgs_est.grad.zero_()
        d_loss = d_lf(timgs_est, timgs_true)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()
        d_lrs.step()

        if tb_writer is not None:
            tb_writer.add_scalar('time/comb_epoch', epoch, epoch)
            tb_writer.add_scalar('time/comb_model_LR', m_lrs.get_lr()[0],
                                (epoch-1)*num_batches + i_batch + 1)
            tb_writer.add_scalar('train_loss/comb_model_loss', m_loss,
                                (epoch-1)*num_batches + i_batch + 1)
            tb_writer.add_scalar('time/comb_discrim_LR', d_lrs.get_lr()[0],
                                (epoch-1)*num_batches + i_batch + 1)
            tb_writer.add_scalar('train_loss/comb_discrim_loss', d_loss,
                                (epoch-1)*num_batches + i_batch + 1)

            if logmore:
                timg_comparison = rescale_pimg(
                                        torch.cat((timgs[0,0,:,:],
                                                timgs_est[0,0,:,:],
                                                timgs_true[0,0,:,:]), dim=1))
                tb_writer.add_image('train_compare_timg_comb',
                                    timg_comparison,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')

                with torch.no_grad():
                    gray_init = ntimg2img(timgs[0,0,:,:])
                    gray_est = ntimg2img(timgs_est[0,0,:,:])
                    gray_true = ntimg2img(timgs_true[0,0,:,:])
                    gray_comparison = torch.cat((gray_init, gray_est, gray_true),
                                                dim=1)
                    tb_writer.add_image('train_compare_gray_comb',
                                        gray_comparison,
                                        (epoch-1)*num_batches + i_batch + 1,
                                        dataformats='HW')
                attention_input = torch.abs(timgs.grad[0,0,:,:]) # from m_loss
                # attention_true_m is stored already after m_lrs.step()
                attention_comparison_m = rescale_pimg(
                                            torch.cat((attention_input,
                                                        attention_true_m),
                                                    dim=1))
                tb_writer.add_image('train_attention_m_comb',
                                    attention_comparison_m,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')
                attention_est = torch.abs(timgs_est.grad[0,0,:,:])
                attention_true_d = torch.abs(timgs_true.grad[0,0,:,:])
                attention_comparison_d = rescale_pimg(
                                            torch.cat((attention_est,
                                                        attention_true_d),
                                                    dim=1))
                tb_writer.add_image('train_attention_d_comb',
                                    attention_comparison_d,
                                    (epoch-1)*num_batches + i_batch + 1,
                                    dataformats='HW')
    return

def validate_monly(model, loss_fn, val_loader, epoch, output_dir=None,
                tb_writer=None):
    model.eval()
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_batches = len(val_loader)
    num_processed = 0
    for i_batch, batch in enumerate(val_loader):
        timgs = batch['timg'].cuda()
        timgs.requires_grad = True
        if timgs.grad is not None:
            timgs.grad.zero_()
        timgs_true = batch['gt'].cuda()
        timgs_true.requires_grad = True
        if timgs_true.grad is not None:
            timgs_true.grad.zero_()

        timgs_est = model(timgs)
        loss = loss_fn(timgs_est, timgs_true)
        if tb_writer is not None:
            tb_writer.add_scalar('validation_loss/monly_loss', loss,
                                (epoch-1)*num_batches + i_batch + 1)
        # Save output
        if output_dir is not None:
            with torch.no_grad():
                gray_init = ntimg2img(timgs)
                gray_est = ntimg2img(timgs_est)
                gray_true = ntimg2img(timgs_true)

            loss.backward()
            attention_input = torch.abs(timgs.grad)
            attention_true = torch.abs(timgs_true.grad)

            timgs_init_np = timgs.detach().cpu().numpy()
            timgs_est_np = timgs_est.detach().cpu().numpy()
            timgs_true_np = timgs_true.detach().cpu().numpy()
            gray_init_np = gray_init.cpu().numpy()
            gray_est_np = gray_est.cpu().numpy()
            gray_true_np = gray_true.cpu().numpy()
            att_input_np = attention_input.cpu().numpy()
            att_true_np = attention_true.cpu().numpy()
            for k in range(gray_est_np.shape[0]):
                timg_init_k = timgs_init_np[k,0,:,:]
                timg_est_k = timgs_est_np[k,0,:,:]
                timg_true_k = timgs_true_np[k,0,:,:]
                output_img = rescale_pimg(
                                np.hstack((np.clip(timg_init_k, None, 1),
                                            timg_est_k,
                                            timg_true_k)))
                gray_init_k = gray_init_np[k,0,:,:]
                gray_est_k = gray_est_np[k,0,:,:]
                gray_true_k = gray_true_np[k,0,:,:]
                output_img = np.vstack((output_img,
                                        np.hstack((gray_init_k,
                                                gray_est_k,
                                                gray_true_k))))
                save_path = os.path.join(output_dir,
                        'est_{:04d}.png'.format(num_processed + k))
                io_utils.save_img(io_utils.array_to_img(output_img, 'L'),
                                save_path)
                att_input_k = att_input_np[k,0,:,:]
                att_true_k = att_true_np[k,0,:,:]
                output_img = rescale_pimg(np.hstack((att_input_k, att_true_k)))
                save_path = os.path.join(output_dir,
                        'att_{:04d}.png'.format(num_processed + k))
                io_utils.save_img(io_utils.array_to_img(output_img, 'L'),
                                save_path)
        num_processed += timgs_est.shape[0]
    return

def validate_combined(model, model_loss_fn, discrim, discrim_loss_fn,
                    val_loader, epoch, output_dir=None, tb_writer=None):
    model.eval()
    discrim.eval()
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_batches = len(val_loader)
    num_processed = 0
    for i_batch, batch in enumerate(val_loader):
        timgs = batch['timg'].cuda()
        timgs.requires_grad = True
        if timgs.grad is not None:
            timgs.grad.zero_()
        timgs_true = batch['gt'].cuda()
        timgs_true.requires_grad = True
        if timgs_true.grad is not None:
            timgs_true.grad.zero_()

        timgs_est = model(timgs)
        m_loss = model_loss_fn(timgs_est, timgs_true)

        m_loss.backward()
        attention_true_m = torch.abs(timgs_true.grad)
        timgs_true.detach_()
        timgs_true.requires_grad = True
        if timgs_true.grad is not None:
            timgs_true.grad.zero_()
        timgs_est.detach_()
        timgs_est.requires_grad = True
        if timgs_est.grad is not None:
            timgs_est.grad.zero_()

        d_loss = discrim_loss_fn(timgs_est, timgs_true)
        d_loss.backward()

        if tb_writer is not None:
            tb_writer.add_scalar('validation_loss/comb_model_loss', m_loss,
                                (epoch-1)*num_batches + i_batch + 1)
            tb_writer.add_scalar('validation_loss/comb_discrim_loss',d_loss,
                                (epoch-1)*num_batches + i_batch + 1)
        # Save output
        if output_dir is not None:
            with torch.no_grad():
                gray_init = ntimg2img(timgs)
                gray_est = ntimg2img(timgs_est)
                gray_true = ntimg2img(timgs_true)

            attention_input = torch.abs(timgs.grad)
            attention_est = torch.abs(timgs_est.grad)
            attention_true_d = torch.abs(timgs_true.grad)
            timgs_init_np = timgs.detach().cpu().numpy()
            timgs_est_np = timgs_est.detach().cpu().numpy()
            timgs_true_np = timgs_true.detach().cpu().numpy()
            gray_init_np = gray_init.cpu().numpy()
            gray_est_np = gray_est.cpu().numpy()
            gray_true_np = gray_true.cpu().numpy()
            att_input_np = attention_input.cpu().numpy()
            att_true_m_np = attention_true_m.cpu().numpy()
            att_est_np = attention_est.cpu().numpy()
            att_true_d_np = attention_true_m.cpu().numpy()
            for k in range(gray_est_np.shape[0]):
                timg_init_k = timgs_init_np[k,0,:,:]
                timg_est_k = timgs_est_np[k,0,:,:]
                timg_true_k = timgs_true_np[k,0,:,:]
                output_img = rescale_pimg(
                                np.hstack((np.clip(timg_init_k, None, 1),
                                            timg_est_k,
                                            timg_true_k)))
                gray_init_k = gray_init_np[k,0,:,:]
                gray_est_k = gray_est_np[k,0,:,:]
                gray_true_k = gray_true_np[k,0,:,:]
                output_img = np.vstack((output_img,
                                        np.hstack((gray_init_k,
                                                gray_est_k,
                                                gray_true_k))))
                save_path = os.path.join(output_dir,
                        'est_{:04d}.png'.format(num_processed + k))
                io_utils.save_img(io_utils.array_to_img(output_img, 'L'),
                                save_path)
                att_input_k = att_input_np[k,0,:,:]
                att_true_m_k = att_true_m_np[k,0,:,:]
                output_img = rescale_pimg(np.hstack((att_input_k,
                                                    att_true_m_k)))
                att_est_k = att_est_np[k,0,:,:]
                att_true_d_k = att_true_d_np[k,0,:,:]
                output_img = np.vstack((output_img,
                                        rescale_pimg(
                                            np.hstack((att_input_k,
                                                    att_true_d_k)))))
                save_path = os.path.join(output_dir,
                        'att_{:04d}.png'.format(num_processed + k))
                io_utils.save_img(io_utils.array_to_img(output_img, 'L'),
                                save_path)
        num_processed += timgs_est.shape[0]
    return

def main(args):
    tb_writer = None
    if args.log_dir is not None:
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        tb_writer = SummaryWriter(args.log_dir)
    assert torch.cuda.is_available()
    # torch.backends.cudnn.enabled = False
    
    train_metadata = json.load(args.train_metadata)['image_metadata']
    val_metadata = json.load(args.val_metadata)['image_metadata']
    train_dataset = TimgDataset_2K(train_metadata,
                                num_timgs_per_img=args.num_timgs_per_img,
                                crops_per_img=args.crops_per_img,
                                crop_size=args.crop_size)
    val_dataset = TimgDataset_2K(val_metadata, crop_size=args.crop_size)
    train_dloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            shuffle=True, pin_memory=True,
                            num_workers=args.dloader_workers,
                            worker_init_fn=lambda _: np.random.seed())
    val_dloader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                            shuffle=True, pin_memory=True,
                            num_workers=args.dloader_workers,
                            worker_init_fn=lambda _: np.random.seed())

    print('{} batches in train loader.'.format(len(train_dloader)))
    print('{} batches in val loader.'.format(len(val_dloader)))
    
    num_batches = len(train_dloader)
    # model = Timg_DenoiseNet().cuda()
    # model = Timg_DenoiseNet_LinT_1Layer().cuda()
    # model = Timg_KPN_1Layer().cuda()
    model = Timg_KPN_2Layer().cuda()
    discrim = Timg_Discrim().cuda()
    reconst_loss_fn = reconstruction_lossfn(nn.L1Loss(),
                                            grad_lossfn=nn.L1Loss())
    # reconst_loss_fn = reconstruction_lossfn(nn.MSELoss(),
    #                                         grad_lossfn=nn.L1Loss())
    # reconst_loss_fn = reconstruction_lossfn(nn.MSELoss())
    # reconst_loss_fn = reconstruction_lossfn(nn.L1Loss())
    # reconst_loss_fn = reconstruction_lossfn(nn.SmoothL1Loss())
    # reconst_loss_fn = gc_int_lossfn(nn.MSELoss(), nn.L1Loss())

    discrim_loss_fn = disc_lossfn(discrim)
    if args.wd > 0:
        model_loss_fn = gen_lossfn(reconst_loss_fn, discrim, args.wd)
    else:
        model_loss_fn = reconst_loss_fn

    model_optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=0.9)
    discrim_optimizer = torch.optim.SGD(discrim.parameters(),
                                    lr=args.lr, momentum=0.9)
    model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                    model_optimizer,
                                    step_size=num_batches*5,
                                    gamma=0.9)
    discrim_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                    discrim_optimizer,
                                    step_size=num_batches*5,
                                    gamma=0.9)

    val_dir = os.path.join(args.val_output_dir, 'initial')
    validate_combined(model, reconst_loss_fn, discrim, discrim_loss_fn,
                    val_dloader, 0, output_dir=val_dir, tb_writer=tb_writer)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'state_initial.tar')
    save_checkpoint(checkpoint_path, model, discrim, 0)

    if args.epochs_monly > 0:
        for epoch in range(1, 1+args.epochs_monly):
            train_monly(model, model_optimizer,
                        reconst_loss_fn, train_dloader, epoch,
                        tb_writer=tb_writer)
            val_dir = os.path.join(args.val_output_dir,
                                    'monly_epoch{:04d}'.format(epoch))
            validate_monly(model, reconst_loss_fn, val_dloader, epoch,
                        output_dir=val_dir, tb_writer=tb_writer)
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                'state_monly_epoch{:04d}.tar'.format(epoch))
            save_checkpoint(checkpoint_path, model, discrim, epoch)

    if args.epochs_donly > 0:
        for epoch in range(1, 1+args.epochs_donly):
            train_donly(model, discrim, discrim_optimizer,
                        discrim_loss_fn, train_dloader, epoch,
                        tb_writer=tb_writer)
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                'state_donly_epoch{:04d}.tar'.format(epoch))
            save_checkpoint(checkpoint_path, model, discrim, epoch)

    if args.epochs_combined > 0:
        for epoch in range(1, 1+args.epochs_combined):
            train_combined(model,
                    model_optimizer, model_lr_scheduler, model_loss_fn,
                    discrim,
                    discrim_optimizer, discrim_lr_scheduler, discrim_loss_fn,
                    train_dloader, epoch, tb_writer=tb_writer)
            val_dir = os.path.join(args.val_output_dir,
                                'combined_epoch{:04d}'.format(epoch))
            validate_combined(model, model_loss_fn, discrim, discrim_loss_fn,
                            val_dloader, epoch, output_dir=val_dir,
                            tb_writer=tb_writer)
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                'state_combined_epoch{:04d}.tar'.format(epoch))
            save_checkpoint(checkpoint_path, model, discrim, epoch)
    tb_writer.close()

    # Save final checkpoint regardless of if it has been done already.
    checkpoint_path = os.path.join(args.checkpoint_dir, 'state_final.tar')
    save_checkpoint(checkpoint_path, model, discrim, -1)
    return

if __name__ == '__main__':
    help_str = 'Train a CNN to estimate a grayscale image from a timg'
    parser = argparse.ArgumentParser(description=help_str)

    help_str = 'Training images metadata'
    parser.add_argument('train_metadata', type=argparse.FileType('r'),
                        help=help_str)

    help_str = 'Validation images metadata'
    parser.add_argument('val_metadata', type=argparse.FileType('r'),
                        help=help_str)

    help_str = 'Number of simulated timgs present for every real image'
    parser.add_argument('--num-timgs-per-img', type=int, help=help_str,
                        default=1)

    help_str = 'Number of crops to extract per training image'
    parser.add_argument('--crops-per-img', type=int, help=help_str,
                        default=1)

    help_str = 'Image crop size'
    parser.add_argument('--crop-size', type=int, help=help_str)

    help_str = 'Where to store any training logs'
    parser.add_argument('--log-dir', type=str, help=help_str)

    help_str = 'Number of workers in the DataLoader'
    parser.add_argument('--dloader-workers', type=int, help=help_str,
                        default=8)

    help_str = 'Number of epochs to train the model individually'
    parser.add_argument('--epochs-monly', type=int, help=help_str,
                        default=0)

    help_str = 'Number of epochs to train the discriminator only'
    parser.add_argument('--epochs-donly', type=int, help=help_str,
                        default=0)
    
    help_str = 'Number of epochs to train combined'
    parser.add_argument('--epochs-combined', type=int, help=help_str,
                        default=10)

    help_str = 'Weight of discriminator loss in model loss function'
    parser.add_argument('--wd', type=float, help=help_str,
                        default=0.0)

    help_str = 'Number of training images per batch'
    parser.add_argument('--train-batch-size', type=int, help=help_str,
                        default=4)

    help_str = 'Initial learning rate'
    parser.add_argument('--lr', type=float, help=help_str,
                        default=1.0e-2)

    help_str = 'Where to save checkpoints'
    parser.add_argument('--checkpoint-dir', type=str, help=help_str,
                        default='./cnn-checkpoints')

    help_str = 'Number of images in a validation batch'
    parser.add_argument('--val-batch-size', type=int, help=help_str,
                        default=8)

    help_str = 'Where to store the validation output'
    parser.add_argument('--val-output-dir', type=str, help=help_str,
                        default='./cnn-val-output')

    args = parser.parse_args()
    main(args)
