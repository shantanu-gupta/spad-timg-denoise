""" scripts.base.kpn_timg.train_logtimg
"""
import argparse
import os
import sys
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.py.pytorch_data import TimgDataset
from src.py.kpn import KPN_MildenhallEtAl_logtimg, KPN_MildenhallEtAl
from src.py.pytorch_ops import GradientXY
import src.py.io as io_utils
from src.py.spad import invert_spad_logtimg

def _create_parser():
    help_str = 'Train a CNN to estimate a grayscale image from a timg'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Training images metadata'
    parser.add_argument('train_metadata', type=argparse.FileType('r'),
                        help=help_str)
    help_str = 'Validation images metadata'
    parser.add_argument('val_metadata', type=argparse.FileType('r'),
                        help=help_str)
    help_str = 'A location where we can load the model from (optional)'
    parser.add_argument('--load-checkpoint', type=str, help=help_str)
    help_str = 'Size of predicted kernels'
    parser.add_argument('--Kout', type=int, help=help_str,
                        default=7)
    help_str = 'Image crop size'
    parser.add_argument('--crop-size', type=int, help=help_str)
    help_str = 'Number of workers in the DataLoader'
    parser.add_argument('--dloader-workers', type=int, help=help_str,
                        default=4)
    help_str = 'Number of epochs to train the model'
    parser.add_argument('--epochs', type=int, help=help_str,
                        default=0)
    help_str = 'Whether to use L1 loss function for intensity'
    parser.add_argument('--use-l1-vloss', action='store_true', help=help_str)
    help_str = 'Training batch size'
    parser.add_argument('--batch-size', type=int, help=help_str,
                        default=1)
    help_str = 'Initial learning rate'
    parser.add_argument('--lr', type=float, help=help_str,
                        default=1.0e-2)
    help_str = 'Where to store any training logs'
    parser.add_argument('--log-dir', type=str, help=help_str,
                        required=True)
    help_str = 'Where to save checkpoints'
    parser.add_argument('--checkpoint-dir', type=str, help=help_str,
                        default='./generated/models/ckpt')
    help_str = 'Where to store the validation output'
    parser.add_argument('--val-output-dir', type=str, help=help_str,
                        default='./generated/models/val')
    return parser

def _rescale_img(img):
    with torch.no_grad():
        m, M = img.min(), img.max()
        return (img - m) / (M - m)

def train(model, optimizer, loss_fn, train_loader, epoch, tb_writer=None):
    model.train()
    num_batches = len(train_loader)
    for i_batch, batch in enumerate(train_loader):
        logtimgs = batch['data'].cuda()
        gt = batch['gt'].cuda()

        logmore = (20*i_batch) % num_batches == 0
        if logmore:
            logtimgs.requires_grad = True
            gt.requires_grad = True
            if logtimgs.grad is not None:
                logtimgs.grad.zero_()
            if gt.grad is not None:
                gt.grad.zero_()

        logtimgs_est, w_est = model(logtimgs)
        loss = None
        step = (epoch-1)*num_batches + i_batch + 1
        losses = loss_fn(logtimgs_est, gt)
        for loss_name, wnv in losses.items():
            if loss is None:
                loss = wnv['weight'] * wnv['value']
            else:
                loss = loss + wnv['weight'] * wnv['value']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb_writer is not None:
            tb_writer.add_scalar('time/epoch', epoch, epoch)
            with torch.no_grad():
                gt_means = torch.mean(gt, (1, 2, 3), keepdim=True)
                gt_dev = gt - gt_means
                gt_var = torch.pow(gt_dev, 2).mean(dim=(1, 2, 3))
            tb_writer.add_scalar('train_loss/gt_var', gt_var.min(),
                                (epoch-1)*num_batches + i_batch + 1)
            tb_writer.add_scalar('train_loss/mean_wsum',
                                w_est.sum(dim=1).mean(),
                                step)
            for loss_name, wnv in losses.items():
                lv = wnv['value']
                tb_writer.add_scalar('train_loss/{}'.format(loss_name),lv,step)
            if logmore:
                with torch.no_grad():
                    comparison = torch.cat((logtimgs[0,0,:,:],
                                            logtimgs_est[0,0,:,:],
                                            gt[0,0,:,:]),
                                        dim=1)
                    comparison = torch.clamp(comparison, 0, 1)
                    tb_writer.add_image('train_compare',
                                        comparison,
                                        step,
                                        dataformats='HW')
                logtimgs.detach_()
                logtimgs_est.detach_()
                gt.detach_()
                attention_input = torch.max(torch.abs(logtimgs.grad[0,:,:,:]),
                                            dim=0, keepdim=False)[0]
                attention_true = torch.abs(gt.grad[0,0,:,:])
                attention_comparison = _rescale_img(
                                            torch.cat((attention_input,
                                                        attention_true),
                                                    dim=1))
                tb_writer.add_image('train_attention',
                                    attention_comparison,
                                    step,
                                    dataformats='HW')
    return

def validate(model, loss_fn, val_loader, epoch, output_dir=None,
            tb_writer=None):
    model.eval()
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_batches = len(val_loader)
    num_processed = 0
    for i_batch, batch in enumerate(val_loader):
        logtimgs = batch['data'].cuda()
        logtimgs.requires_grad = True
        if logtimgs.grad is not None:
            logtimgs.grad.zero_()
        gt = batch['gt'].cuda()
        gt.requires_grad = True
        if gt.grad is not None:
            gt.grad.zero_()

        logtimgs_est, w_est = model(logtimgs)
        loss = None
        step = (epoch-1) * num_batches + i_batch + 1
        losses = loss_fn(logtimgs_est, gt)
        for loss_name, wnv in losses.items():
            if loss is None:
                loss = wnv['weight'] * wnv['value']
            else:
                loss = loss + wnv['weight'] * wnv['value']
        if tb_writer is not None:
            tb_writer.add_scalar('validation_loss/mean_wsum',
                                w_est.sum(dim=1).mean(), step)
            with torch.no_grad():
                gt_means = torch.mean(gt, (1, 2, 3), keepdim=True)
                gt_dev = gt - gt_means
                gt_var = torch.pow(gt_dev, 2).mean(dim=(1, 2, 3))
            tb_writer.add_scalar('validation_loss/gt_var', gt_var.min(),
                                (epoch-1)*num_batches + i_batch + 1)
            for loss_name, wnv in losses.items():
                lv = wnv['value']
                tb_writer.add_scalar('validation_loss/{}'.format(loss_name),
                                    lv, step)
        # Save output
        if output_dir is not None:
            loss.backward()
            attention_input = torch.max(torch.abs(logtimgs.grad), dim=1,
                                        keepdim=True)[0]
            attention_true = torch.abs(gt.grad)

            logtimgs.detach_()
            logtimgs_est.detach_()
            gt.detach_()
            logtimgs_init_np = logtimgs.cpu().numpy()
            logtimgs_est_np = logtimgs_est.cpu().numpy()
            gt_np = gt.cpu().numpy()
            att_input_np = attention_input.cpu().numpy()
            att_true_np = attention_true.cpu().numpy()
            for k in range(logtimgs_est_np.shape[0]):
                logtimgs_k = logtimgs_init_np[k,0,:,:]
                logtimgs_est_k = logtimgs_est_np[k,0,:,:]
                gt_k = gt_np[k,0,:,:]
                output_img = np.hstack((logtimgs_k,
                                        logtimgs_est_k,
                                        gt_k))
                # For 8-bit data
                gray_k = invert_spad_logtimg(logtimgs_k, tmin=1, tmax=255)
                gray_est_k = invert_spad_logtimg(logtimgs_est_k, tmin=1,
                                                tmax=255)
                gray_gt_k = invert_spad_logtimg(gt_k, tmin=1, tmax=255)
                output_img = np.vstack((output_img,
                                        np.hstack((gray_k,
                                                gray_est_k,
                                                gray_gt_k))))
                save_path = os.path.join(output_dir,
                        'est_{:04d}.tiff'.format(num_processed + k))
                io_utils.save_float_img(save_path, output_img)
                att_input_k = att_input_np[k,0,:,:]
                att_true_k = att_true_np[k,0,:,:]
                output_img = _rescale_img(np.hstack((att_input_k, att_true_k)))
                save_path = os.path.join(output_dir,
                        'att_{:04d}.tiff'.format(num_processed + k))
                io_utils.save_float_img(save_path, output_img)
        num_processed += logtimgs_est.shape[0]
    return

def as32bit(n):
    return n & 0xffffffff

def main(args):
    tb_writer = None
    if args.log_dir is not None:
        if os.path.exists(args.log_dir):
            print('log dir exists at {}. Removing it.'.format(args.log_dir))
            shutil.rmtree(args.log_dir)
        tb_writer = SummaryWriter(args.log_dir)
    assert torch.cuda.is_available()
    
    train_metadata = json.load(args.train_metadata)
    val_metadata = json.load(args.val_metadata)
    train_dataset = TimgDataset(train_metadata, crop_size=args.crop_size,
                            crops_per_img=args.batch_size,
                            img_type='logtimg')
    val_dataset = TimgDataset(val_metadata, crop_size=args.crop_size,
                            crops_per_img=1,
                            img_type='logtimg')
    train_sampler = RandomSampler(train_dataset, replacement=True)
    train_dloader = DataLoader(train_dataset, batch_size=None, pin_memory=True,
                            sampler=train_sampler,
                            num_workers=args.dloader_workers,
                            worker_init_fn=lambda _: \
                                    np.random.seed(
                                        seed=as32bit(torch.initial_seed())))
    val_dloader = DataLoader(val_dataset, batch_size=None, pin_memory=True,
                            num_workers=args.dloader_workers,
                            worker_init_fn=lambda _: \
                                    np.random.seed(
                                        seed=as32bit(torch.initial_seed())))
    print('{} batches in train loader.'.format(len(train_dloader)))
    print('{} batches in val loader.'.format(len(val_dloader)))
    
    num_batches = len(train_dloader)
    if args.load_checkpoint is not None:
        model = KPN_MildenhallEtAl_logtimg.load_checkpoint(
                                            args.load_checkpoint).cuda()
    else:
        model = KPN_MildenhallEtAl_logtimg(Kout=args.Kout).cuda()
    model_loss_fn = KPN_MildenhallEtAl.loss_fn(beta=0, alpha=0,
                                            use_l1_vloss=args.use_l1_vloss)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_dir = os.path.join(args.val_output_dir, 'initial')
    validate(model, model_loss_fn, val_dloader, 0,
            output_dir=val_dir, tb_writer=tb_writer)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'state_initial.tar')
    model.save_checkpoint(checkpoint_path, 0)

    if args.epochs > 0:
        try:
            for epoch in range(1, 1+args.epochs):
                train(model, model_optimizer, model_loss_fn, train_dloader,
                    epoch, tb_writer=tb_writer)
                val_dir = os.path.join(args.val_output_dir,
                                        'epoch{:04d}'.format(epoch))
                validate(model, model_loss_fn, val_dloader, epoch,
                        output_dir=val_dir, tb_writer=tb_writer)
                save= 2*(epoch-1) % args.epochs == 0
                if save:
                    checkpoint_path = os.path.join(args.checkpoint_dir,
                                        'state_epoch{:04d}.tar'.format(epoch))
                    model.save_checkpoint(checkpoint_path, epoch)
        except KeyboardInterrupt:
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                        'state_last.tar')
            model.save_checkpoint(checkpoint_path, epoch)
            tb_writer.close()
            sys.exit(0)
    tb_writer.close()

    # Save final checkpoint regardless of if it has been done already.
    checkpoint_path = os.path.join(args.checkpoint_dir, 'state_final.tar')
    model.save_checkpoint(checkpoint_path, -1)
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())

