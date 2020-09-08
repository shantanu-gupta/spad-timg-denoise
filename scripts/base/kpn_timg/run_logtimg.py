""" scripts.base.kpn_timg.run_logtimg
"""
import argparse
import numpy as np
from src.py.kpn import KPN_MildenhallEtAl_logtimg
import torch

def _create_parser():
    help_str = 'Denoise a single logtimg using a KPN'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input (.npy)'
    parser.add_argument('input', type=str, help=help_str)
    help_str = 'Output (.npy)'
    parser.add_argument('output', type=str, help=help_str)
    help_str = 'Model checkpoint'
    parser.add_argument('-m', '--model', type=str, help=help_str,
                        required=True)
    help_str = 'Centre crop granularity'
    parser.add_argument('--crop-granularity', type=int, help=help_str,
                        default=128)
    return parser

def main(args):
    logtimg = torch.from_numpy(np.load(args.input)).type(torch.float)
    L = args.crop_granularity
    H, W = logtimg.shape
    gap_H, gap_W = H % (L//2), W % (L//2)
    i0, i1 = gap_H//2, H - (gap_H - (gap_H//2))
    j0, j1 = gap_W//2, W - (gap_W - (gap_W//2))
    logtimg = logtimg[i0:i1, j0:j1]
    if torch.cuda.is_available():
        logtimg = logtimg.cuda()
    logtimg = logtimg.view(1, 1, logtimg.shape[0], logtimg.shape[1])
    model = KPN_MildenhallEtAl_logtimg.load_checkpoint(args.model)
    denoised = model(logtimg)[0][0,0,:,:]
    if torch.cuda.is_available():
        denoised = denoised.cpu()
    denoised = denoised.detach().numpy()
    np.save(args.output, denoised)
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
