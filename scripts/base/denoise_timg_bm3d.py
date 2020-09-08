""" scripts.base.denoise_timg_bm3d
"""
import argparse
import numpy as np
import bm3d
import src.py.io as io_utils

def _create_parser():
    help_str = 'Denoise a single timg or logtimg using BM3D'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input file (npy)'
    parser.add_argument('input', type=str, help=help_str)
    help_str = 'Output file (png)'
    parser.add_argument('output', type=str, help=help_str)
    help_str = 'Sigma parameter for BM3D'
    parser.add_argument('--sigma-bm3d', type=float, help=help_str,
                        default=0.1)
    help_str = 'Is this image a logtimg?'
    parser.add_argument('--is-logt', action='store_true', help=help_str)
    help_str = 'Bias correction term (needed for logtimgs, 0 otherwise)'
    parser.add_argument('--bias-correction', type=float, help=help_str,
                        default=0.5722)
    return parser

def main(args):
    input_arr = np.load(args.input)
    H, W = input_arr.shape
    L = 256
    gap_H, gap_W = H % (L//2), W % (L//2)
    i0, i1 = gap_H//2, H - (gap_H - (gap_H//2))
    j0, j1 = gap_W//2, H - (gap_W - (gap_W//2))
    input_arr = input_arr[i0:i1, j0:j1]
    H, W = input_arr.shape
    if args.is_logt:
        logtimg_norm = 0.5 * (1 + (input_arr / (3 * np.log(10))))
        output = bm3d.bm3d(logtimg_norm, args.sigma_bm3d, profile='high')
        output += ((0.5 * args.bias_correction) / (3 * np.log(10)))
        output = np.clip(output, 0.5, None)
        output = np.exp(-((2 * output - 1) * (3 * np.log(10))))
    else:
        assert input_arr.min() >= 0
        max_val = input_arr.max()
        timg_norm = input_arr / max_val
        output = max_val * bm3d.bm3d(timg_norm, args.sigma_bm3d, profile='high')
    io_utils.save_img(io_utils.array_to_img(output, 'L'), args.output)
    return

if __name__ == '__main__':
    parser = _create_parser()
    args = parser.parse_args()
    main(args)
