""" scripts.base.create_timg_from_float_img
"""
import os
import argparse
import numpy as np
from skimage.transform import rescale
from src.py.spad import sample_spad_timestamps
from src.py.io import load_float_img

def _create_parser():
    help_str = 'Create a timestamp image from a source image'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input image file (.tiff)'
    parser.add_argument('input', type=str, help=help_str)
    help_str = 'Output timestamp file (npy)'
    parser.add_argument('-T', '--timg', type=str, help=help_str)
    help_str = 'Output log(timestamp) file (npy)'
    parser.add_argument('-L', '--logtimg', type=str, help=help_str)
    help_str = 'RNG seed'
    parser.add_argument('-s', '--rng-seed', type=int, help=help_str)
    help_str = 'Minimum recordable time'
    parser.add_argument('--tmin', type=float, help=help_str)
    help_str = 'Maximum recordable time'
    parser.add_argument('--tmax', type=float, help=help_str)
    help_str = 'How many photons we are recording'
    parser.add_argument('--num-avg', type=int, help=help_str,
                        default=1)
    help_str = 'How to calculate average arrival time'
    parser.add_argument('--avg-fn', type=str, help=help_str,
                        choices=['AM', 'GM'],
                        default='AM')
    return parser

def main(args):
    input_arr = load_float_img(args.input)
    if args.rng_seed is not None:
        np.random.seed(seed=args.rng_seed)
    timg = sample_spad_timestamps(input_arr, N=args.num_avg, tmin=args.tmin,
                                    tmax=args.tmax, avg_fn=args.avg_fn)
    if args.timg is not None:
        np.save(args.timg, timg)
    if args.logtimg is not None:
        np.save(args.logtimg, np.log(timg))

if __name__ == '__main__':
    main(_create_parser().parse_args())
