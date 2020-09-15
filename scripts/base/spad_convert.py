""" scripts.base.spad_convert
"""
import argparse
import numpy as np
from skimage.util import img_as_float
from src.py.spad import invert_spad_timg, spad_timg
from src.py.spad import invert_spad_logtimg, spad_logtimg
import src.py.io as io_utils

def _create_parser():
    help_str = 'Convert between radiance and SPAD-related images'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input image (.tiff or .npy)'
    parser.add_argument('input', type=str, help=help_str)
    help_str = 'Output image (.tiff or .npy)'
    parser.add_argument('output', type=str, help=help_str)
    help_str = 'Type of mapping'
    parser.add_argument('-m', '--mapping', type=str, help=help_str,
                        choices=['counts-to-radiance',
                                'timg-to-radiance',
                                'logtimg-to-radiance',
                                'radiance-to-counts',
                                'radiance-to-timg',
                                'radiance-to-logtimg'],
                        default='counts-to-radiance')
    help_str = 'Max photon count (needed for some mappings)'
    parser.add_argument('-p', '--max-photon-rate', type=float, help=help_str)
    return parser

def main(args):
    if args.input.endswith('.tiff'):
        img = io_utils.load_float_img(args.input)
    elif args.input.endswith('.npy'):
        img = np.load(args.input).astype(float)
    else:
        raise NotImplementedError

    max_rate = args.max_photon_rate
    if args.mapping == 'counts-to-radiance':
        img = invert_spad_avgcounts(img)
    elif args.mapping == 'timg-to-radiance':
        img = invert_spad_timg(img)
    elif args.mapping == 'logtimg-to-radiance':
        img = invert_spad_logtimg(img)
    elif args.mapping == 'radiance-to-counts':
        img = spad_p1(img * max_rate)
    elif args.mapping == 'radiance-to-timg':
        img = spad_timg(img * max_rate)
    elif args.mapping == 'radiance-to-logtimg':
        img = spad_logtimg(img * max_rate)

    if args.output.endswith('.tiff'):
        io_utils.save_float_img(args.output, img)
    elif args.output.endswith('.npy'):
        np.save(args.output, img)
    else:
        raise NotImplementedError
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())

