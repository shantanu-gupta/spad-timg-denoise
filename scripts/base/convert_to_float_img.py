""" scripts.base.convert_to_float_img
"""
import argparse
from src.py.io import save_float_img
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import rescale

def _create_parser():
    help_str = ('Convert an image to a floating point image (with optional '
                'downsampling)')
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input'
    parser.add_argument('input', type=str, help=help_str)
    help_str = 'Output'
    parser.add_argument('output', type=str, help=help_str)
    help_str = 'Downsampling factor (common for both axes)'
    parser.add_argument('-D', '--spatial-downscale', type=int, help=help_str,
                        default=1)
    return parser

def main(args):
    img = img_as_float(imread(args.input, as_gray=True))
    if args.spatial_downscale > 1:
        img = rescale(img, 1. / args.spatial_downscale, anti_aliasing=True,
                    multichannel=False, preserve_range=True)
    save_float_img(args.output, img)
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
