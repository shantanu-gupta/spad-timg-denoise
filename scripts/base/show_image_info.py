""" scripts.base.show_image_info
"""
import numpy as np
import argparse
from skimage.util import img_as_float
from src.py.spad import invert_spad_avgcounts, spad_p1
from src.py.io import load_float_img

def _create_parser():
    help_str = 'Show image info'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input'
    parser.add_argument('input', type=str, help=help_str)
    help_str = 'Gain'
    parser.add_argument('--gain', type=float, help=help_str,
                        default=1.0)
    help_str = 'Low percentile'
    parser.add_argument('--low-percentile', type=float, help=help_str,
                        default=3)
    help_str = 'High percentile'
    parser.add_argument('--high-percentile', type=float, help=help_str,
                        default=97)
    help_str = 'Type of image (representing radiance or counts)'
    parser.add_argument('--image-type', type=str, help=help_str,
                        choices=['radiance', 'counts'],
                        default='radiance')
    return parser

def main(args):
    if args.input.endswith('.tiff'):
        image = load_float_img(args.input) * args.gain
    else:
        raise NotImplementedError
    str_parts = []
    str_parts.append('Size {}'.format(image.shape))
    v0, vlp, vhp, v100 = np.percentile(image,
                                    [0,
                                    args.low_percentile,
                                    args.high_percentile,
                                    100],
                                    interpolation='nearest')
    str_parts.append('min={:.3f},'.format(v0))
    str_parts.append('{:.3f} percentile={:.3f},'.format(
                                                    args.low_percentile,
                                                    vlp))
    str_parts.append('{:.3f} percentile={:.3f},'.format(
                                                    args.high_percentile,
                                                    vhp))
    str_parts.append('max={:.3f}'.format(v100))
    if args.image_type == 'counts':
        max_photon_rate_est = invert_spad_avgcounts(vhp)
        str_parts.append('max_photon_rate_estimate={:3f}'
                            .format(max_photon_rate_est))
    elif args.image_type == 'radiance':
        max_avg_counts_est = spad_p1(vhp)
        str_parts.append('max_count_estimate={:3f}'.format(max_avg_counts_est))
    print(' '.join(str_parts))
    return

if __name__ == '__main__':
    parser = _create_parser()
    args = parser.parse_args()
    main(args)
