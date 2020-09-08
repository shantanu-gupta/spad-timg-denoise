""" scripts.base.image_comparison
"""
import argparse
import numpy as np
import src.py.io as io_utils
from PIL import Image, ImageFont, ImageDraw

def _create_parser():
    help_str = 'Generate side-by-side comparison of multiple images'
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Output'
    parser.add_argument('output', type=str, help=help_str)
    help_str = 'Images'
    parser.add_argument('-i', '--images', type=str, help=help_str, nargs='+')
    help_str = 'Labels'
    parser.add_argument('-l', '--labels', type=str, help=help_str, nargs='+')
    help_str = 'Crop size'
    parser.add_argument('-c', '--crop-size', type=int, help=help_str)
    help_str = 'Crop top-left corner (x, y)'
    parser.add_argument('-o', '--crop-origin', type=int, help=help_str, nargs=2,
                        default=(10, 10))
    help_str = 'Text intensity/"color" -- 0 is black, 1 is white'
    parser.add_argument('-v', '--text-val', type=float, help=help_str,
                        default=1)
    help_str = 'Font location (expected to be truetype)'
    parser.add_argument('-f', '--font', type=str, help=help_str)
    help_str = 'Font size'
    parser.add_argument('-s', '--font-size', type=int, help=help_str,
                        default=11)
    help_str = 'Label location (x, y)'
    parser.add_argument('-x', '--label-location', type=int, help=help_str,
                        nargs=2,
                        default=(5, 5))
    help_str = 'Image grid size'
    parser.add_argument('--image-grid-size', type=int, help=help_str,
                        nargs=2)
    return parser

def main(args):
    imgs = (Image.fromarray(io_utils.load_float_img(p)) for p in args.images)
    if args.labels is None:
        args.labels = ('Image {}'.format(i) for i in range(len(args.images)))
    else:
        assert len(args.labels) == len(args.images)
    if args.crop_size is not None:
        cx, cy = args.crop_origin
        csize = args.crop_size
        imgs = (y.crop(box=(cx, cy, cx+csize, cy+csize)) for y in imgs)
    if args.font is not None:
        font = ImageFont.truetype(args.font, args.font_size)
    else:
        font = ImageFont.load_default()
    fx, fy = args.label_location
    imgs_wtext = []
    for img, label in zip(imgs, args.labels):
        ImageDraw.Draw(img).text((fx, fy), label, args.text_val, font=font)
        imgs_wtext.append(np.array(img))
    if args.image_grid_size is None:
        H, W = imgs_wtext[0].shape
        if H < W:
            comparison = np.vstack(imgs_wtext)
        else:
            comparison = np.hstack(imgs_wtext)
    else:
        gH, gW = args.image_grid_size
        rows = []
        for i in range(gH):
            row = []
            for j in range(gW):
                idx = i * gW + j
                if idx >= len(imgs_wtext):
                    row.append(np.zeros_like(imgs_wtext[0]))
                else:
                    row.append(imgs_wtext[idx])
            row = np.hstack(row)
            rows.append(row)
        comparison = np.vstack(rows)
    io_utils.save_float_img(args.output, comparison)
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
