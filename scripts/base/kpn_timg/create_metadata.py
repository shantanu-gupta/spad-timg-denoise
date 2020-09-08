""" scripts.base.kpn_timg.create_metadata
"""
import os.path as osp
import sys
import argparse
import json
from src.py.io import read_listfile

def _create_parser():
    help_str = 'Create metadata file from list of image paths' 
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input list file'
    parser.add_argument('listfile', nargs='?', type=argparse.FileType('r'),
                        help=help_str,
                        default=sys.stdin)
    help_str = 'Output metadata file'
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'),
                        help=help_str,
                        default=sys.stdout)
    help_str = 'Original data base directory'
    parser.add_argument('--orig-data-dir', type=str, help=help_str,
                        default='')
    help_str = 'Generated data base directory'
    parser.add_argument('--gen-data-dir', type=str, help=help_str)
    help_str = 'Spatial downscale'
    parser.add_argument('--spatial-downscale', type=int, help=help_str)
    help_str = 'Max photon arrival rate per unit time'
    parser.add_argument('--max-photon-rate', type=float, help=help_str,
                        default=1.0)
    help_str = 'How many timgs to simulate for a given true image?'
    parser.add_argument('--num-timgs', type=int, help=help_str,
                        default=1)
    help_str = 'Minimum recordable time'
    parser.add_argument('--tmin', type=float, help=help_str)
    help_str = 'Maximum recordable time'
    parser.add_argument('--tmax', type=float, help=help_str)
    help_str = 'Number of photons recorded in a single image'
    parser.add_argument('--num-avg', type=int, help=help_str,
                        default=1)
    help_str = 'How the "average" timestamp is computed'
    parser.add_argument('--avg-fn', type=str, help=help_str,
                        choices=['AM', 'GM'],
                        default='AM')
    help_str = 'Whether to fix RNG seeds when generating timgs'
    parser.add_argument('--fix-rng-seeds', action='store_true',
                        help=help_str)
    help_str = 'RNG seed base'
    parser.add_argument('--seed-base', type=int, help=help_str,
                        default=0)
    return parser

def _img_name_div2k(path):
    return osp.splitext(osp.basename(path))[0]

def _img_metadata_div2k(args, img_path, img_idx):
    datadir = osp.join(args.gen_data_dir, _img_name_div2k(img_path))
    timg_dir = osp.join(datadir, 'timgs')
    logtimg_dir = osp.join(datadir, 'logtimgs')
    info = []
    for n in range(args.num_timgs):
        if args.fix_rng_seeds:
            rng_seed = args.seed_base + img_idx * args.num_timgs + n
        else:
            rng_seed = None
        timg_path = osp.join(timg_dir, '{}.npy'.format(n))
        logtimg_path = osp.join(logtimg_dir, '{}.npy'.format(n))
        info.append({'timg': timg_path,
                    'logtimg': logtimg_path,
                    'rng-seed': rng_seed,
                    })
    metadata = {
        'original-path': img_path,
        'data-dir': datadir,
        'timg-dir': timg_dir,
        'logtimg-dir': logtimg_dir,
        'original-copy-path': osp.join(datadir, 'original.png'),
        'grayscale-path': osp.join(datadir, 'grayscale.tiff'),
        'true-timg-path': osp.join(datadir, 'true_timg.tiff'),
        'true-logtimg-path': osp.join(datadir, 'true_logtimg.tiff'),
        'timgs': info,
    }
    return metadata

def main(args):
    img_paths = [osp.join(args.orig_data_dir, p)
                for p in read_listfile(args.listfile)]
    img_entries = [_img_metadata_div2k(args, p, i)
                for i, p in enumerate(img_paths)]
    metadata = {
            'data-dir': args.gen_data_dir,
            'num-timgs': args.num_timgs,
            'spatial-downscale': args.spatial_downscale,
            'max-photon-rate': args.max_photon_rate,
            'tmin': args.tmin,
            'tmax': args.tmax,
            'num-avg': args.num_avg,
            'avg-fn': args.avg_fn,
            'image-metadata': img_entries,
    }
    json.dump(metadata, args.output, indent=2)
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
