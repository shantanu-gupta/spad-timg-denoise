### io.py

import os
import numpy as np
import skimage.io as skio

def read_listfile(listfile):
    def _read_fplines(fp):
        return [line.strip() for line in fp.readlines() if line.strip() != '']

    assert listfile is not None
    if isinstance(listfile, str):
        with open(listfile) as f:
            paths = _read_fplines(f)
    else:
        paths = _read_fplines(listfile)
    return paths

def list_files_sorted(dir_path):
    return sorted([os.path.join(dir_path, p) for p in os.listdir(dir_path)])

def load_float_img(fpath):
    assert fpath.endswith('.tiff')
    return skio.imread(fpath, as_gray=True, plugin='tifffile')

def save_float_img(fpath, img):
    assert fpath.endswith('.tiff')
    skio.imsave(fpath, img.astype(np.float32), plugin='tifffile')
    return

