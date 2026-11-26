#!/usr/bin/env python3
"""Kannada-MNIST's npz files to MNIST's four idx files (scripts/get_data.sh kannada).

  npz_to_idx.py NPZ_DIR OUT_DIR

Reads X_kannada_MNIST_{train,test}.npz (N x 28 x 28 bytes) and y_... (N labels 0-9) with
pickles refused, checks the shapes, and writes train-images-idx3-ubyte and the other three.
"""

import os
import struct
import sys

import numpy as np


def first_array(path):
    with np.load(path, allow_pickle=False) as z:
        return z[z.files[0]]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    for part, name, count in (("train", "train", 60000), ("test", "t10k", 10000)):
        x = first_array(os.path.join(src, "X_kannada_MNIST_%s.npz" % part))
        y = first_array(os.path.join(src, "y_kannada_MNIST_%s.npz" % part))
        if x.shape != (count, 28, 28) or y.shape != (count,):
            print("kannada %s: shapes %s %s, wanted %d x 28 x 28" % (part, x.shape, y.shape, count))
            return 1
        if x.min() < 0 or x.max() > 255 or y.min() < 0 or y.max() > 9:
            print("kannada %s: values out of range" % part)
            return 1
        with open(os.path.join(dst, "%s-images-idx3-ubyte" % name), "wb") as f:
            f.write(struct.pack(">IIII", 0x803, count, 28, 28))
            f.write(np.ascontiguousarray(x, dtype=np.uint8).tobytes())
        with open(os.path.join(dst, "%s-labels-idx1-ubyte" % name), "wb") as f:
            f.write(struct.pack(">II", 0x801, count))
            f.write(np.ascontiguousarray(y, dtype=np.uint8).tobytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
