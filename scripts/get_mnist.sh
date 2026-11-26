#!/usr/bin/env bash
# Download the four standard MNIST idx files into data/mnist/ (git-ignored), check them, and
# unpack them. Safe to rerun: files that are already there and check out are kept.
#   scripts/get_mnist.sh
# Mirrors, tried in order: the PyTorch S3 mirror, then Google's CVDF mirror. Each .gz is
# checked against the md5 torchvision uses, and each unpacked file against its known size.
set -euo pipefail
cd "$(dirname "$0")/.."
dir=data/mnist
mkdir -p "$dir"
mirrors=(
    https://ossci-datasets.s3.amazonaws.com/mnist
    https://storage.googleapis.com/cvdf-datasets/mnist
)
# name, md5 of the .gz, size of the unpacked file
files=(
    "train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873 47040016"
    "train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432 60008"
    "t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3 7840016"
    "t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c 10008"
)
for entry in "${files[@]}"; do
    read -r name md5 size <<<"$entry"
    if [ -f "$dir/$name" ] && [ "$(stat -c %s "$dir/$name")" = "$size" ]; then
        echo "have $name"
        continue
    fi
    got=0
    for m in "${mirrors[@]}"; do
        if curl -fsSL --retry 2 -o "$dir/$name.gz" "$m/$name.gz" &&
            [ "$(md5sum "$dir/$name.gz" | cut -d' ' -f1)" = "$md5" ]; then
            got=1
            break
        fi
        echo "mirror $m failed or gave a bad file for $name" >&2
    done
    if [ "$got" != 1 ]; then
        echo "could not get $name" >&2
        exit 1
    fi
    gunzip -f "$dir/$name.gz"
    if [ "$(stat -c %s "$dir/$name")" != "$size" ]; then
        echo "$name unpacked to the wrong size" >&2
        exit 1
    fi
    echo "got $name"
done
