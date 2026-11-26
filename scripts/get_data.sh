#!/usr/bin/env bash
# Download a data set in MNIST's own file format (28 by 28 grey, ten classes, 60,000 training
# and 10,000 test images, four idx files) into data/NAME/ (git-ignored), check it, unpack it.
# Safe to rerun: files that are already there and check out are kept.
#   scripts/get_data.sh fashion     Fashion-MNIST: clothes (Zalando Research; MIT license)
#   scripts/get_data.sh kmnist      Kuzushiji-MNIST: cursive Japanese characters (CODH; CC BY-SA 4.0)
#   scripts/get_data.sh kannada     Kannada-MNIST: Kannada digits (V. U. Prabhu; CC BY 4.0 by its
#                                   paper and listings, the repository has no license file),
#                                   made into idx files from the author's npz files (numpy)
#   scripts/get_data.sh mnist       the same as scripts/get_mnist.sh
#   scripts/get_data.sh all         fashion, kmnist and kannada
# Plain anonymous downloads from each set's public home. Each .gz is checked against the md5
# torchvision uses, each unpacked file against its known size. To use a set, name it as the
# task: scripts/classic.sh kmnist, scripts/pathlook.sh fashion, scripts/halt.sh kmnist.
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:-}"
sizes=(
    "train-images-idx3-ubyte 47040016"
    "train-labels-idx1-ubyte 60008"
    "t10k-images-idx3-ubyte 7840016"
    "t10k-labels-idx1-ubyte 10008"
)

have_all() {
    for entry in "${sizes[@]}"; do
        read -r f size <<<"$entry"
        [ -f "$1/$f" ] && [ "$(stat -c %s "$1/$f")" = "$size" ] || return 1
    done
}

# idx .gz files from a list of mirrors: fetch_idx DIR "mirror ..." "file md5" ...
fetch_idx() {
    local dir="$1" mirrors="$2"
    shift 2
    mkdir -p "$dir"
    for entry in "$@"; do
        read -r f md5 <<<"$entry"
        local size=""
        for s in "${sizes[@]}"; do
            [ "${s%% *}" = "$f" ] && size="${s##* }"
        done
        if [ -f "$dir/$f" ] && [ "$(stat -c %s "$dir/$f")" = "$size" ]; then
            echo "have $dir/$f"
            continue
        fi
        local got=0
        for m in $mirrors; do
            if curl -fsSL --retry 2 -o "$dir/$f.gz" "$m/$f.gz" &&
                [ "$(md5sum "$dir/$f.gz" | cut -d' ' -f1)" = "$md5" ]; then
                got=1
                break
            fi
            echo "mirror $m failed or gave a bad file for $f" >&2
        done
        [ "$got" = 1 ] || { echo "could not get $f" >&2; exit 1; }
        gunzip -f "$dir/$f.gz"
        [ "$(stat -c %s "$dir/$f")" = "$size" ] || { echo "$f unpacked to the wrong size" >&2; exit 1; }
        echo "got $dir/$f"
    done
}

case "$name" in
mnist)
    exec scripts/get_mnist.sh
    ;;
fashion)
    fetch_idx data/fashion \
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com https://ossci-datasets.s3.amazonaws.com/fashion-mnist" \
        "train-images-idx3-ubyte 8d4fb7e6c68d591d4c3dfef9ec88bf0d" \
        "train-labels-idx1-ubyte 25c81989df183df01b3e8a0aad5dffbe" \
        "t10k-images-idx3-ubyte bef4ecab320f06d8554ea6380940ec79" \
        "t10k-labels-idx1-ubyte bb300cfdad3c16e7a12a480ee83cd310"
    ;;
kmnist)
    fetch_idx data/kmnist "http://codh.rois.ac.jp/kmnist/dataset/kmnist" \
        "train-images-idx3-ubyte bdb82020997e1d708af4cf47b453dcf7" \
        "train-labels-idx1-ubyte e144d726b3acfaa3e44228e80efcd344" \
        "t10k-images-idx3-ubyte 5c965bf0a639b31b8f53240b1b52f4d7" \
        "t10k-labels-idx1-ubyte 7320c461ea6c1c855c0b718fb2a4b134"
    ;;
kannada)
    dir=data/kannada
    if have_all "$dir"; then
        echo "have $dir"
        exit 0
    fi
    # the author's npz files, kept apart from the idx files made from them
    raw=data/kannada-npz
    mkdir -p "$dir" "$raw"
    base=https://raw.githubusercontent.com/vinayprabhu/Kannada_MNIST/master/data/output_tensors/MNIST_format
    for f in X_kannada_MNIST_train y_kannada_MNIST_train X_kannada_MNIST_test y_kannada_MNIST_test; do
        [ -s "$raw/$f.npz" ] || curl -fsSL --retry 2 -o "$raw/$f.npz" "$base/$f.npz"
    done
    py=python3
    [ -x "${LADDER_VENV:-out/ladder2/venv}/bin/python" ] && py="${LADDER_VENV:-out/ladder2/venv}/bin/python"
    "$py" -I scripts/npz_to_idx.py "$raw" "$dir"
    have_all "$dir" || { echo "kannada: the idx files came out the wrong size" >&2; exit 1; }
    echo "got $dir"
    ;;
all)
    scripts/get_data.sh fashion
    scripts/get_data.sh kmnist
    scripts/get_data.sh kannada
    ;;
*)
    sed -n '2,14p' "$0"
    exit 1
    ;;
esac
