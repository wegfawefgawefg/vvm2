#!/usr/bin/env bash
# One box per vop on MNIST with small (rank-2) vops: what each is fetched for, what it looks
# at, what it does. Picture: docs/img/vops-small.png.   scripts/lookvops.sh [OPS]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/mnist ] || scripts/get_mnist.sh
nice -n 19 .venv/bin/python -m vvm2.lookvops "${1:-64}"
