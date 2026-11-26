#!/usr/bin/env bash
# One box per vop on MNIST with small (rank-2) vops: what each is fetched for, what it looks
# at, what it does. Also how unevenly vops are used and what knocking some out costs.
#   scripts/lookvops.sh                                  64 vops, 3,000 steps -> docs/img/vops-small.png
#   scripts/lookvops.sh 64 12000 0.05 2 vops-small-trained   steps, weight decay, shift, picture name
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/mnist ] || scripts/get_mnist.sh
nice -n 19 .venv/bin/python -m vvm2.lookvops "$@"
