#!/usr/bin/env bash
# What each vop does on MNIST and what the walks add up to: text, and three pictures in
# docs/img/ (walk-mnist.png, program-mnist.png, vops-mnist.png).   scripts/look.sh [OPS FETCHED TICKS]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/mnist ] || scripts/get_mnist.sh
nice -n 19 .venv/bin/python -m old.look "${1:-16}" "${2:-2}" "${3:-3}"
