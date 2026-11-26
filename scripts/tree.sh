#!/usr/bin/env bash
# Does the walk form a tree? (vvm2/tree.py)   scripts/tree.sh [OPS FETCHED TICKS]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/mnist ] || scripts/get_mnist.sh
nice -n 19 .venv/bin/python -m vvm2.tree "${1:-16}" "${2:-2}" "${3:-3}"
