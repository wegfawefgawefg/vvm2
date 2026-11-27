#!/usr/bin/env bash
# The machine on MNIST, Kuzushiji-MNIST or Fashion-MNIST beside an MLP (vvm2/image.py).
#   scripts/image.sh [mnist|kmnist|fashion] [STEPS]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
t="${1:-mnist}"
[ -d "data/$t" ] || scripts/get_data.sh "$t"
nice -n 19 .venv/bin/python -m old.image6 "$@"
