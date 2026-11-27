#!/usr/bin/env bash
# The ordinary tasks: the vvm against an MLP of the same parameter count.
#   scripts/classic.sh xor | sin | spirals | mnist [--steps N --seeds N --entries N ...]
#   scripts/classic.sh all
# mnist needs scripts/get_mnist.sh once. Flags are those of `python -m old.vop -h`.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
if [ "${1:-all}" = all ]; then
    for t in xor sin spirals; do nice -n 19 .venv/bin/python -m old.vop "$t"; echo; done
    [ -d data/mnist ] || scripts/get_mnist.sh
    nice -n 19 .venv/bin/python -m old.vop mnist --steps 3000 --seeds 2 --threads 8
else
    nice -n 19 .venv/bin/python -m old.vop "$@"
fi
