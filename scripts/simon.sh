#!/usr/bin/env bash
# The lights game, tested on longer games than trained (vvm2/simon.py).
#   scripts/simon.sh [STEPS] [hold=1 lag=1 max=6 colours=4 scratch=64 seed=0 gru=1 threads=4]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -u -m vvm2.simon "$@"
