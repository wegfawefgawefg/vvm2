#!/usr/bin/env bash
# Lists and the lights game, tested on longer cases than trained (vvm2/tape.py).
#   scripts/tape.sh copy | reverse | simon [STEPS]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m vvm2.tape "$@"
