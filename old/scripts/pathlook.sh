#!/usr/bin/env bash
# See the header of vvm2/pathlook.py.   scripts/pathlook.sh [TASK ...]    TASK: mnist kmnist fashion
set -euo pipefail
cd "$(dirname "$0")/.."
t="${1:-mnist}"
[ -d "data/$t" ] || scripts/get_data.sh "$t"
nice -n 19 .venv/bin/python -m old.pathlook "$@"
