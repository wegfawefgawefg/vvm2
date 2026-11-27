#!/usr/bin/env bash
# vvm on a vector RAM (vvm2/ram.py): memory is slots of 16 numbers; a vop reads two slots
# and writes one.   scripts/ram.sh [mnist|kmnist|fashion]
set -euo pipefail
cd "$(dirname "$0")/.."
t="${1:-mnist}"
[ -d "data/$t" ] || scripts/get_data.sh "$t"
nice -n 19 .venv/bin/python -m old.ram "$t"
