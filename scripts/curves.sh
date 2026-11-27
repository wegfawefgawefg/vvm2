#!/usr/bin/env bash
# Ordinary curves on the frozen core, beside an MLP: xor, a sine, two spirals
# (vvm2/curves.py; its header says how the numbers are laid into slots and read back).
#   scripts/curves.sh all 3000 seeds=3                 the main table, layout 'one'
#   scripts/curves.sh all 3000 seeds=3 layout=each     one input slot a number
#   scripts/curves.sh spirals 3000 ticks=8 ops=32      more ticks, a bigger bank
#   scripts/curves.sh --quick                          every task, 200 steps, one seed
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m vvm2.curves "$@"
