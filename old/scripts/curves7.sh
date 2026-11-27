#!/usr/bin/env bash
# Ordinary curves on the frozen core, beside an MLP: xor, a sine, two spirals
# (old/curves7.py; its header says how the numbers are laid into slots and read back).
#   old/scripts/curves7.sh all 3000 seeds=3                 the main table, layout 'one'
#   old/scripts/curves7.sh all 3000 seeds=3 layout=each     one input slot a number
#   old/scripts/curves7.sh spirals 3000 ticks=8 ops=32      more ticks, a bigger bank
#   old/scripts/curves7.sh --quick                          every task, 200 steps, one seed
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m old.curves7 "$@"
