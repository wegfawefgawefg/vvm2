#!/usr/bin/env bash
# Cellular automata learned on small boards and run on larger ones (old/automata7.py).
#   old/scripts/automata7.sh line  [STEPS] [rule=110] [edge=wall|ring] [reads=3] [k=1] [seed=0]
#   old/scripts/automata7.sh rules [STEPS] [reads=4]      the rule's table given beside the row
#   old/scripts/automata7.sh life  [STEPS]                Conway's game on a grid
#   old/scripts/automata7.sh --quick                      every task for a few steps
# Other options: ops=128 tpc=2 base=0 sizes=12,24 whole=0.5. One job, 1 thread, nice 19.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m old.automata7 "$@"
