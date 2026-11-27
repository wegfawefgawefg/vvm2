#!/usr/bin/env bash
# Cellular automata learned on small boards and run on larger ones (vvm2/automata.py).
#   scripts/automata.sh line  [STEPS] [rule=110] [edge=wall|ring] [reads=3] [k=1] [seed=0]
#   scripts/automata.sh rules [STEPS] [reads=4]      the rule's table given beside the row
#   scripts/automata.sh life  [STEPS]                Conway's game on a grid
#   scripts/automata.sh --quick                      every task for a few steps
# Other options: ops=128 tpc=2 base=0 sizes=12,24. One job, 4 threads, nice 19.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m vvm2.automata "$@"
