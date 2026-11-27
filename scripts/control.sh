#!/usr/bin/env bash
# Control on the frozen core, trained without labels, beside an MLP and a GRU
# (vvm2/control.py; its header says how the state gets in, how the push is read, and how
# the evolution strategy and the policy gradient work; the worlds are vvm2/worlds.py).
#   scripts/control.sh pole 60 seed=0                 balance a pole; evolution strategy
#   scripts/control.sh pole 300 train=pg seed=0       the same by policy gradient
#   scripts/control.sh blind 150 seed=0               the two velocities hidden
#   scripts/control.sh swing 150 seed=0               the pendulum swing-up
#   scripts/control.sh --quick                        a tiny run of both kinds of training
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m vvm2.control "$@"
