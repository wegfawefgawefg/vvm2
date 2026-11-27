#!/usr/bin/env bash
# Control on the frozen core, trained without labels, beside an MLP and a GRU
# (old/control7.py; its header says how the state gets in, how the push is read, and how
# the evolution strategy and the policy gradient work; the worlds are vvm2/worlds.py).
#   old/scripts/control7.sh pole 60 seed=0                 balance a pole; evolution strategy
#   old/scripts/control7.sh pole 300 train=pg seed=0       the same by policy gradient
#   old/scripts/control7.sh blind 150 seed=0               the two velocities hidden
#   old/scripts/control7.sh swing 150 seed=0               the pendulum swing-up
#   old/scripts/control7.sh --quick                        a tiny run of both kinds of training
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || scripts/setup.sh
nice -n 19 .venv/bin/python -m old.control7 "$@"
