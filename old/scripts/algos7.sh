#!/usr/bin/env bash
# A ladder of small algorithms on the frozen core (old/algos7.py), each trained on lengths
# 2 to 8 and tested at 8, 16, 32 and 64, with two GRUs beside it. Output is also kept in
# out/algos/.
#   old/scripts/algos7.sh TASK [STEPS] [options]     one run; tasks and options: old/algos7.py
#   old/scripts/algos7.sh seeds TASK STEPS [options] seeds 0, 1 and 2, one after another
#   old/scripts/algos7.sh --quick                    every task for 30 steps, to check it all runs
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || scripts/setup.sh
mkdir -p out/algos
if [ "${1:-}" = "--quick" ]; then
    for task in shift double twice keep count parity compare addbin adddec brackets sort \
                cond condflash; do
        nice -n 19 .venv/bin/python -m old.algos7 "$task" 30 tests=8,16
    done
    exit
fi
if [ "${1:-}" = "seeds" ]; then
    shift
    for seed in 0 1 2; do "$0" "$@" seed=$seed; done
    exit
fi
name="$1-$(echo "${@:2}" | tr ' =' '-_')"
nice -n 19 .venv/bin/python -m old.algos7 "$@" | tee "out/algos/$name.txt"
