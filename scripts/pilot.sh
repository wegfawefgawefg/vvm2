#!/usr/bin/env bash
# Pilots, not scoring runs: pick learning rates and weight decay on settings that are in
# no results table. One seed each.
#   scripts/pilot.sh store     storage: 16384 items, a bank of 1024 entries (16 items an
#                              entry); vvm dense lr x bank lr, value init, one tick;
#                              baselines at equal parameters, two learning rates
#   scripts/pilot.sh modadd    modular addition with p = 59: lr x weight decay for the
#                              vvm and the parameter-matched MLP and transformer
# Rows in out/pilot/<name>.jsonl, table printed at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=".venv/bin/python -m vvm2.train --threads 4 --evals 2"
case "${1:-}" in
store)
    RUN="$RUN --out out/pilot/store.jsonl --task assoc --items 16384 --entries 1024 --batch 512 --steps 3000"
    {
        for lrb in 0.003 0.01 0.03; do
            for lr in 0.001 0.003; do
                echo "$RUN --lr $lr --lr_bank $lrb --label 'vvm lr=$lr bank=$lrb' >/dev/null"
            done
        done
        echo "$RUN --lr 0.001 --lr_bank 0.01 --value_std 0.5 --label 'vvm value_std=0.5' >/dev/null"
        echo "$RUN --lr 0.001 --lr_bank 0.01 --ticks 1 --label 'vvm T=1' >/dev/null"
        for m in mlp gru transformer; do
            for lr in 0.001 0.003; do
                echo "$RUN --model $m --match params --lr $lr --label '$m lr=$lr' >/dev/null"
            done
        done
    } | scripts/par.sh
    .venv/bin/python -m vvm2.table out/pilot/store.jsonl
    ;;
modadd)
    RUN="$RUN --out out/pilot/modadd.jsonl --task modadd --p 59 --entries 1024 --batch 512 --steps 4000 --lr_bank 0.03"
    {
        for wd in 0 0.3 1; do
            for lr in 0.001 0.003; do
                echo "$RUN --lr $lr --wd $wd --label 'vvm lr=$lr wd=$wd' >/dev/null"
                for m in mlp transformer; do
                    echo "$RUN --model $m --match params --lr $lr --wd $wd --label '$m lr=$lr wd=$wd' >/dev/null"
                done
            done
        done
    } | scripts/par.sh
    .venv/bin/python -m vvm2.table out/pilot/modadd.jsonl
    ;;
*)
    echo "usage: scripts/pilot.sh store|modadd"; exit 1 ;;
esac
