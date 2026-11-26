#!/usr/bin/env bash
# 0d pointer chasing: does the walk form.
#
# Recipe. Task: f a random permutation of 256 items; input (x, h), h in 1..4; target f
# applied h times to x. Training: every (x, 1) and 70% of the pairs with h >= 2. Test: the
# held-out 30%, which only a walk of f can answer. 6000 steps, batch 256, lr 3e-3, bank lr
# 3e-2, weight decay 1 and 0 on the dense matrices; seeds 0 1 2.
# Reference machine: d 64, 1 register, 2 heads, k 8, key width 32, ALU 128, bank 1024
# entries (4 an item), 4 ticks, the input written on the first tick. Lines:
#   machine T=1|2|4|6        ticks against the count h (T=4 is the reference)
#   machine T=4 hold         the input written on every tick, as docs/design.md says
#   machine T=4 N=256        a bank with one entry an item
#   machine T=4 vdecay       decay of the bank rows that were read
#   machine T=4 no bank      the ALU alone
#   machine T=4 no alu       the bank alone
#   machine T=4|6 tick loss  not the task as set: the readout is supervised after every
#                            tick with the walk's item (vvm2/probe.py walk_loss)
#   machine T=4 hold tick loss   the same with the input written on every tick
#   mlp|gru|transformer params|madds    sized to the reference machine
#   gru+4 params|madds       a GRU with 4 more steps on a zero input (a dense walker)
#
#   scripts/0d_chase.sh        rows in out/0d_chase.jsonl
#   .venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd
#   .venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd --groups
#   .venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd --lens "machine T=4"
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=".venv/bin/python -m vvm2.train --threads 4 --out out/0d_chase.jsonl --task chase \
  --items 256 --hops 4 --holdout 0.3 --batch 256 --steps 6000"
REF="--entries 1024 --ticks 4"
{
    for wd in 1 0; do
        for seed in 0 1 2; do
            T="$RUN --wd $wd --seed $seed"
            for ticks in 1 2 4 6; do
                echo "$T --entries 1024 --ticks $ticks --label 'machine T=$ticks' >/dev/null"
            done
            echo "$T $REF --hold 1 --label 'machine T=4 hold' >/dev/null"
            echo "$T --entries 256 --ticks 4 --label 'machine T=4 N=256' >/dev/null"
            echo "$T $REF --vdecay 1 --label 'machine T=4 vdecay' >/dev/null"
            echo "$T $REF --no_bank 1 --label 'machine T=4 no bank' >/dev/null"
            echo "$T $REF --no_alu 1 --label 'machine T=4 no alu' >/dev/null"
            echo "$T $REF --tick_loss 1 --label 'machine T=4 tick loss' >/dev/null"
            echo "$T --entries 1024 --ticks 6 --tick_loss 1 --label 'machine T=6 tick loss' >/dev/null"
            echo "$T $REF --hold 1 --tick_loss 1 --label 'machine T=4 hold tick loss' >/dev/null"
            for match in params madds; do
                for model in mlp gru transformer; do
                    echo "$T $REF --model $model --match $match --label '$model $match' >/dev/null"
                done
                echo "$T $REF --model gru --ponder 4 --match $match --label 'gru+4 $match' >/dev/null"
            done
        done
    done
} | scripts/par.sh
.venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd
.venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd --groups
