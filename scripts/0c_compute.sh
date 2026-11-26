#!/usr/bin/env bash
# 0c modular addition and parity: can retrieved edits compute and generalise.
#
# Recipe, modular addition: (a + b) mod 97, 60% of the 9409 pairs trained on, the rest the
# test. 8000 steps, batch 512, lr 3e-3, bank lr 3e-2, weight decay 1 on the dense matrices
# (from scripts/pilot.sh modadd: at p = 59 the MLP and the transformer generalised only at
# this setting). Seeds 0 1 2.
# Recipe, parity: 12 bits, 60% of the 4096 strings trained on. 6000 steps, batch 512, the
# same learning rates, weight decay 0 and 1 (not piloted, so both).
# Reference vvm: d 64, 1 register, 2 heads, k 8, key width 32, ALU 128, bank 4096
# entries, 2 ticks, the tokens of an example written as one input. Lines:
#   vvm                  the reference
#   vvm T=4              four ticks
#   vvm vdecay           plus decay of the bank rows that were read (--vdecay 1)
#   vvm no bank          the ALU alone
#   vvm seq              parity only: one bit an input, one tick a bit
#   mlp|gru|transformer params|madds    sized to the reference vvm
#
#   scripts/0c_compute.sh      rows in out/0c_modadd.jsonl and out/0c_parity.jsonl
#   .venv/bin/python -m vvm2.table out/0c_modadd.jsonl
#   .venv/bin/python -m vvm2.table out/0c_parity.jsonl --split wd
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=".venv/bin/python -m vvm2.train --threads 4 --batch 512"
REF="--entries 4096 --ticks 2"

lines() {  # $1: the run prefix with task and training settings; $2: seed
    echo "$1 $REF --seed $2 --label 'vvm' >/dev/null"
    echo "$1 --entries 4096 --ticks 4 --seed $2 --label 'vvm T=4' >/dev/null"
    echo "$1 $REF --vdecay 1 --seed $2 --label 'vvm vdecay' >/dev/null"
    echo "$1 $REF --no_bank 1 --seed $2 --label 'vvm no bank' >/dev/null"
    for model in mlp gru transformer; do
        for match in params madds; do
            echo "$1 $REF --model $model --match $match --seed $2 --label '$model $match' >/dev/null"
        done
    done
}

{
    for seed in 0 1 2; do
        lines "$RUN --out out/0c_modadd.jsonl --task modadd --p 97 --train_share 0.6 --steps 8000 --wd 1" $seed
        for wd in 0 1; do
            P="$RUN --out out/0c_parity.jsonl --task parity --bits 12 --train_share 0.6 --steps 6000 --wd $wd"
            lines "$P" $seed
            echo "$P --entries 4096 --feed seq --ticks 1 --seed $seed --label 'vvm seq' >/dev/null"
        done
    done
} | scripts/par.sh
.venv/bin/python -m vvm2.table out/0c_modadd.jsonl
.venv/bin/python -m vvm2.table out/0c_parity.jsonl --split wd
