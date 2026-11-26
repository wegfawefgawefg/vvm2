#!/usr/bin/env bash
# 0b stored associations: accuracy on the stored pairs against the number of pairs.
#
# Recipe. Task: `items` random pairs; key = 8 random base-16 digits, value = one of 64
# classes (6 bits a pair). items in 4096, 16384, 65536; steps 4000, 6000, 12000; batch 512;
# lr 3e-3, bank lr 3e-2 (from scripts/pilot.sh store), no weight decay; seeds 0 1 2.
# Reference machine: d 64, 1 register, 2 heads, k 8, key width 32, ALU 128, bank 4096
# entries, 2 ticks. Lines:
#   machine                  the reference
#   machine T=1              one tick
#   machine N=1024           a bank a quarter the size
#   machine N=65536          a bank 16 times the size (items >= 16384 only)
#   machine no bank          the ALU alone
#   mlp|gru|transformer params   sized to the reference machine's parameter count
#   mlp|gru|transformer madds    sized to the reference machine's multiply-adds
#
#   scripts/0b_assoc.sh        rows in out/0b_assoc.jsonl; finished runs are skipped
#   .venv/bin/python -m vvm2.table out/0b_assoc.jsonl --split items
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=".venv/bin/python -m vvm2.train --threads 4 --out out/0b_assoc.jsonl --task assoc --batch 512"
REF="--entries 4096 --ticks 2"
{
    for pair in 4096:4000 16384:6000 65536:12000; do
        items=${pair%%:*}
        T="$RUN --items $items --steps ${pair##*:}"
        for seed in 0 1 2; do
            echo "$T $REF --seed $seed --label 'machine' >/dev/null"
            echo "$T --entries 4096 --ticks 1 --seed $seed --label 'machine T=1' >/dev/null"
            echo "$T --entries 1024 --ticks 2 --seed $seed --label 'machine N=1024' >/dev/null"
            if [ "$items" -ge 16384 ]; then
                echo "$T --entries 65536 --ticks 2 --seed $seed --label 'machine N=65536' >/dev/null"
            fi
            echo "$T $REF --no_bank 1 --seed $seed --label 'machine no bank' >/dev/null"
            for model in mlp gru transformer; do
                for match in params madds; do
                    echo "$T $REF --model $model --match $match --seed $seed --label '$model $match' >/dev/null"
                done
            done
        done
    done
} | scripts/par.sh
.venv/bin/python -m vvm2.table out/0b_assoc.jsonl --split items
