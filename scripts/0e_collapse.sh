#!/usr/bin/env bash
# 0e, a first look: how collapsed is the bank with no remedy, and which noise helps most.
#
# Recipe. Two settings where the bank is larger than the task needs, so unused entries
# show. Seeds 0 1 2. One strength per switch; this is a screen, not a tuning.
#   store   0b storage, 16384 items, bank 65536 entries (4 entries an item), 2 ticks,
#           6000 steps, batch 512, no weight decay
#   chase   0d pointer chasing as in scripts/0d_chase.sh, bank 4096 entries (4 entries a
#           training pair), 4 ticks, 6000 steps, batch 256, weight decay 1
# Lines (the machine otherwise as in 0b and 0d; all noise is in training only):
#   none             no remedy; query normalised by layer norm, the default
#   pick_from 32     the exact top 32 entries, 8 of them kept at random
#   gumbel 1         Gumbel noise of scale 1 on the sub-key scores before the top-k
#   qnoise 0.5       Gaussian noise, std 0.5, on the normalised query
#   snoise 0.3       Gaussian noise on the state each tick, 0.3 of its rms
#   qnorm none       no query normalisation (is the default norm what prevents collapse)
#   qnorm batch      batch norm on the query, as in Lample et al. 2019
#
#   scripts/0e_collapse.sh     rows in out/0e_store.jsonl and out/0e_chase.jsonl
#   .venv/bin/python -m vvm2.table out/0e_store.jsonl
#   .venv/bin/python -m vvm2.table out/0e_store.jsonl --heads
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=".venv/bin/python -m vvm2.train --threads 4"
STORE="$RUN --out out/0e_store.jsonl --task assoc --items 16384 --entries 65536 --ticks 2 --batch 512 --steps 6000"
CHASE="$RUN --out out/0e_chase.jsonl --task chase --items 256 --hops 4 --holdout 0.3 --entries 4096 --ticks 4 --batch 256 --steps 6000 --wd 1"
{
    for seed in 0 1 2; do
        for base in "$STORE" "$CHASE"; do
            echo "$base --seed $seed --label 'none' >/dev/null"
            echo "$base --seed $seed --pick_from 32 --label 'pick_from 32' >/dev/null"
            echo "$base --seed $seed --gumbel 1 --label 'gumbel 1' >/dev/null"
            echo "$base --seed $seed --qnoise 0.5 --label 'qnoise 0.5' >/dev/null"
            echo "$base --seed $seed --snoise 0.3 --label 'snoise 0.3' >/dev/null"
            echo "$base --seed $seed --qnorm none --label 'qnorm none' >/dev/null"
            echo "$base --seed $seed --qnorm batch --label 'qnorm batch' >/dev/null"
        done
    done
} | scripts/par.sh
.venv/bin/python -m vvm2.table out/0e_store.jsonl
.venv/bin/python -m vvm2.table out/0e_chase.jsonl
