#!/usr/bin/env bash
# Smoke test: the checks, then every task with every model, tiny, end to end, then the
# table printer. Says nothing about quality; it only has to run.
#   scripts/smoke.sh           rows in out/smoke/runs.jsonl (remade each time)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=out/smoke/runs.jsonl
mkdir -p out/smoke
: > "$OUT"
nice -n 19 .venv/bin/python -m vvm2.check
RUN=".venv/bin/python -m vvm2.train --threads 2 --steps 60 --evals 2 --entries 256 --out $OUT"
{
    for task in "assoc --items 256" "modadd --p 11" "parity --bits 6" \
                "chase --items 16 --hops 3" "kvrecall --pairs 2"; do
        name=${task%% *}
        echo "$RUN --task $task --ticks 3 --label '$name machine' >/dev/null"
        echo "$RUN --task $task --feed seq --ticks 1 --regs 2 --mem 1 --label '$name machine seq+mem' >/dev/null"
        for model in mlp gru transformer; do
            echo "$RUN --task $task --model $model --match params --label '$name $model params' >/dev/null"
            echo "$RUN --task $task --model $model --match madds --label '$name $model madds' >/dev/null"
        done
    done
    # every noise switch at once, the parts switched off, the other options
    echo "$RUN --task assoc --items 256 --pick_from 16 --gumbel 1 --qnoise 0.3 --snoise 0.1 --label 'assoc machine noisy' >/dev/null"
    echo "$RUN --task assoc --items 256 --no_bank 1 --label 'assoc machine no bank' >/dev/null"
    echo "$RUN --task assoc --items 256 --no_alu 1 --hold 1 --qnorm batch --label 'assoc machine no alu' >/dev/null"
    echo "$RUN --task chase --items 16 --hops 3 --model gru --ponder 3 --match params --label 'chase gru ponder' >/dev/null"
    echo "$RUN --task chase --items 16 --hops 3 --ticks 3 --tick_loss 1 --vdecay 1 --label 'chase machine tick loss' >/dev/null"
} | scripts/par.sh
.venv/bin/python -m vvm2.table "$OUT"
.venv/bin/python -m vvm2.table "$OUT" --groups
.venv/bin/python -m vvm2.table "$OUT" --heads
.venv/bin/python -m vvm2.table "$OUT" --lens "chase machine"
rows=$(wc -l < "$OUT")
[ "$rows" -eq 45 ] || { echo "smoke: expected 45 rows, got $rows"; exit 1; }
echo "smoke ok: $rows runs"
