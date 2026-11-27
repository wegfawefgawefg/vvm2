#!/usr/bin/env bash
# The memory-mapped vvm (vvm2/mm.py): input and output are their own domains, the state is a
# scratchpad, vops are flat, 32 fetched a tick.
#   scripts/mm.sh class [mnist|kmnist|fashion]     classify
#   scripts/mm.sh draw  [mnist|kmnist|fashion]     generate images -> docs/img/drawn-TASK.png
set -euo pipefail
cd "$(dirname "$0")/.."
t="${2:-mnist}"
[ -d "data/$t" ] || scripts/get_data.sh "$t"
nice -n 19 .venv/bin/python -m old.mm "${1:-class}" "$t"
