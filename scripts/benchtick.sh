#!/usr/bin/env bash
# How fast is one tick on this CPU, with a small vop (one store a tick) and with a wide vop
# (the store, the next pc and both next addresses in one tick)? bench/tick.cpp, one core.
#   scripts/benchtick.sh            banks of 1 MB, 64 MB and 2 GB, one run and 16 runs
#   scripts/benchtick.sh 256 4      a 256 MB bank, 4 runs sharing it
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
g++ -O3 -march=native -funroll-loops -o build/tick bench/tick.cpp
if [ $# -gt 0 ]; then
    nice -n 5 build/tick "$1" "${2:-1}" $((4000000 / ${2:-1}))
else
    for mb in 1 64 2048; do for runs in 1 16; do nice -n 5 build/tick $mb $runs $((4000000 / runs)); done; done
fi
