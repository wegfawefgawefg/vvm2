#!/usr/bin/env bash
# Build and run the C++ machine on the copy task (cpp/copy.cpp).
#   scripts/evolve.sh hand | speed | evolve [GENS] [POP] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
g++ -O3 -march=native -fopenmp -std=c++17 -o build/copy cpp/copy.cpp
OMP_NUM_THREADS="${THREADS:-12}" nice -n 19 build/copy "$@"
