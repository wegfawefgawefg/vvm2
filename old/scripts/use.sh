#!/usr/bin/env bash
# How unevenly the vops are used, busiest to least, beside a Zipf line: docs/img/use.png
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/mnist ] || scripts/get_mnist.sh
nice -n 19 .venv/bin/python -m old.use
