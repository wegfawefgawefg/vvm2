#!/usr/bin/env bash
# Make .venv with CPU torch and numpy (nothing system-wide). Run once.
#   scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
    .venv/bin/pip install --quiet numpy matplotlib
fi
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
