#!/usr/bin/env bash
# Reproduce the project-local venv used for SwitchCost. Run from the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "venv ready at .venv — activate with: source .venv/bin/activate"
