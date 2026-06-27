#!/usr/bin/env bash
# Minimal DYSCO reproduction (arXiv:2606.13260).
# Installs the `dysco` library and runs the canonical single-view (V=1, ~DYNCL)
# vs multi-view (V=4,8 — DYSCO) comparison of latent Lorenz-dynamics recovery
# from noisy nonlinear MLP observations. Writes EVAL.md.
set -euo pipefail

PY=$(command -v python3 || command -v python)
"$PY" -m pip install -q -e .

"$PY" -m dysco \
  --T 5000 --obs_dim 32 --noise 1.5 \
  --steps 6000 --batch 512 --lr 2e-3 --kappa 4 \
  --views 1,4,8 --seed 0

echo "=== EVAL.md ==="
cat EVAL.md
