#!/usr/bin/env bash
# Minimal end-to-end reproduction of arXiv:2606.00206.
# Installs deps, runs the quantized reasoning model with/without the
# overthinking-marker logit penalty, and writes EVAL.md.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Installing dependencies ==="
pip install -q --upgrade pip
# Pin a vLLM that supports per-request logits_processors and AWQ for Qwen2.
pip install -q "vllm==0.6.3.post1" "transformers>=4.45,<4.46" "datasets>=2.20" \
    "hf_transfer" "autoawq" 2>&1 | tail -5 || \
pip install -q "vllm==0.6.3.post1" "transformers>=4.45,<4.46" "datasets>=2.20" "hf_transfer"

echo "=== GPU info ==="
nvidia-smi || true

echo "=== Running reproduction ==="
# Minimal config: MATH-500 slice, the benchmark where AWQ overthinking is most
# dramatic in the paper. lambda=2.0 is in the paper's effective sweep range.
python repro.py \
    --dataset math500 \
    --n "${N:-50}" \
    --lam "${LAM:-2.0}" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --seed 0

echo "=== Done. EVAL.md written. ==="
cat EVAL.md
