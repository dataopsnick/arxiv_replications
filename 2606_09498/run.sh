#!/usr/bin/env bash
# Minimal Self-Harness reproduction (arXiv 2606.09498): runs the full
# self-improvement loop end to end (agent + proposer = deepseek-v4-flash via
# OpenRouter) and writes EVAL.md. This is the frozen run-command contract.
set -euo pipefail

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set" >&2
  exit 1
fi

python3 --version
# Run the packaged loop as a module (no install needed; repo root on sys.path).
python3 -m self_harness "$@"
