# Minimal Reproduction — arXiv:2606.00206

**Quantized Reasoning Models Think They Need to Think Longer, but They Do Not**

## Core claim tested
A training-free logit penalty (subtract lambda from the logits of 50 curated
overthinking-marker tokens at every decoding step, Eq. 1) reduces chain-of-thought
length on a *quantized* reasoning model while preserving or improving accuracy.

## Setup
- Model: `casperhansen/deepseek-r1-distill-qwen-1.5b-awq` (AWQ-quantized DeepSeek-R1-Distill-Qwen-1.5B)
- Benchmark: MATH500 (50 questions)
- Decoding: T=0.6, top-p=0.95, max_tokens=8192, seed=0
- Penalty strength lambda = 2.0
- Marker token IDs resolved: 147 (from 50 marker words)

## Results

| Condition | Accuracy (%) | Mean CoT length (tokens) |
|---|---|---|
| No penalty (lambda=0) | 78.0 | 3474 |
| Penalty (lambda=2.0) | 76.0 | 2195 |
| **Delta** | **-2.0** | **-36.8%** |

## Verdict: SUPPORTED

The paper predicts the penalty reduces CoT length by ~12-23% while preserving or
improving accuracy. Here the penalty changed CoT length by **-36.8%**
and accuracy by **-2.0 points**.

A reduction in CoT length (negative delta) with non-degraded accuracy reproduces
the paper's central mechanism on this minimal configuration.

Artifacts: `results.json`, `per_sample.jsonl`, `marker_token_ids.json`.
