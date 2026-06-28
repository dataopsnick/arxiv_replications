# Minimal reproduction — arXiv:2606.00206

**"Quantized Reasoning Models Think They Need to Think Longer, but They Do Not"**
(Lotfi, Kirichenko, Li, Liu — FAIR at Meta / Meta AI, June 2026)
Paper: https://arxiv.org/abs/2606.00206

This repo is a small, self-contained, end-to-end reproduction of the paper's
**central claim and mechanism**:

> A quantized reasoning model overthinks — it produces long chains of thought and
> abandons correct intermediate answers. A **training-free logit penalty** that
> subtracts a fixed amount λ from the logits of 50 curated "overthinking marker"
> tokens (e.g. *wait, but, alternatively, maybe, however …*) at every decoding
> step (Eq. 1 of the paper) **reduces chain-of-thought length while preserving or
> improving accuracy.**

## Result (this reproduction)

Model `casperhansen/deepseek-r1-distill-qwen-1.5b-awq` (AWQ-quantized
DeepSeek-R1-Distill-Qwen-1.5B), 50 MATH-500 questions, T=0.6, top-p=0.95,
max_tokens=8192, seed=0, penalty λ=2.0:

| Condition | Accuracy (%) | Mean CoT length (tokens) |
|---|---|---|
| No penalty (λ=0) | 78.0 | 3474 |
| Penalty (λ=2.0) | 76.0 | **2195** |
| **Delta** | **−2.0** | **−36.8%** |

**Verdict: SUPPORTED.** The penalty cut chain-of-thought length by **36.8%**
while accuracy stayed essentially flat (−2.0 pts = one question out of 50). This
reproduces the paper's predicted direction (CoT down 12–23%, accuracy
preserved/improved). Full numbers in [`EVAL.md`](EVAL.md) and
[`results/results.json`](results/results.json).

## How to replicate

### Requirements
- 1× NVIDIA GPU (a single H100 is plenty; the AWQ 1.5B model needs ~2 GB VRAM,
  generation is the cost). CUDA-capable.
- Python 3.10–3.11.
- Internet access (downloads the model and MATH-500 from the Hugging Face Hub).

### One command

```bash
bash run.sh
```

`run.sh` installs the pinned dependencies, runs both conditions (no-penalty vs.
penalty) on the quantized model, and writes `EVAL.md` plus the artifacts under
`.openresearch/artifacts/`.

The original run completed in **~7 minutes** on one H100 (≈90s of generation per
condition + model download/compile).

### Knobs (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `N` | `50` | number of MATH-500 questions |
| `LAM` | `2.0` | penalty strength λ (paper sweeps 0.5–4.0) |
| `MAX_TOKENS` | `8192` | max generation length per question |

Example — stronger penalty on more questions:

```bash
N=100 LAM=3.0 bash run.sh
```

### Manual install (if not using `run.sh`)

```bash
pip install "vllm==0.6.3.post1" "transformers>=4.45,<4.46" "datasets>=2.20" hf_transfer autoawq
python repro.py --dataset math500 --n 50 --lam 2.0 --max-tokens 8192 --seed 0
```

`repro.py` also supports `--dataset gsm8k` and `--model <hf-id>` (any
AWQ/GPTQ-quantized Qwen2-family reasoning model that vLLM can load).

## What's in here

| File | Purpose |
|---|---|
| `run.sh` | End-to-end entry point: installs deps, runs, writes `EVAL.md`. |
| `repro.py` | Loads the quantized model (vLLM), runs both conditions, grades, writes results. |
| `markers.py` | The 50 overthinking markers from **Table 8** of the paper, plus a tokenizer-agnostic resolver to vocab token IDs. |
| `EVAL.md` | The reproduction's verdict and result table (committed copy of the run output). |
| `results/results.json` | Exact metrics produced by the run. |
| `results/marker_token_ids.json` | The resolved marker token IDs (147 IDs from 50 marker words). |

## Method details

**The penalty (Eq. 1).** At every decoding step the logit of each token `v` in
the overthinking-marker set `S` is replaced by `logit(v) − λ`; all other tokens
are untouched. Implemented as a per-request vLLM logits processor in
`repro.py` (`make_logits_processor`). It is training-free, adds no forward
passes, and has a single hyperparameter λ.

**The marker set `S`.** The 50 words from the paper's Table 8 (each a leading-
space tokenizer token: *perhaps, maybe, wait, Wait, actually, hold, Hmm, …,
mistake, error, incorrect*). Because exact token IDs are tokenizer-specific,
`markers.py` resolves each marker word to **all** vocabulary token IDs whose
decoded surface form matches it (covering both the leading-space and bare
variants), reproducing the paper's intent in a tokenizer-agnostic way — 147 IDs
for this checkpoint's tokenizer.

**Grading.** Answers are extracted from the last `\boxed{...}` (with a
last-number fallback) and compared to the gold answer after LaTeX/numeric
normalization (`grade` / `normalize_answer` in `repro.py`).

## Notes, scope, and faithfulness

- **This is the *minimal* proof-of-concept**, not the full paper. The paper spans
  5 models (1.5B–32B), 3 quantization methods (AWQ/GPTQ/FlatQuant), 6 bit-widths,
  and 5 benchmarks (AIME, MATH-500, GSM8K, GPQA-Diamond, LiveCodeBench), plus a
  KL-divergence token analysis and a GPT-5-judged error taxonomy. Here we isolate
  the single core mechanism: penalty vs. no-penalty on one quantized model and one
  benchmark slice.
- **4-bit vs. the paper's flagship 3-bit.** This uses a reliably loadable 4-bit
  AWQ checkpoint. At 4-bit the no-penalty baseline overthinks far less than the
  paper's dramatic 3-bit AWQ MATH-500 number (47% acc / 23.4k tokens); the paper
  itself reports a milder −14.8% CoT reduction at 4-bit AWQ (Table 3). We observe
  an even larger reduction (−36.8%) with accuracy preserved, so the mechanism is
  well supported. To see the dramatic 3-bit baseline, swap in a 3-bit AWQ/GPTQ
  checkpoint via `--model`.
- **Decoding** matches the paper: T=0.6, top-p=0.95.

## Suggested extensions

- 3-bit AWQ/GPTQ checkpoint to surface the dramatic overthinking baseline.
- λ sweep 0.5→4.0 to reproduce the Pareto frontier of Figure 5.
- Control token lists (random / low-KL / high-KL, Tables 9–11) as the ablation
  showing overthinking markers give the best accuracy-vs-cost tradeoff.
