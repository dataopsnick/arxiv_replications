"""Minimal reproduction of arXiv:2606.00206 — "Quantized Reasoning Models Think
They Need to Think Longer, but They Do Not".

Core claim being reproduced
---------------------------
A quantized reasoning model overthinks (long chains of thought, abandoned
correct intermediate answers). A *training-free* logit penalty (Eq. 1: subtract
a fixed lambda from the logits of a curated set of 50 "overthinking marker"
tokens at every decoding step) reduces chain-of-thought length while preserving
or improving accuracy.

This script runs a small reasoning model (DeepSeek-R1-Distill-Qwen-1.5B,
AWQ-quantized) on a slice of MATH-500 and/or GSM8K, twice:
  (1) no penalty (lambda = 0)   -> baseline quantized overthinking
  (2) with penalty (lambda > 0) -> the paper's intervention
and reports accuracy and mean CoT length for each, plus the deltas the paper
predicts (CoT length down, accuracy preserved or up).
"""

import argparse
import json
import os
import re
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from markers import build_marker_token_ids, OVERTHINKING_MARKERS

ART_DIR = os.path.join(os.path.dirname(__file__), ".openresearch", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Answer extraction / verification
# --------------------------------------------------------------------------- #
def extract_boxed(text):
    """Extract the content of the last \\boxed{...} in the text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j].strip()
    return None


def normalize_answer(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace(" ", "")
    s = s.replace("\\text{", "").replace("}", "").replace("{", "")
    s = s.rstrip(".")
    # numeric normalization (drop commas, trailing zeros)
    t = s.replace(",", "")
    try:
        f = float(t)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s.lower()


def extract_final_number(text):
    """Fallback extraction for GSM8K-style answers: last number in the text."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_math500(n):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.select(range(min(n, len(ds))))
    items = []
    for ex in ds:
        items.append({
            "problem": ex["problem"],
            "gold": normalize_answer(ex["answer"]),
            "kind": "math",
        })
    return items


def load_gsm8k(n):
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))
    items = []
    for ex in ds:
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "problem": ex["question"],
            "gold": normalize_answer(gold),
            "kind": "gsm8k",
        })
    return items


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def build_prompts(tokenizer, items):
    prompts = []
    for it in items:
        instr = (
            it["problem"]
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        messages = [{"role": "user", "content": instr}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def make_logits_processor(marker_ids, lam):
    """vLLM logits processor implementing Eq. (1): subtract lambda from the
    logits of every overthinking-marker token at each decoding step."""
    import torch
    ids = torch.tensor(marker_ids, dtype=torch.long)

    def proc(past_token_ids, logits):
        logits[ids] = logits[ids] - lam
        return logits

    return proc


def grade(items, outputs):
    n_correct = 0
    total_len = 0
    per_sample = []
    for it, out in zip(items, outputs):
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        pred = extract_boxed(text)
        if pred is None:
            pred = extract_final_number(text)
        pred_n = normalize_answer(pred)
        correct = (pred_n is not None and pred_n == it["gold"])
        n_correct += int(correct)
        total_len += n_tokens
        per_sample.append({
            "problem": it["problem"][:200],
            "gold": it["gold"],
            "pred": pred_n,
            "correct": correct,
            "cot_tokens": n_tokens,
        })
    acc = 100.0 * n_correct / len(items)
    mean_len = total_len / len(items)
    return acc, mean_len, per_sample


def run_condition(llm, prompts, items, sp, label):
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dt = time.time() - t0
    acc, mean_len, per_sample = grade(items, outputs)
    print(f"[{label}] acc={acc:.1f}%  mean_cot_tokens={mean_len:.0f}  "
          f"({len(items)} q, {dt:.0f}s)")
    return {"label": label, "acc": acc, "mean_cot_tokens": mean_len,
            "per_sample": per_sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/deepseek-r1-distill-qwen-1.5b-awq")
    ap.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--lam", type=float, default=2.0, help="penalty strength lambda")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Config: {vars(args)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    marker_ids = build_marker_token_ids(tokenizer)
    print(f"Resolved {len(marker_ids)} marker token IDs from "
          f"{len(OVERTHINKING_MARKERS)} marker words.")
    with open(os.path.join(ART_DIR, "marker_token_ids.json"), "w") as f:
        json.dump({"marker_words": OVERTHINKING_MARKERS,
                   "marker_token_ids": marker_ids,
                   "decoded": [tokenizer.convert_ids_to_tokens(t) for t in marker_ids]},
                  f, indent=2)

    if args.dataset == "math500":
        items = load_math500(args.n)
    else:
        items = load_gsm8k(args.n)
    print(f"Loaded {len(items)} questions from {args.dataset}.")

    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.9,
              max_model_len=args.max_tokens + 1024, seed=args.seed,
              enforce_eager=False)
    prompts = build_prompts(tokenizer, items)

    base_sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                             max_tokens=args.max_tokens, seed=args.seed)
    pen_sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=args.seed,
                            logits_processors=[make_logits_processor(marker_ids, args.lam)])

    base = run_condition(llm, prompts, items, base_sp, f"no-penalty (lam=0)")
    pen = run_condition(llm, prompts, items, pen_sp, f"penalty (lam={args.lam})")

    d_acc = pen["acc"] - base["acc"]
    d_len_pct = 100.0 * (pen["mean_cot_tokens"] - base["mean_cot_tokens"]) / base["mean_cot_tokens"]

    results = {"config": vars(args), "no_penalty": {k: base[k] for k in ("acc", "mean_cot_tokens")},
               "penalty": {k: pen[k] for k in ("acc", "mean_cot_tokens")},
               "delta_acc": d_acc, "delta_len_pct": d_len_pct}
    with open(os.path.join(ART_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(ART_DIR, "per_sample.jsonl"), "w") as f:
        for tag, cond in (("no_penalty", base), ("penalty", pen)):
            for s in cond["per_sample"]:
                f.write(json.dumps({"condition": tag, **s}) + "\n")

    # ----- EVAL.md -----
    claim_supported = (d_len_pct < -2.0) and (d_acc >= -2.0)
    verdict = "SUPPORTED" if claim_supported else "NOT CLEARLY SUPPORTED"
    md = f"""# Minimal Reproduction — arXiv:2606.00206

**Quantized Reasoning Models Think They Need to Think Longer, but They Do Not**

## Core claim tested
A training-free logit penalty (subtract lambda from the logits of 50 curated
overthinking-marker tokens at every decoding step, Eq. 1) reduces chain-of-thought
length on a *quantized* reasoning model while preserving or improving accuracy.

## Setup
- Model: `{args.model}` (AWQ-quantized DeepSeek-R1-Distill-Qwen-1.5B)
- Benchmark: {args.dataset.upper()} ({len(items)} questions)
- Decoding: T={args.temperature}, top-p={args.top_p}, max_tokens={args.max_tokens}, seed={args.seed}
- Penalty strength lambda = {args.lam}
- Marker token IDs resolved: {len(marker_ids)} (from {len(OVERTHINKING_MARKERS)} marker words)

## Results

| Condition | Accuracy (%) | Mean CoT length (tokens) |
|---|---|---|
| No penalty (lambda=0) | {base['acc']:.1f} | {base['mean_cot_tokens']:.0f} |
| Penalty (lambda={args.lam}) | {pen['acc']:.1f} | {pen['mean_cot_tokens']:.0f} |
| **Delta** | **{d_acc:+.1f}** | **{d_len_pct:+.1f}%** |

## Verdict: {verdict}

The paper predicts the penalty reduces CoT length by ~12-23% while preserving or
improving accuracy. Here the penalty changed CoT length by **{d_len_pct:+.1f}%**
and accuracy by **{d_acc:+.1f} points**.

A reduction in CoT length (negative delta) with non-degraded accuracy reproduces
the paper's central mechanism on this minimal configuration.

Artifacts: `results.json`, `per_sample.jsonl`, `marker_token_ids.json`.
"""
    with open("EVAL.md", "w") as f:
        f.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
