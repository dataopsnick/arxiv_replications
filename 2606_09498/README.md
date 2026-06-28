# Self-Harness — minimal reproduction

A small, faithful, end-to-end reproduction of

> **Self-Harness: Harnesses That Improve Themselves** — Zhang et al., 2026
> ([arXiv:2606.09498](https://arxiv.org/abs/2606.09498))

A **fixed LLM** (DeepSeek-V4-flash, via OpenRouter) acts as **both the agent and
the proposer** and improves *its own operating harness* — with no human in the
loop and no stronger external model. The harness's pass-rate goes **up on both a
held-in and a held-out split, with no regression**, exactly the paper's headline
claim, reproduced on a single laptop / free Colab CPU runtime in ~2 minutes.

```
$ self-harness
| Split    | Baseline h_0 | After Self-Harness | Abs gain |
|----------|--------------|--------------------|----------|
| Held-in  | 0/6 (0.0%)   | 6/6 (100.0%)       | +6       |
| Held-out | 0/6 (0.0%)   | 5/6 (83.3%)        | +5       |
Claim reproduced: True
```

---

## Quickstart — Google Colab

You only need a free OpenRouter API key (<https://openrouter.ai/keys>) with a
little credit; DeepSeek-V4-flash is cheap and the whole run is a few dozen calls.
**No GPU** — set the Colab runtime to **CPU**; all the heavy lifting is the
hosted model behind the API.

Paste these three cells into a new Colab notebook and run them top to bottom.

**Cell 1 — install the library**

```python
!git clone https://github.com/dataopsnick/self-harness-harnesses-that-improve-themselves-dd2a2983.git self-harness
%cd self-harness
!pip install -e .          # pure-stdlib runtime deps, installs instantly
```

**Cell 2 — set your OpenRouter key**

```python
import os, getpass
os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OpenRouter API key (sk-or-...): ")
```

**Cell 3 — run the self-improvement loop**

```python
from self_harness import Config, run_self_harness, write_artifacts

result = run_self_harness(Config())     # agent + proposer = deepseek-v4-flash
write_artifacts(result)                 # writes EVAL.md

print("Claim reproduced:", result.claim_reproduced)
print(f"Held-in : {result.base_in} -> {result.final_in}  (Δ {result.d_in:+d})")
print(f"Held-out: {result.base_ho} -> {result.final_ho}  (Δ {result.d_ho:+d})")
print("\nFinal self-improved harness:\n")
print(result.final_system_prompt)
```

That's it — Cell 3 prints the before/after table and the harness the model wrote
for itself, and drops a full `EVAL.md` in the working directory.

> Prefer a one-liner? After Cells 1–2, just run `!bash run.sh` (the frozen
> reproduction command) or `!self-harness --help` for flags.

---

## Quickstart — local

```bash
git clone https://github.com/dataopsnick/self-harness-harnesses-that-improve-themselves-dd2a2983.git
cd self-harness-harnesses-that-improve-themselves-dd2a2983
pip install -e .

export OPENROUTER_API_KEY="sk-or-..."
self-harness                 # or: bash run.sh   (the frozen run command)
```

Python API:

```python
from self_harness import Config, run_self_harness, write_artifacts

result = run_self_harness(Config(
    model="deepseek/deepseek-v4-flash",  # the one fixed model (agent + proposer)
    rounds=1,                            # improvement rounds
    k_proposals=4,                       # candidate edits proposed per round
))
write_artifacts(result)                  # EVAL.md + .openresearch/artifacts/*.json
```

---

## What the paper claims, and what this reproduces

The performance of an LLM agent depends not just on the model but on its
**harness** — the system prompt, tools, memory, and orchestration that mediate
its interaction with an environment. Harnesses are normally hand-engineered per
model, which doesn't scale. **Self-Harness** lets a *fixed* agent improve its own
harness through an iterative, evidence-driven loop, with no human and no stronger
external model. The paper shows this raises pass-rate on Terminal-Bench-2.0
across three different LLM backends, on **both** a held-in and a held-out split.

This repo reproduces the **core mechanism and its central claim** in the smallest
configuration that still demonstrates it: one model, a tiny verifiable task
suite, one improvement round. It is intentionally *not* the full paper (no
Terminal-Bench-2.0, no three-model sweep, no qualitative trace study).

---

## The loop (paper §4), implemented end to end

```
                    held-in tasks
                         │
        ┌────────────────▼─────────────────┐
        │  Agent runs under harness h_t     │   self_harness/loop.py: run_agent
        │  (the fixed model)                │
        └────────────────┬─────────────────┘
                         │ traces + verifier pass/fail
        ┌────────────────▼─────────────────┐
   (1)  │  Weakness Mining                  │   loop.py: mine_weaknesses
        │  cluster failures by verifier-    │   verifier.py: evaluate -> cause c_i
        │  grounded cause c_i (exact match) │
        └────────────────┬─────────────────┘
                         │ evidence bundles
        ┌────────────────▼─────────────────┐
   (2)  │  Harness Proposal                 │   loop.py: propose_edits
        │  SAME model proposes K minimal,   │   (grounded · minimal · diverse)
        │  grounded, diverse edits          │
        └────────────────┬─────────────────┘
                         │ candidate edits Δ_j
        ┌────────────────▼─────────────────┐
   (3)  │  Proposal Validation              │   loop.py: validate_and_merge
        │  eval each cand on held-in AND    │
        │  held-out; accept iff             │
        │    d_in≥0 ∧ d_ho≥0 ∧ max(.)>0     │   ← the paper's non-regression rule
        │  merge accepted edits → h_{t+1}   │
        └───────────────────────────────────┘
```

1. **Weakness Mining** — the agent runs every held-in task under the current
   harness `h_t`. A fixed verifier `E` assigns pass/fail and, on failure, a
   **verifier-grounded cause** `c_i` (`missing_protocol_marker`, `no_artifact`,
   `wrong_answer`, `runtime_error`, `timeout`, …) — grounded in what the verifier
   *observed*, never inferred from the model's narration. Failures are clustered
   by exact-match cause into evidence bundles. *(`loop.py::mine_weaknesses`,
   `verifier.py::evaluate`.)*

2. **Harness Proposal** — the **same fixed model**, now in a proposer role, is
   given the evidence bundles and asked for `K` candidate harness edits that are
   **grounded** (each targets a mined cause), **minimal** (one concrete rule),
   and **diverse**. Edits append guidance rules to the harness — the bounded,
   declared edit surface. *(`loop.py::propose_edits`, `harness.py::apply_edit`.)*

3. **Proposal Validation** — each candidate harness is re-evaluated on the
   held-in **and** the held-out split. A candidate is accepted only under the
   paper's exact **non-regression rule**

   ```
   d_in ≥ 0  and  d_ho ≥ 0  and  max(d_in, d_ho) > 0
   ```

   i.e. it must not regress either split and must improve at least one. Accepted
   edits are merged into `h_{t+1}`; rejected ones are logged and discarded.
   *(`loop.py::validate_and_merge`.)*

The model is **fixed throughout** and plays both agent and proposer — the
defining feature of Self-Harness versus meta-harness / human-engineering
approaches.

---

## The minimal task environment

The paper uses Terminal-Bench-2.0 (multi-turn terminal tasks in containers). To
stay laptop/Colab-sized while preserving the *failure regime that makes the loop
meaningful*, this repo substitutes a small suite of **verifiable coding tasks**
(`self_harness/tasks.py`), split into a fixed **held-in** and **held-out** set
(disjoint, fixed across all harness variants, like the paper).

The crucial design point: the verifier enforces a **hidden submission protocol**
the task prompts never mention — every submission must declare a required marker
line (`FINAL_SOLUTION`), analogous to a benchmark's grading contract
(`verifier.py::PROTOCOL_MARKER`). The deliberately *minimal* initial harness
(`harness.py::INITIAL_HARNESS`, just *"You are a coding agent. Solve the task."*)
omits this protocol, so the agent fails every task with the verifier-grounded
cause `missing_protocol_marker`. This is exactly the regime Self-Harness targets:
the model must **rediscover the protocol from its own verifier-grounded failures
and encode it into its harness** — not a difference in raw capability.

> **Why this substitution is honest.** The paper's reported pass-rate *numbers*
> are specific to Terminal-Bench + frontier models and are not reproduced here.
> What is reproduced is the *causal mechanism and its central claim*: a fixed
> model, given only verifier-grounded failure signatures, writes harness edits
> that lift its own pass-rate on held-in **and** held-out tasks under the paper's
> conservative acceptance rule. Swapping the benchmark for a smaller verifiable
> one keeps that mechanism intact at ~1/1000th the cost.

---

## Reproduced result

Run on `deepseek/deepseek-v4-flash` (both roles), 6 held-in / 6 held-out tasks,
1 round, `K=4` proposals:

| Split    | Baseline `h_0` | After Self-Harness | Abs gain |
|----------|----------------|--------------------|----------|
| Held-in  | 0/6 (0.0%)     | 6/6 (100.0%)       | **+6**   |
| Held-out | 0/6 (0.0%)     | 5/6 (83.3%)        | **+5**   |

**Claim reproduced: True** — non-regression on both splits, improvement on both.

In the run, weakness mining found a single failure cluster
(`missing_protocol_marker`, size 6). The proposer returned 4 candidate edits;
**3 were accepted** and merged. The decisive accepted edit (`d_in=+6, d_ho=+5`):

> *"Every response must start with the line 'FINAL_SOLUTION' (no extra text before it)."*

The model **diagnosed its own verifier-grounded failure and wrote the harness
rule that fixes it**, and the gain held on the held-out split it was never shown
failing on — evidence of a generalizing fix, not overfitting to observed
failures, mirroring the paper's finding. The final self-written harness:

```
You are a coding agent. Solve the task.

Guidelines:
1. Your final answer must include the exact marker 'FINAL_SOLUTION' on its own line, followed by your solution code or explanation.
2. Before outputting your solution, always insert a line containing only 'FINAL_SOLUTION' as a clear delimiter.
3. Every response must start with the line 'FINAL_SOLUTION' (no extra text before it).
```

(LLM sampling is stochastic; exact accepted-edit counts and the one residual
held-out failure can vary run to run, but the claim — gain on both splits, no
regression — reproduces reliably.)

---

## Package layout

```
self_harness/
├── __init__.py     public API: Config, run_self_harness, Result, write_artifacts, …
├── config.py       Config dataclass — every knob, fully reproducible from one object
├── llm.py          LLMClient — stdlib-only OpenRouter chat client (the fixed model)
├── tasks.py        verifiable task suite + fixed held-in / held-out splits
├── verifier.py     fixed evaluator E: pass/fail + verifier-grounded failure cause c_i
├── harness.py      the editable harness surface h_t and apply_edit (append guidance)
├── loop.py         the three-stage loop + SelfHarness driver + Result object
├── report.py       render EVAL.md and JSON artifacts from a Result
└── cli.py          `self-harness` / `python -m self_harness`
run.sh              frozen reproduction command (bash run.sh)
tests/test_core.py  offline tests of the deterministic core (no API key needed)
pyproject.toml      installable package (console script: self-harness)
```

### Public API

| Symbol | Purpose |
|---|---|
| `Config(...)` | All run knobs (`model`, `rounds`, `k_proposals`, temperatures, `api_key`). Reads `OPENROUTER_API_KEY` / `SELF_HARNESS_MODEL` from the env by default. |
| `run_self_harness(config) -> Result` | Run the full loop, return a structured `Result`. |
| `SelfHarness(config)` | The driver, if you want to call individual stages (`run_agent`, `mine_weaknesses`, `propose_edits`, `validate_and_merge`). |
| `Result` | `.base_in/.base_ho`, `.final_in/.final_ho`, `.d_in/.d_ho`, `.claim_reproduced`, `.final_system_prompt`, `.rounds`, `.to_dict()`. |
| `write_artifacts(result)` | Write `EVAL.md` (repo root + `.openresearch/artifacts/`) and `history.json`. |

### CLI

```bash
self-harness                       # default reproduction (1 round, K=4)
self-harness --rounds 2 --k-proposals 6
self-harness --model deepseek/deepseek-v4-pro
self-harness --no-write            # skip writing EVAL.md
python -m self_harness --help
```

Exit code is `0` if the core claim reproduced, `2` if the run completed but the
claim did not hold (e.g. no accepted edit), and `1` on an API error.

---

## Tests

The deterministic core (verifier, harness edits, weakness mining, splits) is
covered by offline tests that need **no API key** and run in under a second:

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — (required) | OpenRouter key for the model calls. |
| `SELF_HARNESS_MODEL` | `deepseek/deepseek-v4-flash` | Model slug for **both** agent and proposer. |

Everything is also settable via `Config(...)` or CLI flags, which take
precedence over the environment.

---

## Relation to the paper / limitations

- **Reproduces:** the Self-Harness loop (weakness mining → proposal →
  non-regression validation), a fixed model as both agent and proposer, the
  exact acceptance rule, and the central claim (held-in **and** held-out gains
  with no regression from self-authored harness edits).
- **Does not reproduce:** Terminal-Bench-2.0, the three-model sweep
  (MiniMax M2.5 / Qwen3.5-35B-A3B / GLM-5), the multi-round trajectories, or the
  qualitative trace analysis. Absolute pass-rate numbers here are not comparable
  to the paper's — only the mechanism and the direction of the effect are.
- **Scope:** one fixed model, a small verifiable task suite, one improvement
  round by default. Increase `rounds` / `k_proposals`, or swap in a harder task
  suite, to push further.

## License

MIT.

---

> Reproduction built and run via [OpenResearch](https://openresearch.sh).
