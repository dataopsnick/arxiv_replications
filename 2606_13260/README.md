# DYSCO — minimal reproduction

A small, installable Python library that reproduces the core claim of:

> **Extracting Governing Equations from Latent Dynamics via Multi-View Contrastive Learning**
> Paolo Muratore & Mackenzie Weygandt Mathis (EPFL), arXiv:[2606.13260](https://arxiv.org/abs/2606.13260)

**Core claim reproduced:** multi-view temporal contrastive learning recovers
latent trajectories *and* their governing equations from noisy, high-dimensional,
nonlinear observations — where single-view contrastive identification (DYNCL)
fails. Independent views of the same latent process provide a denoising signal
that single-view methods cannot exploit.

> The paper has no public reference implementation; this is a from-scratch
> minimal implementation that keeps only the mechanism needed to illustrate the
> central claim (DYSCO score of Eq. 6 trained with an InfoNCE objective).

---

## Experimental results

Setup: a chaotic **3-D Lorenz** latent system, observed through a fixed random
**4-layer MLP mixing** function into `D = 32` dimensions, corrupted by additive
Gaussian observation noise (`σ = 1.5`). Each "view" is an independent noise
realisation of the same signal. The encoder is a 4-layer MLP; the dynamics are a
degree-2 polynomial library `f̂(x) = Θ·Ξ(x)`; training is multi-view temporal
InfoNCE with RK4 roll-outs (`κ = 4`, 6000 base steps). Metrics are R² after the
optimal affine alignment the paper's identifiability theorem requires:

- **latent R²** — recovery of the latent trajectory,
- **dyn R²** — recovery of the governing flow field.

We compare **single-view (`V = 1`, the DYNCL setting)** against **multi-view
(`V = 4, 8` — DYSCO)**.

| Views (V) | latent R² | dyn R² |
|-----------|-----------|--------|
| 1 (single-view ≈ DYNCL) | 87.8% | 54.1% |
| **4 (multi-view, DYSCO)** | **93.8%** | **68.6%** |
| 8 (multi-view, DYSCO) | 89.8% | 55.4% |

**Verdict: REPRODUCED.** Going from single-view to the best multi-view
configuration improves latent-trajectory recovery by **+6.0 points** and
governing-dynamics recovery by **+14.5 points** — the multi-view denoising effect
the paper relies on (its Table 2 result).

These numbers are from the reproduction run on CPU (~2–4 min). They are
reproducible with `bash run.sh` (or `dysco --noise 1.5 --views 1,4,8`).

### Noise sweep

The multi-view advantage is largest in the moderate-to-high noise regime, where
single-view recovery degrades:

| σ (obs noise) | dyn R² (V=1) | dyn R² (best multi-view) | gain |
|---------------|--------------|--------------------------|------|
| 1.0 | 68.2% | 67.4% | ≈ 0 |
| **1.5** | 54.1% | **68.6%** | **+14.5** |
| 2.0 | 57.2% | 61.9% | +4.7 |

**Caveat / known limitation.** At a fixed training budget, `V = 8` underfits and
lands *below* `V = 4` (see the table above), so the paper's *monotonic-in-views*
trend (its Fig. 3a) is **not** cleanly reproduced here. The central
multi-view-vs-single-view claim is. `run.sh` scales the step budget with the
number of views (`DyscoConfig.fair_budget`) to partly mitigate this.

---

## Installation

```bash
pip install -e .          # installs the `dysco` package + the `dysco` CLI
# or just the deps:
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, NumPy. CPU is sufficient (the models are
small MLPs); CUDA is used automatically if available.

---

## Usage

### As a library

```python
from dysco import DyscoConfig, run_experiment

cfg = DyscoConfig(T=5000, obs_dim=32, noise=1.5, steps=6000)
exp = run_experiment(cfg, views=[1, 4, 8])     # trains one model per view count

print(exp.verdict.label)                       # "REPRODUCED"
for r in exp.results:
    print(r.V, r.latentR2, r.dynR2)
print(exp.to_markdown())                        # full EVAL.md report
exp.to_json("results.json")                     # structured results
```

Finer-grained building blocks are all importable:

```python
from dysco import (
    DyscoConfig, generate, train_views,         # data + single-config training
    Encoder, SymbolicDynamics, RandomMixing,    # models
    simulate_lorenz, poly_library, rollout,     # dynamics + library
    evaluate, affine_fit, r2_score,             # metrics
)

cfg = DyscoConfig(noise=1.5)
data = generate(cfg, V=4)                       # (V, T, D) noisy multi-view obs
result = train_views(cfg, V=4, data=data)       # one trained DYSCO model
print(result.latentR2, result.dynR2)
print(result.encoder, result.dynamics)          # the trained nn.Modules
```

### As a CLI

```bash
# canonical reproduction (writes EVAL.md + results.json):
dysco --noise 1.5 --views 1,4,8 --steps 6000

# or without installing:
python -m dysco --noise 1.5 --views 1,4,8

# end-to-end script (install + run + print EVAL.md):
bash run.sh
```

Key flags: `--noise` (σ), `--views` (comma-separated counts), `--steps`,
`--obs_dim`, `--kappa`, `--T`, `--seed`, `--no-fair-budget`, `--out`, `--json`.
Run `dysco --help` for the full list.

---

## Package layout

```
dysco/
├── config.py       DyscoConfig — all hyperparameters (dataclass)
├── systems.py      ground-truth dynamical systems (Lorenz) + simulators
├── models.py       RandomMixing (g), Encoder (h), SymbolicDynamics (f̂), RK4 rollout
├── data.py         multi-view noisy observation generation
├── train.py        the multi-view temporal InfoNCE training loop (train_views)
├── evaluate.py     affine alignment + latent/dyn R² metrics
├── experiment.py   run_experiment, verdict logic, EVAL.md/JSON rendering
└── cli.py          command-line interface (`dysco` / `python -m dysco`)
```

## Method in one paragraph

A latent state `xₜ ∈ ℝ³` follows Lorenz dynamics and is observed as
`yᵃₜ = g(xₜ) + ξᵃₜ` through a fixed nonlinear mixing `g` with independent
per-view noise `ξᵃₜ`. We learn an encoder `h ≈ g⁻¹` and a symbolic dynamics model
`f̂ = Θ·Ξ`. Training samples an anchor `(view a, time t)` and a positive
`(view b ≠ a, time t+k)`, rolls the encoded anchor forward `k` steps with `f̂`,
and matches it to the encoded positive via InfoNCE (negative-squared-distance
similarity, in-batch negatives). With multiple views the positive carries
*independent* noise, so the only way to match is to discard per-view noise and
keep the shared signal `g(xₜ)` — recovering, up to an affine map, both the latent
states and the dynamics.

## License

MIT.
