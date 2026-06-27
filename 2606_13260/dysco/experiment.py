"""High-level experiment orchestration: the multi-view vs single-view sweep.

This is the main entry point most users want. :func:`run_experiment` trains a
DYSCO model for each requested view count, compares single-view (``V == 1``, the
DYNCL setting) against the best multi-view configuration, and returns a structured
:class:`ExperimentResult` with a reproduction verdict. Helpers serialise it to
JSON and to an ``EVAL.md`` report.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import DyscoConfig
from .train import ViewResult, train_views


@dataclass
class Verdict:
    """The reproduction verdict for an experiment.

    Attributes
    ----------
    label:
        One of ``REPRODUCED``, ``REPRODUCED (latent)``, ``PARTIAL``,
        ``NOT REPRODUCED`` or ``INCONCLUSIVE``.
    latent_gain, dyn_gain:
        Improvement (in absolute R^2) of the best multi-view config over the
        single-view baseline, for the latent and dynamics metrics.
    best_latent_V, best_dyn_V:
        The view counts achieving the best latent / dynamics R^2.
    """

    label: str
    latent_gain: float
    dyn_gain: float
    best_latent_V: Optional[int]
    best_dyn_V: Optional[int]


def decide_verdict(results: List[ViewResult]) -> Verdict:
    """Decide whether the core claim is reproduced.

    The paper's claim is *multi-view > single-view*; the monotonic-in-V curve
    (Fig. 3a) is a secondary ablation. So single-view (``V == 1``) is judged
    against the *best* multi-view config on each metric. A strong improvement in
    the dynamics R^2 is the strongest form of the claim; a strong improvement in
    the latent R^2 (without regressing dynamics) is the denoising mechanism.
    """
    by_v: Dict[int, ViewResult] = {r.V: r for r in results}
    multi = [r.V for r in results if r.V > 1]
    if 1 not in by_v or not multi:
        return Verdict("INCONCLUSIVE", 0.0, 0.0, None, None)

    best_dyn_V = max(multi, key=lambda v: by_v[v].dynR2)
    best_lat_V = max(multi, key=lambda v: by_v[v].latentR2)
    d_gain = by_v[best_dyn_V].dynR2 - by_v[1].dynR2
    l_gain = by_v[best_lat_V].latentR2 - by_v[1].latentR2

    strong_dyn = d_gain > 0.08 and by_v[best_dyn_V].dynR2 > 0.5
    strong_lat = l_gain > 0.03

    if strong_dyn:
        label = "REPRODUCED"
    elif strong_lat and d_gain > -0.02:
        label = "REPRODUCED (latent)"
    elif l_gain > 0.0 or d_gain > 0.0:
        label = "PARTIAL"
    else:
        label = "NOT REPRODUCED"
    return Verdict(label, l_gain, d_gain, best_lat_V, best_dyn_V)


@dataclass
class ExperimentResult:
    """The result of a full view-sweep experiment."""

    config: DyscoConfig
    results: List[ViewResult]
    verdict: Verdict = field(init=False)

    def __post_init__(self):
        self.verdict = decide_verdict(self.results)

    # -- convenience accessors --
    def by_views(self) -> Dict[int, ViewResult]:
        return {r.V: r for r in self.results}

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "verdict": {
                "label": self.verdict.label,
                "latent_gain": self.verdict.latent_gain,
                "dyn_gain": self.verdict.dyn_gain,
                "best_latent_V": self.verdict.best_latent_V,
                "best_dyn_V": self.verdict.best_dyn_V,
            },
        }

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        return render_markdown(self)


def run_experiment(cfg: DyscoConfig, views: List[int], system: str = "lorenz",
                   device: Optional[str] = None, verbose: bool = True
                   ) -> ExperimentResult:
    """Train DYSCO across the requested view counts and return the comparison.

    Parameters
    ----------
    cfg:
        Experiment configuration shared across view counts.
    views:
        View counts to compare, e.g. ``[1, 4, 8]``.
    system:
        Ground-truth system name.
    device:
        Torch device; auto-selects CUDA if available when None.
    verbose:
        Print progress.

    Returns
    -------
    ExperimentResult
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}", flush=True)
        print(f"Config: {cfg.to_dict()}", flush=True)

    results: List[ViewResult] = []
    for V in views:
        if verbose:
            print(f"\n=== Training V={V} ===", flush=True)
        res = train_views(cfg, V, system=system, device=device, verbose=verbose)
        results.append(res)
        if verbose:
            print(f"  -> latentR2={res.latentR2:.4f}  dynR2={res.dynR2:.4f}",
                  flush=True)
    return ExperimentResult(config=cfg, results=results)


def render_markdown(exp: ExperimentResult) -> str:
    """Render an experiment to a markdown report (the ``EVAL.md`` format)."""
    cfg = exp.config
    v = exp.verdict
    by_v = exp.by_views()
    lines: List[str] = []
    lines.append("# DYSCO minimal reproduction — EVAL\n")
    lines.append(f"\n**Verdict: {v.label}**\n")
    lines.append("Paper: *Extracting Governing Equations from Latent Dynamics via "
                 "Multi-View Contrastive Learning* (arXiv:2606.13260)\n")
    lines.append("**Core claim under test:** multi-view temporal contrastive "
                 "learning recovers latent dynamics from noisy nonlinear "
                 "observations where single-view (DYNCL) fails.\n")
    lines.append(f"\nSetup: 3D Lorenz latent, random 4-layer MLP mixing to "
                 f"D={cfg.obs_dim}, additive Gaussian obs noise σ={cfg.noise}, "
                 f"degree-2 polynomial dynamics, InfoNCE w/ RK4 rollout "
                 f"(κ={cfg.kappa}), {cfg.steps} steps.\n")
    lines.append("\n## Results\n")
    lines.append("| Views (V) | latent R² | dyn R² |")
    lines.append("|-----------|-----------|--------|")
    for r in exp.results:
        lines.append(f"| {r.V} | {r.latentR2*100:.1f}% | {r.dynR2*100:.1f}% |")
    lines.append("")

    if v.label != "INCONCLUSIVE":
        lines.append("## Verdict\n")
        l1 = by_v[1].latentR2
        d1 = by_v[1].dynR2
        lines.append(f"- Latent R²  : V=1 {l1*100:.1f}%  →  best (V={v.best_latent_V}) "
                     f"{by_v[v.best_latent_V].latentR2*100:.1f}%  "
                     f"({v.latent_gain*100:+.1f} pts)")
        lines.append(f"- Dyn R²     : V=1 {d1*100:.1f}%  →  best (V={v.best_dyn_V}) "
                     f"{by_v[v.best_dyn_V].dynR2*100:.1f}%  "
                     f"({v.dyn_gain*100:+.1f} pts)")
        msg = {
            "REPRODUCED": "\n**Core claim REPRODUCED:** adding independent views "
                          "substantially improves recovery of the governing "
                          "dynamics (dynR²) under noisy nonlinear observations — "
                          "the multi-view denoising effect the paper relies on.",
            "REPRODUCED (latent)": "\n**Core claim REPRODUCED (latent):** multi-view "
                          "markedly improves latent-trajectory recovery (latentR²) "
                          "— the multi-view denoising mechanism — without "
                          "regressing dynamics recovery.",
            "PARTIAL": "\n**PARTIAL:** multi-view helps but the gap is below the "
                       "strong-reproduction threshold.",
            "NOT REPRODUCED": "\n**NOT reproduced:** multi-view did not improve "
                              "recovery in this run.",
        }.get(v.label, "")
        if msg:
            lines.append(msg)
    return "\n".join(lines) + "\n"
