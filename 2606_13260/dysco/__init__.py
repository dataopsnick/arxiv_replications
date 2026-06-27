"""DYSCO — minimal reproduction library.

A small, self-contained implementation of **DYSCO** (Muratore & Mathis,
*Extracting Governing Equations from Latent Dynamics via Multi-View Contrastive
Learning*, arXiv:2606.13260).

It demonstrates the paper's core claim: multi-view temporal contrastive learning
recovers latent trajectories and their governing equations from noisy,
high-dimensional, nonlinear observations — where single-view contrastive
identification (DYNCL) fails.

Quick start
-----------
>>> from dysco import DyscoConfig, run_experiment
>>> cfg = DyscoConfig(T=5000, obs_dim=32, noise=1.5, steps=6000)
>>> exp = run_experiment(cfg, views=[1, 4, 8])
>>> print(exp.verdict.label)            # e.g. "REPRODUCED"
>>> print(exp.to_markdown())            # full EVAL.md report
"""
from .config import DyscoConfig
from .systems import lorenz_f, simulate_lorenz, simulate, SYSTEMS
from .models import (
    RandomMixing, Encoder, SymbolicDynamics,
    poly_library, rk4_step, rollout, POLY_TERMS, N_LIB,
)
from .data import generate, MultiViewData
from .evaluate import affine_fit, r2_score, evaluate
from .train import train_views, ViewResult
from .experiment import (
    run_experiment, ExperimentResult, Verdict, decide_verdict, render_markdown,
)

__version__ = "0.1.0"

__all__ = [
    "DyscoConfig",
    "lorenz_f", "simulate_lorenz", "simulate", "SYSTEMS",
    "RandomMixing", "Encoder", "SymbolicDynamics",
    "poly_library", "rk4_step", "rollout", "POLY_TERMS", "N_LIB",
    "generate", "MultiViewData",
    "affine_fit", "r2_score", "evaluate",
    "train_views", "ViewResult",
    "run_experiment", "ExperimentResult", "Verdict", "decide_verdict",
    "render_markdown",
    "__version__",
]
