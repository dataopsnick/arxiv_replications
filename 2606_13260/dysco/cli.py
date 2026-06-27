"""Command-line interface for the DYSCO reproduction.

Run the canonical single-view vs multi-view experiment and write an ``EVAL.md``
report plus a ``results.json``:

    python -m dysco --noise 1.5 --views 1,4,8 --steps 6000

or, after ``pip install -e .``:

    dysco --noise 1.5 --views 1,4,8
"""
from __future__ import annotations

import argparse
import os

from .config import DyscoConfig
from .experiment import run_experiment

ART_DIR = os.path.join(".openresearch", "artifacts")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dysco",
        description="Minimal DYSCO reproduction (arXiv:2606.13260): "
                    "single-view vs multi-view recovery of latent dynamics.")
    ap.add_argument("--T", type=int, default=5000, help="trajectory length")
    ap.add_argument("--obs_dim", type=int, default=32, help="observation dim D")
    ap.add_argument("--dt", type=float, default=0.01, help="integration step")
    ap.add_argument("--noise", type=float, default=1.5,
                    help="Gaussian observation-noise sigma")
    ap.add_argument("--steps", type=int, default=6000,
                    help="base optimisation steps (for V=1)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kappa", type=int, default=4,
                    help="max temporal integration horizon")
    ap.add_argument("--temp", type=float, default=1.0, help="InfoNCE temperature")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=str, default="1,4,8",
                    help="comma-separated view counts to compare")
    ap.add_argument("--system", type=str, default="lorenz")
    ap.add_argument("--no-fair-budget", action="store_true",
                    help="disable scaling steps with the number of views")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--out", type=str, default="EVAL.md",
                    help="path for the markdown report")
    ap.add_argument("--json", type=str,
                    default=os.path.join(ART_DIR, "results.json"),
                    help="path for the JSON results")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    views = [int(v) for v in args.views.split(",")]
    cfg = DyscoConfig(
        T=args.T, obs_dim=args.obs_dim, dt=args.dt, noise=args.noise,
        steps=args.steps, batch=args.batch, lr=args.lr, kappa=args.kappa,
        temp=args.temp, seed=args.seed, fair_budget=not args.no_fair_budget,
        log_every=args.log_every,
    )
    exp = run_experiment(cfg, views=views, system=args.system)

    md = exp.to_markdown()
    with open(args.out, "w") as f:
        f.write(md)
    # also drop copies into the artifacts dir for orx
    os.makedirs(ART_DIR, exist_ok=True)
    with open(os.path.join(ART_DIR, "EVAL.md"), "w") as f:
        f.write(md)
    exp.to_json(args.json)

    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
