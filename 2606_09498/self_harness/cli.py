"""Command-line entrypoint: `python -m self_harness` or `self-harness`."""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .llm import LLMError
from .loop import run_self_harness
from .report import write_artifacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="self-harness",
        description="Minimal reproduction of arXiv 2606.09498 (Self-Harness).",
    )
    p.add_argument("--model", default=None, help="OpenRouter model slug (agent+proposer).")
    p.add_argument("--rounds", type=int, default=None, help="Number of improvement rounds.")
    p.add_argument("--k-proposals", type=int, default=None, help="Candidate edits per round.")
    p.add_argument("--artifacts-dir", default=None, help="Where to write EVAL.md / history.")
    p.add_argument("--no-write", action="store_true", help="Do not write EVAL.md artifacts.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {}
    if args.model is not None:
        overrides["model"] = args.model
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.k_proposals is not None:
        overrides["k_proposals"] = args.k_proposals
    if args.artifacts_dir is not None:
        overrides["artifacts_dir"] = args.artifacts_dir

    cfg = Config(**overrides)

    try:
        result = run_self_harness(cfg)
    except LLMError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not args.no_write:
        md = write_artifacts(result, cfg.artifacts_dir)
        print("\n=== EVAL.md written ===", flush=True)
        print(md, flush=True)

    # Exit 0 if the core claim reproduced, else 2 (run still completed fine).
    return 0 if result.claim_reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
