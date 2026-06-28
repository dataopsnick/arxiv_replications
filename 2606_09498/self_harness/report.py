"""Render a Result into the EVAL.md report + supporting JSON artifacts."""
from __future__ import annotations

import json
import pathlib

from . import harness as H
from .loop import Result


def _pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else 0.0


def render_eval_md(result: Result) -> str:
    di, dh = result.d_in, result.d_ho
    md = f"""# Self-Harness Minimal Reproduction -- EVAL

Model (agent **and** proposer): `{result.model}` via OpenRouter.
Tasks: {result.n_in} held-in / {result.n_ho} held-out verifiable coding tasks (Terminal-Bench-2.0 stand-in).
Acceptance rule (paper sec 4.3): `d_in >= 0 and d_ho >= 0 and max(d_in,d_ho) > 0`.

## Core claim: the harness improves itself

| Split    | Baseline h_0 | After Self-Harness | Abs gain |
|----------|--------------|--------------------|----------|
| Held-in  | {result.base_in}/{result.n_in} ({_pct(result.base_in, result.n_in):.1f}%) | {result.final_in}/{result.n_in} ({_pct(result.final_in, result.n_in):.1f}%) | {di:+d} |
| Held-out | {result.base_ho}/{result.n_ho} ({_pct(result.base_ho, result.n_ho):.1f}%) | {result.final_ho}/{result.n_ho} ({_pct(result.final_ho, result.n_ho):.1f}%) | {dh:+d} |

**Claim reproduced: {result.claim_reproduced}** (non-regression on both splits, improvement on at least one).

## Rounds
"""
    for r in result.rounds:
        md += (
            f"\n### Round {r['round']}\n"
            f"- failure clusters: {[(b['cause'], b['size']) for b in r['bundles']]}\n"
            f"- accepted edits: {len(r['accepted'])}, rejected: {len(r['rejected'])}\n"
        )
        for a in r["accepted"]:
            md += (
                f"  - ACCEPT (d_in={a['d_in']:+d}, d_ho={a['d_ho']:+d}) "
                f"target=`{a['edit'].get('target_cause')}`: {a['edit']['text']}\n"
            )

    md += f"\n## Final self-improved harness\n\n```\n{result.final_system_prompt}\n```\n"
    return md


def write_artifacts(result: Result, artifacts_dir: str = ".openresearch/artifacts") -> str:
    """Write EVAL.md (repo root + artifacts) and history.json. Returns the report."""
    art = pathlib.Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    md = render_eval_md(result)
    pathlib.Path("EVAL.md").write_text(md)
    (art / "EVAL.md").write_text(md)
    (art / "history.json").write_text(json.dumps(result.to_dict(), indent=2))
    (art / "final_harness.txt").write_text(result.final_system_prompt)
    return md
