"""The Self-Harness three-stage self-improvement loop (paper sec 4), end to end.

  1. Weakness Mining  -- run agent under h_t, cluster failures by
     verifier-grounded signature (the terminal cause c_i).
  2. Harness Proposal -- the SAME fixed model proposes K minimal, grounded,
     diverse harness edits from the evidence bundle.
  3. Proposal Validation -- evaluate each candidate on held-in AND held-out
     splits; accept only under the non-regression rule
         d_in >= 0 and d_ho >= 0 and max(d_in, d_ho) > 0
     and merge accepted edits into h_{t+1}.

The model is fixed throughout and plays both agent and proposer.
"""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from typing import Any

from . import harness as H
from . import tasks as T
from .config import Config
from .llm import LLMClient
from .verifier import evaluate


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
@dataclass
class Result:
    """Structured outcome of a Self-Harness run."""

    model: str
    n_in: int
    n_ho: int
    base_in: int
    base_ho: int
    final_in: int
    final_ho: int
    final_harness: dict
    rounds: list = field(default_factory=list)

    @property
    def d_in(self) -> int:
        return self.final_in - self.base_in

    @property
    def d_ho(self) -> int:
        return self.final_ho - self.base_ho

    @property
    def claim_reproduced(self) -> bool:
        """Paper's success condition: non-regression on both splits, gain on one."""
        return self.d_in >= 0 and self.d_ho >= 0 and (self.d_in > 0 or self.d_ho > 0)

    @property
    def final_system_prompt(self) -> str:
        return H.render_system(self.final_harness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_in": self.n_in,
            "n_ho": self.n_ho,
            "baseline": {"p_in": self.base_in, "p_ho": self.base_ho},
            "final": {
                "p_in": self.final_in,
                "p_ho": self.final_ho,
                "harness": self.final_harness,
                "system_prompt": self.final_system_prompt,
            },
            "claim_reproduced": self.claim_reproduced,
            "rounds": self.rounds,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class SelfHarness:
    """Runs the self-improvement loop for a given Config."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.client = LLMClient(self.cfg.model, self.cfg.require_key())

    # --- Agent: solve one task under harness h ---
    def run_agent(self, h, task) -> str:
        system = H.render_system(h)
        user = (
            f"Task: {task['prompt']}\n\n"
            "Respond with a single Python function named solve in a ```python code block."
        )
        return self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.cfg.agent_temperature,
            max_tokens=self.cfg.max_tokens,
        )

    def evaluate_split(self, h, split, label) -> tuple[int, list]:
        records, passed = [], 0
        for task in split:
            out = self.run_agent(h, task)
            rec = evaluate(task, out)
            records.append(rec)
            passed += int(rec["passed"])
            status = "PASS" if rec["passed"] else f"FAIL({rec['cause']})"
            print(f"  [{label}] {task['id']:20s} {status}", flush=True)
        return passed, records

    # --- Stage 1: Weakness Mining ---
    @staticmethod
    def mine_weaknesses(records) -> list:
        failures = [r for r in records if not r["passed"]]
        clusters = collections.defaultdict(list)
        for r in failures:
            clusters[r["cause"]].append(r)  # exact-match clustering on cause c_i
        bundles = []
        for cause, recs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            bundles.append(
                {
                    "cause": cause,
                    "size": len(recs),
                    "tasks": [r["task_id"] for r in recs],
                    "evidence": [r.get("detail", "") for r in recs if r.get("detail")][:3],
                }
            )
        return bundles

    # --- Stage 2: Harness Proposal ---
    def propose_edits(self, h, bundles) -> list:
        k = self.cfg.k_proposals
        bundle_text = json.dumps(bundles, indent=2)
        current = H.render_system(h)
        system = (
            "You are improving the operating harness of a coding agent. The agent and "
            "you are the SAME model. You may only APPEND short, grounded guidance rules "
            "to the harness system prompt. Each edit must (1) target a primary failure "
            "cause in the evidence, (2) be minimal (one concrete rule), (3) preserve "
            "unrelated behavior, (4) be diverse from the other proposals."
        )
        user = (
            f"Current harness system prompt:\n---\n{current}\n---\n\n"
            f"Verifier-grounded failure clusters (cause, size, tasks):\n{bundle_text}\n\n"
            f"Propose exactly {k} DISTINCT minimal harness edits as a JSON array. Each "
            'element: {"op": "append_guidance", "target_cause": "<cause>", '
            '"text": "<one concrete guideline sentence>", "rationale": "<why it fixes that cause>"}. '
            "Return ONLY the JSON array."
        )
        out = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.cfg.proposer_temperature,
            max_tokens=1200,
        )
        start, end = out.find("["), out.rfind("]")
        if start == -1 or end == -1:
            print("[propose] no JSON array found; raw:", out[:300], flush=True)
            return []
        try:
            edits = json.loads(out[start : end + 1])
        except json.JSONDecodeError as e:
            print(f"[propose] JSON parse error: {e}", flush=True)
            return []
        clean = [e for e in edits if e.get("op") == "append_guidance" and e.get("text")]
        return clean[:k]

    # --- Stage 3: Proposal Validation (non-regression acceptance rule) ---
    def validate_and_merge(self, h, edits, base_in, base_ho):
        accepted, rejected = [], []
        h_next = H.clone(h)
        for j, edit in enumerate(edits):
            cand = H.apply_edit(h, edit)  # candidate evaluated vs ORIGINAL h_t
            p_in, _ = self.evaluate_split(cand, T.held_in(), f"cand{j}/in")
            p_ho, _ = self.evaluate_split(cand, T.held_out(), f"cand{j}/ho")
            d_in, d_ho = p_in - base_in, p_ho - base_ho
            accept = d_in >= 0 and d_ho >= 0 and max(d_in, d_ho) > 0
            audit = {
                "edit": edit, "p_in": p_in, "p_ho": p_ho,
                "d_in": d_in, "d_ho": d_ho, "accepted": accept,
            }
            print(
                f"[validate] cand{j} target={edit.get('target_cause')} "
                f"d_in={d_in:+d} d_ho={d_ho:+d} -> {'ACCEPT' if accept else 'reject'}",
                flush=True,
            )
            (accepted if accept else rejected).append(audit)
            if accept:
                h_next = H.apply_edit(h_next, edit)  # merge compatible accepted edits
        return h_next, accepted, rejected

    # --- Full run ---
    def run(self) -> Result:
        print("=== Self-Harness minimal reproduction ===", flush=True)
        print(f"model={self.cfg.model}", flush=True)

        h = H.clone(H.INITIAL_HARNESS)
        print("\n--- Baseline h_0 ---", flush=True)
        base_in, in_recs = self.evaluate_split(h, T.held_in(), "base/in")
        base_ho, ho_recs = self.evaluate_split(h, T.held_out(), "base/ho")
        n_in, n_ho = len(T.held_in()), len(T.held_out())
        print(f"baseline held-in {base_in}/{n_in}  held-out {base_ho}/{n_ho}", flush=True)

        rounds_log = []
        cur_in, cur_ho = base_in, base_ho

        for rnd in range(self.cfg.rounds):
            print(f"\n=== Round {rnd+1} ===", flush=True)
            bundles = self.mine_weaknesses(in_recs)
            print(
                f"[mine] {len(bundles)} failure clusters: "
                f"{[(b['cause'], b['size']) for b in bundles]}",
                flush=True,
            )
            if not bundles:
                print("[mine] no held-in failures; nothing to improve.", flush=True)
                break
            edits = self.propose_edits(h, bundles)
            print(f"[propose] {len(edits)} candidate edits", flush=True)
            if not edits:
                break
            h_next, accepted, rejected = self.validate_and_merge(h, edits, cur_in, cur_ho)
            rounds_log.append(
                {"round": rnd + 1, "bundles": bundles,
                 "accepted": accepted, "rejected": rejected}
            )
            if not accepted:
                print("[round] no edit accepted; harness unchanged. Stopping.", flush=True)
                break
            h = h_next
            cur_in, in_recs = self.evaluate_split(h, T.held_in(), "promoted/in")
            cur_ho, ho_recs = self.evaluate_split(h, T.held_out(), "promoted/ho")
            print(
                f"[round {rnd+1}] promoted held-in {cur_in}/{n_in}  held-out {cur_ho}/{n_ho}",
                flush=True,
            )

        return Result(
            model=self.cfg.model,
            n_in=n_in, n_ho=n_ho,
            base_in=base_in, base_ho=base_ho,
            final_in=cur_in, final_ho=cur_ho,
            final_harness=h, rounds=rounds_log,
        )


def run_self_harness(config: Config | None = None) -> Result:
    """Convenience entrypoint: run the loop and return a Result."""
    return SelfHarness(config).run()
