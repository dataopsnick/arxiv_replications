"""Configuration for a Self-Harness run.

A single dataclass collects every knob so that a run is fully reproducible from
one object. Defaults reproduce the minimal experiment reported in the README.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Model (the SAME fixed model is agent AND proposer, per the paper) ---
    model: str = field(
        default_factory=lambda: os.environ.get(
            "SELF_HARNESS_MODEL", "deepseek/deepseek-v4-flash"
        )
    )
    # OpenRouter API key. Read from the environment by default; never hard-code.
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY")
    )

    # --- Loop knobs ---
    rounds: int = 1               # improvement rounds (1 suffices for the claim)
    k_proposals: int = 4          # parallel candidate edits proposed per round
    agent_temperature: float = 0.2
    proposer_temperature: float = 0.6  # higher -> more diverse proposals
    max_tokens: int = 2048

    # --- IO ---
    artifacts_dir: str = ".openresearch/artifacts"

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY or pass "
                "Config(api_key=...)."
            )
        return self.api_key
