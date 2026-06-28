"""self_harness -- a minimal, faithful reproduction of arXiv 2606.09498,
"Self-Harness: Harnesses That Improve Themselves".

A fixed LLM acts as both *agent* and *proposer* and improves its own operating
harness via the paper's three-stage loop (weakness mining -> harness proposal ->
non-regression validation), demonstrated on a small suite of verifiable tasks.

Quickstart
----------
    from self_harness import Config, run_self_harness, write_artifacts

    result = run_self_harness(Config(api_key="sk-or-..."))
    print(result.claim_reproduced, result.d_in, result.d_ho)
    write_artifacts(result)        # writes EVAL.md
"""
from .config import Config
from .harness import INITIAL_HARNESS, apply_edit, render_system
from .llm import LLMClient, LLMError
from .loop import Result, SelfHarness, run_self_harness
from .report import render_eval_md, write_artifacts
from . import tasks
from .verifier import PROTOCOL_MARKER, evaluate

__version__ = "0.1.0"

__all__ = [
    "Config",
    "SelfHarness",
    "run_self_harness",
    "Result",
    "write_artifacts",
    "render_eval_md",
    "LLMClient",
    "LLMError",
    "INITIAL_HARNESS",
    "apply_edit",
    "render_system",
    "evaluate",
    "PROTOCOL_MARKER",
    "tasks",
    "__version__",
]
