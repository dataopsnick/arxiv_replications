"""Fixed evaluator E: runs an agent's solution in a sandboxed subprocess and
returns a pass/fail outcome plus a *verifier-grounded cause* on failure.

The cause is the terminal, verifier-level reason a run failed -- the c_i in the
paper's failure signature phi(r) = (c, q, m). It is grounded in what the
verifier actually observed (no code extracted, code didn't run, wrong answer,
timeout), never inferred from the model's narration.
"""
import json
import re
import subprocess
import sys
import tempfile
import textwrap

TIMEOUT_S = 10

# Hidden grading protocol the verifier enforces (like a benchmark's submission
# contract). The TASK PROMPTS never mention it -- only the verifier knows it, so
# the agent must learn it from verifier-grounded failure causes. This is the
# protocol surface the minimal harness omits and Self-Harness rediscovers.
PROTOCOL_MARKER = "FINAL_SOLUTION"


def extract_code(text):
    """Pull a python code block (the produced artifact) from the agent output."""
    # Prefer fenced ```python ... ``` blocks, then any ``` block.
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if the whole thing looks like a def, use it raw.
    if "def solve" in text:
        return text.strip()
    return None


_RUNNER = textwrap.dedent(
    """
    import json, sys
    {code}

    cases = json.loads(sys.argv[1])
    out = []
    for args, _expected in cases:
        out.append(solve(*args))
    print("__RESULT__" + json.dumps(out))
    """
)


def evaluate(task, agent_output):
    """Return a record dict: passed (bool) and, if failed, a `cause`."""
    code = extract_code(agent_output)
    rec = {"task_id": task["id"], "passed": False, "cause": None}

    # Protocol gate: the submission must declare the required marker line. This
    # is the contract the verifier enforces but the prompt never states.
    if PROTOCOL_MARKER not in agent_output:
        rec["cause"] = "missing_protocol_marker"
        rec["detail"] = f"submission must contain the required marker '{PROTOCOL_MARKER}'"
        return rec

    if code is None:
        rec["cause"] = "no_artifact"  # agent never produced a solution block
        return rec
    if "def solve" not in code:
        rec["cause"] = "no_solve_function"
        return rec

    # Normalize cases to (args_list, expected).
    cases = []
    for c in task["cases"]:
        args, expected = c[0], c[1]
        cases.append([args, expected])

    runner = _RUNNER.format(code=textwrap.indent(code, ""))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(runner)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path, json.dumps(cases)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        rec["cause"] = "timeout"
        return rec

    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        rec["cause"] = "runtime_error"
        rec["detail"] = err[-1] if err else "nonzero exit"
        return rec

    marker = "__RESULT__"
    line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
    if line is None:
        rec["cause"] = "no_output"
        return rec

    try:
        produced = json.loads(line[len(marker):])
    except json.JSONDecodeError:
        rec["cause"] = "bad_output_format"
        return rec

    expected = [c[1] for c in cases]
    if produced == expected:
        rec["passed"] = True
        return rec

    rec["cause"] = "wrong_answer"
    rec["detail"] = f"got {produced} expected {expected}"
    return rec
