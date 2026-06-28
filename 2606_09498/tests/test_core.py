"""Offline tests for the non-LLM machinery (verifier, harness, weakness mining).

These need no API key and run in <1s, so CI / Colab can validate the loop's
deterministic core without spending tokens.
"""
from self_harness import harness as H
from self_harness import tasks as T
from self_harness import verifier as V
from self_harness.loop import SelfHarness
from self_harness.verifier import PROTOCOL_MARKER


def _wrap(code: str, marker: bool = True) -> str:
    prefix = f"{PROTOCOL_MARKER}\n" if marker else ""
    return f"{prefix}```python\n{code}\n```"


def test_correct_solution_passes():
    rec = V.evaluate(T.held_in()[0], _wrap("def solve(nums):\n    return sum(nums)"))
    assert rec["passed"], rec


def test_missing_protocol_marker_is_a_cause():
    rec = V.evaluate(T.held_in()[0], _wrap("def solve(nums):\n    return sum(nums)", marker=False))
    assert rec["cause"] == "missing_protocol_marker", rec


def test_no_artifact_cause():
    rec = V.evaluate(T.held_in()[0], f"{PROTOCOL_MARKER}\nI think the answer is sum(nums).")
    assert rec["cause"] == "no_artifact", rec


def test_wrong_answer_cause():
    rec = V.evaluate(T.held_in()[0], _wrap("def solve(nums):\n    return 999"))
    assert rec["cause"] == "wrong_answer", rec


def test_timeout_cause():
    rec = V.evaluate(T.held_in()[0], _wrap("def solve(nums):\n    while True: pass"))
    assert rec["cause"] == "timeout", rec


def test_multi_arg_task():
    two = [t for t in T.TASKS if t["id"] == "two_sum"][0]
    code = (
        "def solve(nums, target):\n"
        "    seen={}\n"
        "    for i,n in enumerate(nums):\n"
        "        if target-n in seen: return [seen[target-n], i]\n"
        "        seen[n]=i"
    )
    assert V.evaluate(two, _wrap(code))["passed"]


def test_weakness_mining_clusters_by_cause():
    recs = [
        {"task_id": "a", "passed": False, "cause": "missing_protocol_marker"},
        {"task_id": "b", "passed": False, "cause": "missing_protocol_marker"},
        {"task_id": "c", "passed": False, "cause": "wrong_answer", "detail": "x"},
        {"task_id": "d", "passed": True, "cause": None},
    ]
    bundles = SelfHarness.mine_weaknesses(recs)
    by_cause = {b["cause"]: b["size"] for b in bundles}
    assert by_cause["missing_protocol_marker"] == 2
    assert by_cause["wrong_answer"] == 1


def test_apply_edit_appends_guidance():
    h = H.clone(H.INITIAL_HARNESS)
    h2 = H.apply_edit(h, {"op": "append_guidance", "text": "Always emit FINAL_SOLUTION."})
    assert "Always emit FINAL_SOLUTION." in H.render_system(h2)
    # idempotent: re-applying the same text does not duplicate.
    h3 = H.apply_edit(h2, {"op": "append_guidance", "text": "Always emit FINAL_SOLUTION."})
    assert H.render_system(h3).count("Always emit FINAL_SOLUTION.") == 1


def test_splits_are_disjoint_and_cover_all():
    ids_in = {t["id"] for t in T.held_in()}
    ids_ho = {t["id"] for t in T.held_out()}
    assert ids_in.isdisjoint(ids_ho)
    assert ids_in | ids_ho == {t["id"] for t in T.TASKS}
