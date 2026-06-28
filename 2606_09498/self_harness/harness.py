"""The harness h_t -- the editable surface Self-Harness improves.

In the paper the harness is a system prompt + tools + memory + orchestration.
Here we keep the minimal initial harness from the paper's spirit: a short
system prompt plus a list of behavioral "guidance" rules. Self-Harness may only
*append* grounded guidance rules (a bounded, minimal-edit surface) -- it cannot
rewrite the model or the task. This mirrors "modify only declared configuration
points within a minimal harness."
"""
import copy

# The minimal initial harness (h_0): deliberately terse, so characteristic
# failure modes surface and become clusterable.
INITIAL_HARNESS = {
    "system_prompt": "You are a coding agent. Solve the task.",
    "guidance": [],  # appended, grounded rules -- the self-improved surface
}


def clone(h):
    return copy.deepcopy(h)


def render_system(h):
    parts = [h["system_prompt"]]
    if h["guidance"]:
        parts.append("\nGuidelines:")
        for i, g in enumerate(h["guidance"], 1):
            parts.append(f"{i}. {g}")
    return "\n".join(parts)


def apply_edit(h, edit):
    """Apply a minimal harness edit. Supported op: append a guidance rule."""
    new = clone(h)
    if edit.get("op") == "append_guidance" and edit.get("text"):
        text = edit["text"].strip()
        if text and text not in new["guidance"]:
            new["guidance"].append(text)
    return new
