"""The 50 overthinking-marker words from Table 8 of arXiv:2606.00206.

In the paper every marker is a tokenizer token with a leading space (denoted
"_"). Rather than hardcode token IDs (which are tokenizer-specific), we expand
each marker into the set of vocabulary token IDs whose decoded surface form,
once stripped, matches the marker (case-insensitively), covering both the
leading-space variant and the bare variant. This reproduces the paper's intent
("penalize the overthinking-marker tokens") in a tokenizer-agnostic way.
"""

# Table 8 markers (leading space stripped to the bare word).
OVERTHINKING_MARKERS = [
    "perhaps", "maybe", "wait", "Wait", "actually", "hold", "Hmm", "hmm",
    "Alternatively", "alternatively", "However", "however", "instead",
    "Instead", "But", "but", "though", "although", "yet", "rather", "unless",
    "otherwise", "nonetheless", "nevertheless", "regardless", "still",
    "anyway", "Or", "or", "either", "whether", "uncertain", "unsure",
    "possibly", "might", "could", "another", "different", "reconsider",
    "rethink", "backtrack", "retry", "recheck", "revisit", "doubt",
    "confused", "wrong", "mistake", "error", "incorrect",
]


def build_marker_token_ids(tokenizer):
    """Return the set of vocab token IDs corresponding to the overthinking
    markers, matching both leading-space and bare variants per marker word."""
    wanted = set(m.lower() for m in OVERTHINKING_MARKERS)
    marker_ids = set()
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        try:
            surface = tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            continue
        if surface is None:
            continue
        # Normalize the SentencePiece/BPE leading-space markers (Ġ, ▁) and
        # whitespace, then lowercase.
        norm = surface.replace("\u0120", " ").replace("\u2581", " ").strip().lower()
        if norm in wanted:
            marker_ids.add(tid)
    return sorted(marker_ids)
