# -*- coding: utf-8 -*-
"""Conversation-side helpers for DIA-LISAt.

The answer templates of LISA/LISAt end with ``[SEG]``. DIA needs an explicit
concept token in front of it, so every ``[SEG]`` is rewritten into a
``[CON] ... [SEG]`` pair *before* tokenisation (i.e. inside ``collate_fn``),
which keeps the label-masking logic of ``handle_conversation_specifics``
untouched -- it always works on the same strings that get tokenised.

Two styles are available:

    clause   (default)  "Sure, it is [SEG]."
                     -> "Sure, it is [CON], so the segmentation result is [SEG]."
    adjacent            "Sure, it is [SEG]."
                     -> "Sure, it is [CON] [SEG]."

``clause`` is the recommended one: LISA reads the hidden state that *emits* a
special token, so with the adjacent style the ``[SEG]`` prompt would be read
directly off the ``[CON]`` position and the two roles would collapse again --
exactly the entanglement DIA is supposed to remove.
"""

from typing import List, Sequence

SEG_TOKEN = "[SEG]"
CON_TOKEN = "[CON]"

CLAUSE_TEMPLATE = "{con}, so the segmentation result is {seg}"
ADJACENT_TEMPLATE = "{con} {seg}"

__all__ = [
    "CON_TOKEN",
    "SEG_TOKEN",
    "count_tokens",
    "insert_con_tokens",
    "insert_con_tokens_batch",
    "validate_con_seg_pairs",
]


def insert_con_tokens(text: str, style: str = "clause") -> str:
    """Rewrite every ``[SEG]`` of ``text`` into a ``[CON] ... [SEG]`` pair.

    Idempotent: strings that already contain ``[CON]`` are returned unchanged.
    ``style`` may be ``"none"`` (or empty) to disable the rewrite entirely,
    which is the ablation "concept query := the ``[SEG]`` state itself".
    """
    if style in (None, "", "none"):
        return text
    if not text or SEG_TOKEN not in text or CON_TOKEN in text:
        return text
    if style == "clause":
        template = CLAUSE_TEMPLATE
    elif style == "adjacent":
        template = ADJACENT_TEMPLATE
    else:
        raise ValueError(f"unknown [CON] insertion style: {style}")
    return text.replace(SEG_TOKEN, template.format(con=CON_TOKEN, seg=SEG_TOKEN))


def insert_con_tokens_batch(
    conversations: Sequence[str], style: str = "clause"
) -> List[str]:
    return [insert_con_tokens(text, style=style) for text in conversations]


def count_tokens(text: str) -> tuple:
    """Return ``(n_con, n_seg)`` for a conversation string."""
    return text.count(CON_TOKEN), text.count(SEG_TOKEN)


def validate_con_seg_pairs(conversations: Sequence[str], strict: bool = False) -> bool:
    """Check that every conversation holds one ``[CON]`` per ``[SEG]``.

    Returns ``True`` when all rows are well formed. With ``strict=True`` a
    malformed row raises instead, which is useful once during a smoke run and
    better switched off for long trainings (the model already degrades
    gracefully when a pair is missing).
    """
    ok = True
    for text in conversations:
        n_con, n_seg = count_tokens(text)
        if n_seg == 0:
            continue
        if n_con != n_seg:
            ok = False
            message = (
                f"[DIA] unbalanced [CON]/[SEG] pair: n_con={n_con}, n_seg={n_seg} "
                f"in conversation: {text[-160:]!r}"
            )
            if strict:
                raise ValueError(message)
            print(message)
    return ok
