"""Small text utilities shared across the RAG nodes."""

import re

# A complete <think>...</think> reasoning block (deepseek-r1 style).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Any stray opening/closing think tag.
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought reasoning that some models (e.g. ``deepseek-r1``)
    emit inside ``<think>...</think>`` tags, returning only the final answer text.

    Handles:
    - well-formed ``<think>...</think> answer`` -> ``answer``
    - implicit-open ``reasoning... </think> answer`` -> ``answer`` (no opening tag)
    - no tags -> the text unchanged (trimmed)

    Non-string input is returned unchanged.
    """
    if not isinstance(text, str):
        return text
    # If a closing tag exists, the real answer is whatever follows the LAST one
    # (covers both well-formed blocks and reasoning dumps with no opening tag).
    lower = text.lower()
    if "</think>" in lower:
        text = text[lower.rfind("</think>") + len("</think>"):]
    # Drop any remaining complete blocks and stray tags, then trim.
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_TAG.sub("", text)
    return text.strip()
