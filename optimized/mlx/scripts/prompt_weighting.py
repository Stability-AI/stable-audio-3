"""(phrase:weight) inline prompt weighting for SA3's T5Gemma conditioning.

Three independent pieces, in the order the pipeline uses them:
  1. parse_weighted_prompt  — regex-parse (phrase:weight) spans out of the
     raw prompt, producing a clean prompt (parens/weight stripped) plus
     each span's character offsets *into the clean prompt*.
  2. compute_token_spans    — turn those character offsets into token
     offsets by construction (incremental prefix tokenization), not by
     searching for a phrase's tokens after the fact.
  3. apply_prompt_weights   — interpolate each span's embedding toward/
     away from SA3's own learned "empty" (padding) embedding.

Nested/overlapping (phrase:weight) groups are not supported: the regex
below only matches non-nested groups (the phrase character class
excludes parens), so a nested group either fails to match at all
(passes through as literal text) or the outer group matches with the
inner syntax left inside the phrase text.
"""
import re

import numpy as np

_WEIGHT_RE = re.compile(r"\(([^():]+):([+-]?\d+(?:\.\d+)?)\)")


def parse_weighted_prompt(raw: str) -> tuple[str, list[tuple[int, int, float]]]:
    """Parse (phrase:weight) spans out of `raw`.

    Returns (clean_text, spans) where clean_text has the parens/weight
    syntax stripped (bare phrase left in place) and each span is
    (start_char, end_char, weight) — character offsets into clean_text,
    not raw. Malformed syntax (no regex match) passes through untouched
    with no span recorded for it.
    """
    clean_parts = []
    spans: list[tuple[int, int, float]] = []
    pos = 0
    clean_len = 0
    for m in _WEIGHT_RE.finditer(raw):
        literal = raw[pos:m.start()]
        clean_parts.append(literal)
        clean_len += len(literal)

        phrase = m.group(1)
        weight = float(m.group(2))
        start = clean_len
        clean_parts.append(phrase)
        clean_len += len(phrase)
        end = clean_len
        spans.append((start, end, weight))

        pos = m.end()
    clean_parts.append(raw[pos:])
    return "".join(clean_parts), spans
