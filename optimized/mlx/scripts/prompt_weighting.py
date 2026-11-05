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
excludes parens), so for a nested group like "(a (b:2):1.5)" it is the
INNER group that actually matches — the regex scans left to right and
`[^():]+` cannot span across the inner "(", so the first complete
"(...)" the regex finds is the inner one. The result: only the inner
phrase gets a weight (2.0 in that example, not 1.5), and the outer
group's opening "(" and trailing ":1.5)" survive as literal characters
in the clean prompt text fed to the text encoder. Concretely,
"sparse ((rhythm:2):3) tail" cleans to "sparse (rhythm:3) tail" with a
span on "rhythm" at weight 2.0 — the outer ":3)" is literal text, not a
second weight.

Span-boundary exactness is not absolute. A span's token boundaries are
exact when the span begins right after a whitespace or punctuation
separator before the opening "(" — the common case, verified across 56
separator combinations. When a phrase instead immediately follows a
word character with no separator at all (e.g. "un(usual:2) tone",
whose clean text "unusual tone" tokenizes "unusual" as a single
SentencePiece token), there is no token boundary at the span's start to
land on, and the naive prefix-count would return an empty or
misaligned span. This is the same subword-alignment limitation every
SD-style prompt-weighting tool has — `compute_token_spans` detects this
degenerate case and skips the span with a warning printed to stderr,
rather than silently applying a wrong or empty weight.
"""
import re
import sys

import mlx.core as mx
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


def compute_token_spans(
    clean_text: str,
    char_spans: list[tuple[int, int, float]],
    tokenizer,
    max_len: int = 256,
) -> list[tuple[int, int, float]]:
    """Turn character-offset spans (from parse_weighted_prompt) into
    token-offset spans, by construction rather than by search.

    For each span, tokenizing the clean text's *prefix* up to each of
    its two boundaries and taking the resulting token counts gives the
    span's exact [start:end) slice in the full tokenization — this
    relies on SentencePiece prefix tokenization being a stable
    extension (verified against this project's own tokenizer: longer
    prefixes of the same text never retroactively change an earlier
    prefix's tokens) AND on the span's start boundary landing on an
    actual token boundary, which any whitespace or punctuation
    character immediately before the opening "(" ensures. This holds
    for 56 of the 64 separator combinations swept in review; it is NOT
    guaranteed when the phrase instead abuts a preceding word character
    with no separator at all (e.g. "un(usual:2)"), where the "start" a
    caller intends can fall inside a token that only exists in the
    merged (unsplit) form. This function detects that degenerate case
    (`n_before >= n_through` below) and skips the span with a stderr
    warning rather than returning an empty or misaligned one.

    `tokenizer` needs only `.Encode(str) -> list[int]` — a bare
    sentencepiece.SentencePieceProcessor, or T5Gemma's own `.tokenizer`
    attribute at the real call site.

    Spans that fall entirely past `max_len` (already truncated away by
    T5Gemma.tokenize's own max_len truncation) are dropped — nothing
    new to handle, since untruncated positions are exactly what the
    rest of the pipeline already ignores past max_len.
    """
    token_spans = []
    for start_char, end_char, weight in char_spans:
        # If there's a space before the phrase, tokenize up to (but not including)
        # the space so the space merges with the phrase tokens in the full tokenization.
        prefix_end = start_char
        if start_char > 0 and clean_text[start_char - 1] == " ":
            prefix_end = start_char - 1

        # Similarly, if the phrase itself ends in a space, back up so the trailing
        # space merges with the preceding phrase tokens, not as a stray whitespace token.
        suffix_end = end_char
        if end_char > 0 and clean_text[end_char - 1] == " ":
            suffix_end = end_char - 1

        n_before = min(len(tokenizer.Encode(clean_text[:prefix_end])), max_len)
        n_through = min(len(tokenizer.Encode(clean_text[:suffix_end])), max_len)
        if n_before >= max_len:
            continue
        if n_before >= n_through:
            # Degenerate case: no token boundary at this span's start (the
            # phrase immediately follows a word character with no space/
            # punctuation separator before the opening paren, so the
            # phrase's first word got merged into a token that only exists
            # in the unsplit form). Don't silently drop the weight or hand
            # back an empty/misaligned span — make it visible.
            phrase = clean_text[start_char:end_char]
            print(
                f"prompt_weighting: skipping span for {phrase!r} — no token "
                "boundary here (needs a space/punctuation before the "
                "opening paren)",
                file=sys.stderr,
            )
            continue
        token_spans.append((n_before, n_through, weight))
    return token_spans


def apply_prompt_weights(
    embeds,
    token_spans: list[tuple[int, int, float]],
    padding_embedding,
):
    """Reweight spans of T5Gemma output toward/away from SA3's own
    learned "empty" embedding — the same interpolation ComfyUI/compel
    use for SD-style prompt weighting, anchored on this model's own
    padding_embedding instead of an invented reference point.

    embeds            : (B, S, 768) mx.array — T5Gemma last_hidden_state.
    token_spans       : (start, end, weight) triples, as returned by
                         compute_token_spans. weight=1.0 spans are
                         skipped entirely so a prompt with no non-1.0
                         weights returns `embeds` unchanged (bit-exact,
                         not just numerically close).
    padding_embedding : (768,) mx.array — SA3's learned unconditional
                         embedding, the same tensor apply_prompt_padding
                         uses.

    Returns a new array; does not mutate `embeds` in place, matching
    apply_prompt_padding's functional style.
    """
    S = embeds.shape[1]
    weights = np.ones((S,), dtype=np.float32)
    changed = False
    for start, end, weight in token_spans:
        if weight == 1.0 or start >= end:
            continue
        weights[start:end] = weight
        changed = True
    if not changed:
        return embeds

    # Build a mask of which positions have non-1.0 weights
    # Only interpolate positions where weight != 1.0 to preserve bit-exact
    # equality for unchanged positions
    mask = (weights != 1.0).astype(np.float32)

    pe = padding_embedding.astype(embeds.dtype).reshape(1, 1, -1)
    w = mx.array(weights.reshape(1, S, 1), dtype=embeds.dtype)
    m = mx.array(mask.reshape(1, S, 1), dtype=embeds.dtype)

    # Only apply interpolation where mask is 1 (weight != 1.0)
    # For mask=0 positions, just use embeds unchanged
    weighted = pe + w * (embeds - pe)
    return m * weighted + (1 - m) * embeds
