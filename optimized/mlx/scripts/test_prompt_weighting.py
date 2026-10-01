"""Unit tests for prompt_weighting.py — SA3's (phrase:weight) prompt syntax.

Run either way:
    python scripts/test_prompt_weighting.py
    pytest scripts/test_prompt_weighting.py
"""
import contextlib
import io
import os
import sys

import mlx.core as mx
import numpy as np
import sentencepiece as spm

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from prompt_weighting import parse_weighted_prompt, compute_token_spans, apply_prompt_weights


def test_no_weighted_spans():
    clean, spans = parse_weighted_prompt("rigid mechanical grid pulse")
    assert clean == "rigid mechanical grid pulse"
    assert spans == []


def test_single_weighted_span():
    clean, spans = parse_weighted_prompt("a (driving rhythmic pulse:2) in the mix")
    assert clean == "a driving rhythmic pulse in the mix"
    assert spans == [(2, 24, 2.0)]
    assert clean[spans[0][0]:spans[0][1]] == "driving rhythmic pulse"


def test_weighted_span_at_prompt_start():
    clean, spans = parse_weighted_prompt("(cold synthetic timbre:1.7), sparse texture")
    assert clean == "cold synthetic timbre, sparse texture"
    assert spans == [(0, 21, 1.7)]
    assert clean[spans[0][0]:spans[0][1]] == "cold synthetic timbre"


def test_multiple_weighted_spans():
    clean, spans = parse_weighted_prompt("(rhythm:2) and (silence:0.3) together")
    assert clean == "rhythm and silence together"
    phrases = [clean[s:e] for s, e, _ in spans]
    weights = [w for _, _, w in spans]
    assert phrases == ["rhythm", "silence"]
    assert weights == [2.0, 0.3]


def test_repeated_phrase_text_in_and_out_of_span():
    clean, spans = parse_weighted_prompt("pulse and (pulse:2) again")
    assert clean == "pulse and pulse again"
    assert len(spans) == 1
    s, e, w = spans[0]
    assert clean[s:e] == "pulse"
    assert w == 2.0
    # must be the SECOND "pulse" (the weighted one), not the first
    assert s == clean.index("pulse", 1)


def test_negative_and_fractional_weights():
    clean, spans = parse_weighted_prompt("(low end:-1) and (air:0.25)")
    assert [w for _, _, w in spans] == [-1.0, 0.25]


def test_malformed_syntax_passes_through_untouched():
    raw = "(unclosed paren and (no weight) plain text"
    clean, spans = parse_weighted_prompt(raw)
    assert clean == raw
    assert spans == []


def _load_tokenizer():
    """Load just the bundled SentencePiece tokenizer, no model weights —
    matches how T5Gemma.from_npz stores it (models/defs/t5gemma_mlx.py),
    without paying for the full transformer load in every test run."""
    npz_path = os.path.join(REPO, "..", "models", "mlx", "t5gemma_f16.npz")
    arrs = np.load(npz_path)
    tok = spm.SentencePieceProcessor()
    tok.LoadFromSerializedProto(arrs["TOKENIZER_MODEL"].tobytes())
    return tok


def test_boundaries_match_real_phrase_tokens():
    tok = _load_tokenizer()
    raw = "rigid mechanical grid pulse, (driving rhythmic pulse:2), cold synthetic timbre"
    clean, char_spans = parse_weighted_prompt(raw)
    token_spans = compute_token_spans(clean, char_spans, tok, max_len=256)
    assert len(token_spans) == 1
    start, end, weight = token_spans[0]
    assert weight == 2.0
    full_ids = tok.Encode(clean)
    phrase = clean[char_spans[0][0]:char_spans[0][1]]
    # the phrase appears mid-sentence, so it's preceded by a space in the
    # full tokenization — re-tokenizing it WITH a leading space must give
    # exactly the token IDs the computed boundaries slice out.
    assert full_ids[start:end] == tok.Encode(" " + phrase)


def test_boundaries_at_prompt_start_need_no_leading_space():
    tok = _load_tokenizer()
    raw = "(cold synthetic timbre:1.7), sparse texture"
    clean, char_spans = parse_weighted_prompt(raw)
    token_spans = compute_token_spans(clean, char_spans, tok, max_len=256)
    start, end, weight = token_spans[0]
    full_ids = tok.Encode(clean)
    phrase = clean[char_spans[0][0]:char_spans[0][1]]
    assert full_ids[start:end] == tok.Encode(phrase)


def test_multiple_spans_all_resolve_correctly():
    tok = _load_tokenizer()
    raw = "(rhythm:2) and (silence:0.3) together"
    clean, char_spans = parse_weighted_prompt(raw)
    token_spans = compute_token_spans(clean, char_spans, tok, max_len=256)
    assert len(token_spans) == 2
    full_ids = tok.Encode(clean)
    expected_phrases = ["rhythm", "silence"]
    expected_leading_space = [False, True]  # "rhythm" is at position 0
    for (start, end, weight), phrase, needs_space in zip(
        token_spans, expected_phrases, expected_leading_space
    ):
        want = tok.Encode((" " if needs_space else "") + phrase)
        assert full_ids[start:end] == want


def test_span_beyond_max_len_is_dropped():
    tok = _load_tokenizer()
    raw = ("pulse " * 400) + "(rhythm:2) tail"
    clean, char_spans = parse_weighted_prompt(raw)
    token_spans = compute_token_spans(clean, char_spans, tok, max_len=256)
    assert token_spans == []


def test_phrase_with_trailing_space_does_not_include_stray_whitespace_token():
    tok = _load_tokenizer()
    raw = "a (rhythm :2) and silence"
    clean, char_spans = parse_weighted_prompt(raw)
    # parse_weighted_prompt captures "rhythm " (with trailing space)
    assert clean == "a rhythm  and silence"
    assert len(char_spans) == 1
    start_char, end_char, weight = char_spans[0]
    assert clean[start_char:end_char] == "rhythm "

    token_spans = compute_token_spans(clean, char_spans, tok, max_len=256)
    assert len(token_spans) == 1
    start, end, weight = token_spans[0]
    assert weight == 2.0

    full_ids = tok.Encode(clean)
    span_tokens = full_ids[start:end]

    # The phrase appears after "a ", so it gets a leading space in tokenization.
    # The trailing space in the captured phrase text must NOT appear as a
    # stray whitespace-only token in the span. Re-tokenize the phrase with
    # a leading space (since it's mid-sentence) — this should match.
    phrase_text = clean[start_char:end_char].rstrip()  # "rhythm" without trailing space
    expected_tokens = tok.Encode(" " + phrase_text)
    assert span_tokens == expected_tokens, (
        f"span tokens {span_tokens} should equal {expected_tokens} "
        f"(phrase with leading space, no trailing space)"
    )


def test_degenerate_no_separator_before_paren_is_skipped_with_warning():
    """No whitespace/punctuation separates a word from the opening paren
    (e.g. "un(usual:2)" -> clean "unusual tone", a single SentencePiece
    token) — there's no token boundary at the span's start to land on.
    compute_token_spans must not silently drop or misalign the weight;
    it should skip the span and print a visible warning to stderr."""
    tok = _load_tokenizer()
    raw = "un(usual:2) tone"
    clean, char_spans = parse_weighted_prompt(raw)
    assert clean == "unusual tone"
    assert len(char_spans) == 1

    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        token_spans = compute_token_spans(clean, char_spans, tok, max_len=256)

    assert token_spans == []
    warning = stderr_capture.getvalue()
    assert "skipping span" in warning
    assert "usual" in warning


def test_weight_one_is_a_bit_exact_noop():
    embeds = mx.array(np.random.default_rng(0).standard_normal((1, 6, 4)).astype(np.float32))
    pe = mx.array(np.random.default_rng(1).standard_normal((4,)).astype(np.float32))
    out = apply_prompt_weights(embeds, [(1, 3, 1.0)], pe)
    assert np.array_equal(np.asarray(out), np.asarray(embeds))


def test_empty_spans_is_a_bit_exact_noop():
    embeds = mx.array(np.random.default_rng(0).standard_normal((1, 6, 4)).astype(np.float32))
    pe = mx.array(np.random.default_rng(1).standard_normal((4,)).astype(np.float32))
    out = apply_prompt_weights(embeds, [], pe)
    assert np.array_equal(np.asarray(out), np.asarray(embeds))


def test_weight_zero_collapses_span_to_padding_embedding():
    embeds = mx.array(np.random.default_rng(0).standard_normal((1, 6, 4)).astype(np.float32))
    pe = mx.array(np.random.default_rng(1).standard_normal((4,)).astype(np.float32))
    out = np.asarray(apply_prompt_weights(embeds, [(2, 4, 0.0)], pe))
    pe_np = np.asarray(pe)
    for t in (2, 3):
        assert np.allclose(out[0, t], pe_np, atol=1e-5)
    # untouched positions are unchanged
    embeds_np = np.asarray(embeds)
    for t in (0, 1, 4, 5):
        assert np.array_equal(out[0, t], embeds_np[0, t])


def test_weight_extrapolates_linearly_past_plain_embedding():
    embeds = mx.array(np.random.default_rng(0).standard_normal((1, 6, 4)).astype(np.float32))
    pe = mx.array(np.random.default_rng(1).standard_normal((4,)).astype(np.float32))
    out = np.asarray(apply_prompt_weights(embeds, [(0, 2, 2.0)], pe))
    embeds_np = np.asarray(embeds)
    pe_np = np.asarray(pe)
    expected = pe_np + 2.0 * (embeds_np[0, 0] - pe_np)
    assert np.allclose(out[0, 0], expected, atol=1e-5)


def test_multiple_spans_do_not_interfere():
    embeds = mx.array(np.random.default_rng(0).standard_normal((1, 6, 4)).astype(np.float32))
    pe = mx.array(np.random.default_rng(1).standard_normal((4,)).astype(np.float32))
    out = np.asarray(apply_prompt_weights(embeds, [(0, 1, 0.0), (4, 6, 3.0)], pe))
    embeds_np = np.asarray(embeds)
    pe_np = np.asarray(pe)
    assert np.allclose(out[0, 0], pe_np, atol=1e-5)
    for t in (1, 2, 3):
        assert np.array_equal(out[0, t], embeds_np[0, t])
    for t in (4, 5):
        expected = pe_np + 3.0 * (embeds_np[0, t] - pe_np)
        assert np.allclose(out[0, t], expected, atol=1e-5)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    print("\n" + ("ALL PASS" if not failed else f"FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
