"""Unit tests for prompt_weighting.py — SA3's (phrase:weight) prompt syntax.

Run either way:
    python scripts/test_prompt_weighting.py
    pytest scripts/test_prompt_weighting.py
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from prompt_weighting import parse_weighted_prompt


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
