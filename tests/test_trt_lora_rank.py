"""pad_rank(): the stack-rank rounding that keeps the TensorRT branch GEMMs on their fast path.

Pure arithmetic, so this needs neither a GPU nor TensorRT -- which is the point: the rule it
encodes (round up to a multiple of 8, but never past what the engine can bind) is easy to
break in a refactor and expensive to notice, because breaking it costs ~40% on two GEMMs
while still producing correct audio.
"""

from pathlib import Path

import pytest

LORA = (
    Path(__file__).resolve().parents[1] / "optimized" / "tensorRT" / "scripts" / "lora"
)


def _load_pad_rank():
    """Import branch_runtime's pad_rank without importing the package (it needs tensorrt)."""
    src = (LORA / "branch_runtime.py").read_text()
    start = src.index("RANK_GRANULARITY")
    end = src.index("class BranchLora")
    ns: dict = {}
    exec(compile(src[start:end], "branch_runtime:pad_rank", "exec"), ns)
    return ns["RANK_GRANULARITY"], ns["pad_rank"]


GRAN, pad_rank = _load_pad_rank()


def test_granularity_is_eight():
    # 8 fp16 values = 16 bytes = one vectorised load, and the fp16 tensor-core fragment
    # wants K/N in multiples of 8. Changing this silently de-optimises every branch engine.
    assert GRAN == 8


@pytest.mark.parametrize(
    "r,want",
    [
        (1, 8),
        (2, 8),
        (4, 8),
        (7, 8),
        (8, 8),
        (9, 16),
        (12, 16),
        (16, 16),
        (17, 24),
        (24, 24),
        (31, 32),
        (32, 32),
        (64, 64),
        (128, 128),
        (505, 512),
        (512, 512),
    ],
)
def test_rounds_up_to_a_multiple_of_eight(r, want):
    assert pad_rank(r, 512) == want


@pytest.mark.parametrize("r", list(range(1, 65)))
def test_result_is_always_aligned_and_never_shrinks(r):
    out = pad_rank(r, 512)
    assert out % GRAN == 0, f"pad_rank({r}) -> {out} is not a multiple of {GRAN}"
    assert out >= r, f"pad_rank({r}) -> {out} dropped rows"
    assert out - r < GRAN, f"pad_rank({r}) -> {out} padded further than necessary"


@pytest.mark.parametrize("r,cap", [(97, 100), (100, 100), (7, 7), (1, 4), (3, 5)])
def test_never_exceeds_a_cap_that_is_not_a_multiple_of_eight(r, cap):
    # An engine built with --rank-max not divisible by 8 is the one case where rounding up
    # would push the bound rank outside the optimisation profile. set_input_shape only
    # RETURNS False there, so the render would proceed on stale bindings.
    out = pad_rank(r, cap)
    assert out <= cap, f"pad_rank({r}, cap={cap}) -> {out} exceeds the cap"
    assert out >= r


def test_padding_is_skipped_rather_than_truncated_when_capped():
    # Skipping the padding must keep the REAL rank, not round down and drop adapter rows.
    assert pad_rank(97, 100) == 97


def test_zero_and_negative_are_clamped_to_a_legal_rank():
    # Rank 0 is not a legal TensorRT shape here: the profile minimum is 1 and
    # set_input_shape rejects a 0 dimension outright.
    for r in (0, -1, -8):
        assert pad_rank(r, 512) == GRAN


def test_cap_is_optional():
    assert pad_rank(5) == 8
    assert pad_rank(16) == 16
