"""Tests for the torch-free pre-encoded latent dataset (underfit data pipeline port).

No torch, no mlx — numpy + stdlib only, matching the module under test.
"""

import json
import random

import numpy as np
import pytest

from optimized.mlx.models.defs.latent_dataset import (
    PreEncodedLatentDataset,
    build_prompt,
    iterate_batches,
)

D = 8  # latent channels
CROP = 16  # latent_crop_length used throughout

LONG_T = 64  # stored length of the long item (> CROP)
LONG_VALID = 50  # 1s in its padding mask (trailing zeros after)
LONG_SECONDS = 120.5

SHORT_T = 6  # stored length of the short item (< CROP)
SHORT_SECONDS = 2.786

SILENCE_VALUE = 7.5


def _write_item(root, stem, T, seconds_total, tags, valid=None):
    """Write an underfit pre-encode style <stem>.npy + <stem>.json pair.

    Latents are [D, T] with latents[d, t] == t so a crop's start offset can be
    read straight off the values.
    """
    valid = T if valid is None else valid
    latents = np.tile(np.arange(T, dtype=np.float32), (D, 1))
    np.save(str(root / f"{stem}.npy"), latents)
    meta = {
        "relpath": f"{stem}.npy",
        "seconds_total": seconds_total,
        "seconds_start": 0,
        "audio_samples": int(seconds_total * 44100),
        "latent_shape": [D, T],
        "padding_mask": [1] * valid + [0] * (T - valid),
    }
    meta.update(tags)
    (root / f"{stem}.json").write_text(json.dumps(meta))


def _make_root(tmp_path, with_silence):
    root = tmp_path / ("latents" if with_silence else "latents_nosilence")
    root.mkdir()
    _write_item(
        root,
        "long",
        LONG_T,
        LONG_SECONDS,
        {
            "title": "Neon Skyline",
            "artist": "The Testers",
            "genre": "Synthwave",
            "bpm": "120",
        },
        valid=LONG_VALID,
    )
    # "prompt" tag = what pre_encode extracts from a .txt sidecar caption
    _write_item(
        root, "short", SHORT_T, SHORT_SECONDS, {"prompt": "a lofi hip hop beat"}
    )
    if with_silence:
        # Reference stores silence as [1, C, N] (it squeezes axis 0 on load)
        silence = np.full((1, D, 4), SILENCE_VALUE, dtype=np.float32)
        np.save(str(root / "silence.npy"), silence)
    return root


@pytest.fixture
def data_root(tmp_path):
    return _make_root(tmp_path, with_silence=True)


@pytest.fixture
def nosilence_root(tmp_path):
    return _make_root(tmp_path, with_silence=False)


def _get_by_relpath(ds, relpath):
    for i in range(len(ds)):
        item = ds[i]
        if item["relpath"] == relpath:
            return item
    raise AssertionError(f"{relpath} not found in dataset")


# ---------------------------------------------------------------------------
# Cropping / padding / masks
# ---------------------------------------------------------------------------


def test_crop_yields_exact_length_and_mask(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, random_crop=False)
    assert len(ds) == 2  # silence.npy is not a sample

    item = _get_by_relpath(ds, "long.npy")
    assert item["latents"].shape == (D, CROP)
    assert item["latents"].dtype == np.float32
    assert item["padding_mask"].shape == (CROP,)
    assert item["padding_mask"].dtype == np.bool_
    # random_crop=False → start at 0; first CROP frames are all valid
    np.testing.assert_array_equal(item["latents"][0], np.arange(CROP, dtype=np.float32))
    assert item["padding_mask"].all()


def test_random_crop_false_starts_at_zero(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, random_crop=False)
    for _ in range(20):
        item = _get_by_relpath(ds, "long.npy")
        assert item["latents"][0, 0] == 0.0


def test_random_crop_stays_in_valid_region(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, random_crop=True, seed=1234)
    last_ix = LONG_VALID - 1  # last 1 in the stored padding mask
    starts = set()
    for _ in range(200):
        item = _get_by_relpath(ds, "long.npy")
        start = int(item["latents"][0, 0])
        # reference: start = randint(0, last_ix - crop), inclusive
        assert 0 <= start <= last_ix - CROP
        # every crop is contiguous and the mask matches the stored one
        np.testing.assert_array_equal(
            item["latents"][0], np.arange(start, start + CROP, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            item["padding_mask"], (start + np.arange(CROP)) < LONG_VALID
        )
        starts.add(start)
    assert len(starts) > 1, "random_crop=True should vary the crop start"


def test_short_item_pads_with_silence(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, random_crop=False)
    item = _get_by_relpath(ds, "short.npy")

    assert item["latents"].shape == (D, CROP)
    # stored frames untouched
    np.testing.assert_array_equal(
        item["latents"][:, :SHORT_T],
        np.tile(np.arange(SHORT_T, dtype=np.float32), (D, 1)),
    )
    # padded frames come from the tiled/sliced silence latent
    np.testing.assert_array_equal(
        item["latents"][:, SHORT_T:],
        np.full((D, CROP - SHORT_T), SILENCE_VALUE, dtype=np.float32),
    )
    # mask: valid frames then zeros for the padding
    expected_mask = np.array([True] * SHORT_T + [False] * (CROP - SHORT_T))
    np.testing.assert_array_equal(item["padding_mask"], expected_mask)


def test_short_item_zero_pads_without_silence(nosilence_root):
    ds = PreEncodedLatentDataset(nosilence_root, CROP, random_crop=False)
    item = _get_by_relpath(ds, "short.npy")

    np.testing.assert_array_equal(
        item["latents"][:, SHORT_T:], np.zeros((D, CROP - SHORT_T), dtype=np.float32)
    )
    expected_mask = np.array([True] * SHORT_T + [False] * (CROP - SHORT_T))
    np.testing.assert_array_equal(item["padding_mask"], expected_mask)


def test_seconds_total_unchanged_by_crop_and_pad(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, random_crop=True)
    # deliberate underfit convention: FULL stored duration, never recomputed
    assert _get_by_relpath(ds, "long.npy")["seconds_total"] == LONG_SECONDS
    assert _get_by_relpath(ds, "short.npy")["seconds_total"] == SHORT_SECONDS


# ---------------------------------------------------------------------------
# Length filters (rejection redraw)
# ---------------------------------------------------------------------------


def test_min_length_sec_rejection_redraws(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, min_length_sec=50.0)
    # the short item (2.786 s) is always rejected → every index resolves to long
    for i in range(len(ds)):
        for _ in range(5):
            item = ds[i]
            assert item["relpath"] == "long.npy"
            assert item["seconds_total"] == LONG_SECONDS


def test_all_items_rejected_raises(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP, min_length_sec=1e6)
    with pytest.raises(RuntimeError):
        ds[0]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_oversample_epoch_yields_batch_size_x100_samples(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP)
    batch_size = 4  # > len(ds) == 2 → oversample with replacement
    batches = list(iterate_batches(ds, batch_size, seed=0))

    assert len(batches) == 100
    total = sum(b["latents"].shape[0] for b in batches)
    assert total == batch_size * 100

    for b in batches:
        assert b["latents"].shape == (batch_size, D, CROP)
        assert b["latents"].dtype == np.float32
        assert b["padding_mask"].shape == (batch_size, CROP)
        assert b["padding_mask"].dtype == np.bool_
        assert len(b["prompt"]) == batch_size
        assert len(b["seconds_total"]) == batch_size
        assert len(b["relpath"]) == batch_size

    # with replacement over 400 draws, both items must appear
    seen = {rp for b in batches for rp in b["relpath"]}
    assert seen == {"long.npy", "short.npy"}


def test_regular_epoch_drop_last_false(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP)
    # len(ds) == 2 >= batch_size → plain epoch, no oversampling
    batches = list(iterate_batches(ds, 2, seed=0))
    assert len(batches) == 1
    assert sorted(batches[0]["relpath"]) == ["long.npy", "short.npy"]

    # odd split keeps the remainder (drop_last=False)
    batches = list(iterate_batches(ds, 1, seed=0))
    assert [b["latents"].shape[0] for b in batches] == [1, 1]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def test_trigger_probability_approx_80pct():
    meta = {"title": "Song"}
    pc = {"trigger": "zkq"}  # trigger_pct defaults to 80
    rng = random.Random(123)

    prompts = [build_prompt(meta, pc, rng) for _ in range(2000)]
    with_trigger = sum(p.startswith("zkq, ") for p in prompts)
    frac = with_trigger / len(prompts)
    assert 0.77 < frac < 0.83, f"trigger fraction {frac} not ≈ 0.8"

    # tag prompt itself is always present (single tag → augmentation is a no-op)
    assert all(p.endswith("Title: Song") for p in prompts)


def test_shuffle_vs_subset_50_50():
    meta = {"title": "Neon", "artist": "Testers", "genre": "Synthwave", "bpm": "120"}
    expected_parts = {"Title: Neon", "Artist: Testers", "Genre: Synthwave", "BPM: 120"}
    pc = {}  # use_tags default True, shuffle default True, no trigger
    rng = random.Random(7)

    n_draws = 2000
    n_full = 0
    full_orderings = set()
    for _ in range(n_draws):
        prompt = build_prompt(meta, pc, rng)
        parts = prompt.split(", ")
        assert set(parts) <= expected_parts
        assert len(parts) == len(set(parts))
        if len(parts) == 4:
            n_full += 1
            full_orderings.add(prompt)

    # subset branch (p=0.5) draws size k ~ U{1..4}; k<4 w.p. 3/4 → partial ≈ 0.375
    partial_frac = 1 - n_full / n_draws
    assert 0.32 < partial_frac < 0.43, f"partial fraction {partial_frac} not ≈ 0.375"
    # shuffle-all branch occurred and actually shuffles
    assert n_full > 0
    assert len(full_orderings) > 1


def test_legacy_prompt_without_prompt_config():
    meta = {"artist": "The Testers", "title": "Neon Skyline"}
    rng = random.Random(3)
    lengths = set()
    for _ in range(300):
        prompt = build_prompt(meta, None, rng)
        parts = prompt.split(", ")
        assert set(parts) <= {"Artist: The Testers", "Title: Neon Skyline"}
        assert len(parts) >= 1
        lengths.add(len(parts))
    assert lengths == {1, 2}  # subset branch produces 1-tag prompts too

    # no tags at all → "text" fallback
    assert build_prompt({"text": "raw caption"}, None, rng) == "raw caption"


def test_legacy_prompt_via_dataset(data_root):
    ds = PreEncodedLatentDataset(data_root, CROP)  # prompt_config=None → legacy
    expected = {
        "Artist: The Testers",
        "Title: Neon Skyline",
        "BPM: 120",
        "Genre: Synthwave",
    }
    for _ in range(20):
        prompt = _get_by_relpath(ds, "long.npy")["prompt"]
        assert prompt
        assert set(prompt.split(", ")) <= expected


def test_txt_derived_prompt_tag(data_root):
    # the short item's caption came from a .txt sidecar → "prompt" tag key
    ds = PreEncodedLatentDataset(data_root, CROP, prompt_config={"shuffle": False})
    assert _get_by_relpath(ds, "short.npy")["prompt"] == "Prompt: a lofi hip hop beat"

    ds = PreEncodedLatentDataset(
        data_root,
        CROP,
        prompt_config={"shuffle": False, "hide_tag_names": True},
    )
    assert _get_by_relpath(ds, "short.npy")["prompt"] == "a lofi hip hop beat"


def test_path_prompt_and_space_joined_trigger():
    meta = {"relpath": "artistX/track01.npy"}
    rng = random.Random(11)

    pc = {"use_tags": False, "use_paths": True, "path_opts": {"hideExt": True}}
    assert build_prompt(meta, pc, rng) == "artistX/track01"

    # non-tag method → trigger joined with a space, not ", "
    pc = {
        "use_tags": False,
        "use_paths": True,
        "path_opts": {"hideExt": True},
        "trigger": "zkq",
        "trigger_pct": 100,
    }
    assert build_prompt(meta, pc, rng) == "zkq artistX/track01"
