"""Unit tests for optimized/mlx/scripts/pre_encode_mlx.py (stub encoder, no weights)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
soundfile = pytest.importorskip("soundfile")

import mlx.core as mx

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "optimized" / "mlx" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pre_encode_mlx as pe

SR = 44100


class StubEncoder:
    """[B, 512, P] patches → [B, 8, P/16]: 16-stride mean pool, first 8 channels.

    Stride-local like the real encoder's latent rate, so chunked stitching is
    bit-exact against unchunked — no model download needed.
    """

    def __call__(self, patches):
        batch, channels, patch_count = patches.shape
        pooled = mx.mean(
            patches.reshape(batch, channels, patch_count // 16, 16), axis=-1
        )
        return pooled[:, :8, :]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Audio tree: sub/three_sec.wav (stereo + JSON sidecar), long45.wav (mono + .txt)."""
    root = tmp_path_factory.mktemp("audio")

    t3 = np.arange(3 * SR) / SR
    stereo = np.stack(
        [
            0.4 * np.sin(2 * np.pi * 440 * t3) + 0.1,
            0.4 * np.sin(2 * np.pi * 330 * t3) + 0.1,
        ]
    ).astype(np.float32)
    (root / "sub").mkdir()
    soundfile.write(root / "sub" / "three_sec.wav", stereo.T, SR)
    (root / "sub" / "three_sec.json").write_text(
        json.dumps(
            {"artist": "Test Artist", "bpm": 128, "nested": {"x": 1}, "empty": ""}
        )
    )

    t45 = np.arange(45 * SR) / SR
    mono = (0.3 * np.sin(2 * np.pi * 220 * t45) + 0.05).astype(np.float32)
    soundfile.write(root / "long45.wav", mono, SR)
    (root / "long45.txt").write_text("a funky bassline, live drums\n")

    return root


@pytest.fixture()
def encoded(dataset, tmp_path):
    out = tmp_path / "latents"
    stats = pe.run(dataset, out, "same-s", StubEncoder())
    return out, stats


# ---------------------------------------------------------------------------
# End-to-end run: outputs, relpath preservation, details.json
# ---------------------------------------------------------------------------


def test_run_outputs_preserve_relpaths(encoded):
    out, stats = encoded
    assert stats == {"encoded": 2, "skipped": 0, "errors": 0, "count": 2}
    for rel in ("sub/three_sec.npy", "sub/three_sec.json", "long45.npy", "long45.json"):
        assert (out / rel).exists(), rel

    details = json.loads((out / "details.json").read_text())
    assert details["codec"] == "same-s"
    assert details["sample_rate"] == SR
    assert details["max_duration"] == 600.0
    assert details["count"] == 2


def test_three_sec_latents_and_metadata(encoded, dataset):
    out, _ = encoded
    latents = np.load(out / "sub" / "three_sec.npy")
    # 3 s = 132300 samples → padded to 17*8192 = 139264 (same-s alignment) → 34 latents
    assert latents.shape == (8, 34)
    assert latents.dtype == np.float32

    meta = json.loads((out / "sub" / "three_sec.json").read_text())
    assert meta["path"] == str(dataset / "sub" / "three_sec.wav")
    assert meta["relpath"] == "sub/three_sec.npy"
    assert meta["src_relpath"] == "sub/three_sec.wav"
    assert meta["seconds_total"] == 3.0
    assert meta["seconds_start"] == 0
    assert meta["audio_samples"] == 3 * SR
    assert meta["latent_shape"] == [8, 34]
    # ceil(132300 / 4096) = 33 valid latents; final latent is alignment padding
    mask = meta["padding_mask"]
    assert len(mask) == 34
    assert mask[:33] == [1] * 33
    assert mask[33] == 0
    # JSON sidecar tags merged: string/number values kept, nested/empty dropped
    assert meta["artist"] == "Test Artist"
    assert meta["bpm"] == "128"
    assert "nested" not in meta
    assert "empty" not in meta


def test_long45_chunked_mono_and_txt_prompt(encoded, dataset):
    out, _ = encoded
    latents = np.load(out / "long45.npy")
    # 45 s = 1984500 samples → padded to 243*8192 = 1990656 → 486 latents
    assert latents.shape == (8, 486)

    meta = json.loads((out / "long45.json").read_text())
    assert meta["audio_samples"] == 45 * SR
    assert meta["seconds_total"] == 45.0
    mask = meta["padding_mask"]
    assert len(mask) == 486
    assert sum(mask) == 485  # ceil(1984500/4096) = 485 valid
    assert mask[-1] == 0
    assert meta["prompt"] == "a funky bassline, live drums"

    # >30 s took the chunked path; stub is stride-local → must match unchunked
    audio = pe.load_audio(dataset / "long45.wav")
    ref = pe.encode_audio(
        StubEncoder(), audio[None, ...], pad_modulo=32, chunked=False
    )
    np.testing.assert_allclose(latents, np.asarray(ref.latents)[0], atol=1e-6)
    assert np.any(latents != 0.0)


# ---------------------------------------------------------------------------
# Skip vs overwrite
# ---------------------------------------------------------------------------


def test_skip_then_overwrite(dataset, tmp_path):
    out = tmp_path / "latents"
    stub = StubEncoder()

    first = pe.run(dataset, out, "same-s", stub)
    assert first["encoded"] == 2 and first["skipped"] == 0

    second = pe.run(dataset, out, "same-s", stub)
    assert second["encoded"] == 0 and second["skipped"] == 2

    # Corrupt an output, then --overwrite must re-encode it
    np.save(str(out / "long45.npy"), np.zeros((8, 486), dtype=np.float32))
    third = pe.run(dataset, out, "same-s", stub, overwrite=True)
    assert third["encoded"] == 2 and third["skipped"] == 0
    assert np.any(np.load(out / "long45.npy") != 0.0)


# ---------------------------------------------------------------------------
# Max-duration cap aligned down to 4096
# ---------------------------------------------------------------------------


def test_max_duration_aligns_down_to_4096(dataset, tmp_path):
    out = tmp_path / "latents"
    stats = pe.run(dataset, out, "same-s", StubEncoder(), max_duration=1.0)
    assert stats["encoded"] == 2

    # 1 s = 44100 samples → aligned down to 40960 = 10 * 4096
    assert pe.max_samples_for_duration(1.0) == 40960
    for name in ("sub/three_sec", "long45"):
        meta = json.loads((out / f"{name}.json").read_text())
        assert meta["audio_samples"] == 40960
        assert meta["seconds_total"] == round(40960 / SR, 3) == 0.929
        latents = np.load(out / f"{name}.npy")
        assert latents.shape == (8, 10)  # 40960 is already 8192-aligned, no padding
        assert meta["padding_mask"] == [1] * 10


# ---------------------------------------------------------------------------
# encode_file core
# ---------------------------------------------------------------------------


def test_encode_file_seconds_rounding_and_mask():
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((2, 130000)).astype(np.float32)

    latents, mask, seconds_total = pe.encode_file(
        StubEncoder(), audio, pad_modulo=32
    )
    assert seconds_total == round(130000 / SR, 3) == 2.948
    # 130000 → padded to 16*8192 = 131072 → 32 latents; ceil(130000/4096) = 32 valid
    assert latents.shape == (8, 32)
    assert latents.dtype == np.float32
    assert mask == [1] * 32

    # Cropping semantics: caller-declared valid length shortens the mask
    latents2, mask2, seconds2 = pe.encode_file(
        StubEncoder(), audio, pad_modulo=32, actual_samples=100000
    )
    assert seconds2 == round(100000 / SR, 3)
    assert len(mask2) == latents2.shape[-1]
    assert sum(mask2) == 25  # ceil(100000/4096)


def test_encode_file_rejects_non_stereo():
    with pytest.raises(ValueError, match=r"\(2, T\)"):
        pe.encode_file(StubEncoder(), np.zeros((1, 8192), dtype=np.float32), pad_modulo=32)


# ---------------------------------------------------------------------------
# Audio loading: mono→stereo, >2ch, resample
# ---------------------------------------------------------------------------


def test_load_audio_mono_duplicates_to_stereo(dataset):
    audio = pe.load_audio(dataset / "long45.wav")
    assert audio.shape == (2, 45 * SR)
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio[0], audio[1])


def test_load_audio_takes_first_two_of_multichannel(tmp_path):
    data = np.stack(
        [np.full(SR, 0.1), np.full(SR, 0.2), np.full(SR, 0.3)], axis=1
    ).astype(np.float32)
    path = tmp_path / "three_ch.wav"
    soundfile.write(path, data, SR)

    audio = pe.load_audio(path)
    assert audio.shape == (2, SR)
    np.testing.assert_allclose(audio[0], 0.1, atol=1e-3)
    np.testing.assert_allclose(audio[1], 0.2, atol=1e-3)


def test_load_audio_resamples_to_44100(tmp_path):
    sr_in = 22050
    t = np.arange(sr_in) / sr_in  # 1 s
    mono = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    path = tmp_path / "half_rate.wav"
    soundfile.write(path, mono, sr_in)

    audio = pe.load_audio(path)
    assert audio.shape == (2, SR)
    # Still a ~110 Hz sine after resampling: correlate with the ideal signal
    ideal = 0.3 * np.sin(2 * np.pi * 110 * np.arange(SR) / SR)
    corr = np.dot(audio[0], ideal) / (
        np.linalg.norm(audio[0]) * np.linalg.norm(ideal)
    )
    assert corr > 0.99


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------


def test_extract_tags_json_beats_txt(tmp_path):
    wav = tmp_path / "clip.wav"
    (tmp_path / "clip.json").write_text(json.dumps({"title": "A", "year": 2020}))
    (tmp_path / "clip.txt").write_text("prompt text")
    assert pe.extract_tags(wav) == {"title": "A", "year": "2020"}


def test_extract_tags_txt_becomes_prompt(tmp_path):
    wav = tmp_path / "clip.wav"
    (tmp_path / "clip.txt").write_text("  spaced out prompt \n")
    assert pe.extract_tags(wav) == {"prompt": "spaced out prompt"}


def test_extract_tags_sibling_dirs(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav = audio_dir / "clip.wav"

    (tmp_path / "json").mkdir()
    (tmp_path / "json" / "clip.json").write_text(json.dumps({"genre": "funk"}))
    assert pe.extract_tags(wav) == {"genre": "funk"}

    (tmp_path / "json" / "clip.json").unlink()
    (tmp_path / "txt").mkdir()
    (tmp_path / "txt" / "clip.txt").write_text("sibling prompt")
    assert pe.extract_tags(wav) == {"prompt": "sibling prompt"}


def test_extract_tags_no_sidecar_returns_empty(tmp_path):
    assert pe.extract_tags(tmp_path / "clip.wav") == {}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_find_audio_files_filters_and_sorts(tmp_path):
    rng = np.random.default_rng(3)
    big = (0.1 * rng.standard_normal(SR)).astype(np.float32)  # noise: flac stays >4 KB
    soundfile.write(tmp_path / "b.wav", big, SR)
    (tmp_path / "deep").mkdir()
    soundfile.write(tmp_path / "deep" / "a.flac", big, SR)
    soundfile.write(tmp_path / "._resource.wav", big, SR)  # macOS fork prefix
    (tmp_path / "tiny.wav").write_bytes(b"RIFF")  # below min size
    (tmp_path / "notes.txt").write_text("not audio")

    found = pe.find_audio_files(tmp_path)
    assert found == [tmp_path / "b.wav", tmp_path / "deep" / "a.flac"]
