"""
Tests for the stable-audio CLI (stable_audio_3/cli.py).

These are unit tests: the model and audio I/O are mocked so they run without
downloading weights or touching the GPU. Model behaviour is covered separately
in test_inference.py; here we verify that every CLI flag is wired correctly.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
CHANNELS = 2
FAKE_AUDIO_PATH = "some/audio.wav"
_FAKE_WAVEFORM = torch.zeros(CHANNELS, SAMPLE_RATE * 5)
_FAKE_LOAD_RESULT = (_FAKE_WAVEFORM, SAMPLE_RATE)


def _fake_audio(batch: int = 1, duration: float = 5.0) -> torch.Tensor:
    return torch.zeros(batch, CHANNELS, int(SAMPLE_RATE * duration))


def _make_model_mock(batch: int = 1, duration: float = 5.0):
    model = MagicMock()
    model.model.sample_rate = SAMPLE_RATE
    model.generate.return_value = _fake_audio(batch, duration)
    return model


def _run(argv: list[str]):
    """Invoke cli.main() with the given argument list."""
    from stable_audio_3.cli import main

    with patch.object(sys, "argv", ["stable-audio"] + argv):
        main()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_torchaudio_save():
    with patch("stable_audio_3.cli.torchaudio.save") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_torchaudio_load():
    with patch("stable_audio_3.cli.torchaudio.load", return_value=_FAKE_LOAD_RESULT):
        yield


@pytest.fixture()
def mock_model():
    model = _make_model_mock()
    with patch(
        "stable_audio_3.cli.StableAudioModel.from_pretrained", return_value=model
    ):
        yield model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        "medium",
        "small-music",
        "small-sfx",
        "medium-base",
        "small-music-base",
        "small-sfx-base",
    ],
)
def test_model_selection(mock_model, model_name):
    _run(["--model", model_name, "-p", "test"])
    from stable_audio_3.cli import StableAudioModel

    StableAudioModel.from_pretrained.assert_called_once_with(
        model_name, device=None, model_half=True
    )


def test_device_flag(mock_model):
    _run(["--device", "cpu", "-p", "test"])
    from stable_audio_3.cli import StableAudioModel

    _, kwargs = StableAudioModel.from_pretrained.call_args
    assert kwargs["device"] == "cpu"


def test_no_half_flag(mock_model):
    _run(["--no-half", "-p", "test"])
    from stable_audio_3.cli import StableAudioModel

    _, kwargs = StableAudioModel.from_pretrained.call_args
    assert kwargs["model_half"] is False


# ---------------------------------------------------------------------------
# Text-to-audio
# ---------------------------------------------------------------------------


def test_text_to_audio_defaults(mock_model):
    _run(["-p", "ocean waves"])
    mock_model.generate.assert_called_once()
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["prompt"] == "ocean waves"
    assert kwargs["duration"] == 120.0
    assert kwargs["steps"] == 8
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["seed"] == -1
    assert kwargs["batch_size"] == 1
    assert kwargs["negative_prompt"] is None
    assert kwargs["init_audio"] is None
    assert kwargs["inpaint_audio"] is None
    assert kwargs["chunked_decode"] is None


def test_generation_flags(mock_model):
    _run(
        [
            "-p",
            "drums",
            "--duration",
            "20",
            "--steps",
            "50",
            "--cfg-scale",
            "7",
            "--seed",
            "42",
        ]
    )
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["duration"] == 20.0
    assert kwargs["steps"] == 50
    assert kwargs["cfg_scale"] == 7.0
    assert kwargs["seed"] == 42


def test_negative_prompt(mock_model):
    _run(["-p", "jazz", "--negative-prompt", "bad quality"])
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["negative_prompt"] == "bad quality"


# ---------------------------------------------------------------------------
# Output file saving
# ---------------------------------------------------------------------------


def test_output_single(mock_model, mock_torchaudio_save, tmp_path):
    out = str(tmp_path / "out.wav")
    _run(["-p", "test", "-o", out])
    assert mock_torchaudio_save.call_count == 1
    saved_path, saved_tensor, saved_sr = mock_torchaudio_save.call_args.args
    assert saved_path == out
    assert saved_sr == SAMPLE_RATE
    assert torch.equal(saved_tensor, mock_model.generate.return_value[0].cpu())


def test_output_batch_naming(mock_torchaudio_save, tmp_path):
    model = _make_model_mock(batch=3)
    with patch(
        "stable_audio_3.cli.StableAudioModel.from_pretrained", return_value=model
    ):
        out = str(tmp_path / "out.wav")
        _run(["-p", "a", "b", "c", "--batch-size", "3", "-o", out])

    saved_paths = [c.args[0] for c in mock_torchaudio_save.call_args_list]
    base = str(tmp_path / "out")
    assert saved_paths == [f"{base}_0.wav", f"{base}_1.wav", f"{base}_2.wav"]


# ---------------------------------------------------------------------------
# Batch with per-batch prompts and durations
# ---------------------------------------------------------------------------


def test_batch_per_batch_prompts(mock_torchaudio_save):
    model = _make_model_mock(batch=3)
    with patch(
        "stable_audio_3.cli.StableAudioModel.from_pretrained", return_value=model
    ):
        _run(["-p", "p1", "p2", "p3", "--batch-size", "3"])
    kwargs = model.generate.call_args.kwargs
    assert kwargs["prompt"] == ["p1", "p2", "p3"]
    assert kwargs["batch_size"] == 3


def test_batch_per_batch_durations(mock_torchaudio_save):
    model = _make_model_mock(batch=2)
    with patch(
        "stable_audio_3.cli.StableAudioModel.from_pretrained", return_value=model
    ):
        _run(["-p", "p1", "p2", "--batch-size", "2", "--duration", "20", "30"])
    kwargs = model.generate.call_args.kwargs
    assert kwargs["duration"] == [20.0, 30.0]


def test_batch_per_batch_negative_prompts(mock_torchaudio_save):
    model = _make_model_mock(batch=2)
    with patch(
        "stable_audio_3.cli.StableAudioModel.from_pretrained", return_value=model
    ):
        _run(["-p", "p1", "p2", "--batch-size", "2", "--negative-prompt", "n1", "n2"])
    kwargs = model.generate.call_args.kwargs
    assert kwargs["negative_prompt"] == ["n1", "n2"]


# ---------------------------------------------------------------------------
# Audio-to-audio
# ---------------------------------------------------------------------------


def test_audio_to_audio(mock_model):
    _run(
        [
            "-p",
            "bossa nova",
            "--init-audio",
            FAKE_AUDIO_PATH,
            "--init-noise-level",
            "0.7",
        ]
    )
    kwargs = mock_model.generate.call_args.kwargs
    sr, waveform = kwargs["init_audio"]
    assert sr == SAMPLE_RATE
    assert torch.equal(waveform, _FAKE_WAVEFORM)
    assert kwargs["init_noise_level"] == 0.7
    assert kwargs["inpaint_audio"] is None


def test_audio_to_audio_default_noise_level(mock_model):
    _run(["-p", "test", "--init-audio", FAKE_AUDIO_PATH])
    assert mock_model.generate.call_args.kwargs["init_noise_level"] == 0.9


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------


def test_inpaint_single_region(mock_model):
    _run(
        [
            "-p",
            "kick drum",
            "--inpaint-audio",
            FAKE_AUDIO_PATH,
            "--inpaint-start",
            "2.0",
            "--inpaint-end",
            "5.0",
        ]
    )
    kwargs = mock_model.generate.call_args.kwargs
    sr, waveform = kwargs["inpaint_audio"]
    assert sr == SAMPLE_RATE
    assert torch.equal(waveform, _FAKE_WAVEFORM)
    assert kwargs["init_audio"] is None
    assert kwargs["inpaint_mask_start_seconds"] == 2.0
    assert kwargs["inpaint_mask_end_seconds"] == 5.0


def test_inpaint_multiple_regions(mock_model):
    _run(
        [
            "-p",
            "fill",
            "--inpaint-audio",
            FAKE_AUDIO_PATH,
            "--inpaint-start",
            "1.0",
            "--inpaint-start",
            "8.0",
            "--inpaint-end",
            "4.0",
            "--inpaint-end",
            "12.0",
        ]
    )
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["inpaint_mask_start_seconds"] == [1.0, 8.0]
    assert kwargs["inpaint_mask_end_seconds"] == [4.0, 12.0]


def test_inpaint_continuation(mock_model):
    """Continuation: inpaint_start == length of source audio, duration > source length."""
    _run(
        [
            "-p",
            "continue",
            "--inpaint-audio",
            FAKE_AUDIO_PATH,
            "--inpaint-start",
            "5.0",
            "--inpaint-end",
            "15.0",
            "--duration",
            "15",
        ]
    )
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["inpaint_mask_start_seconds"] == 5.0
    assert kwargs["inpaint_mask_end_seconds"] == 15.0
    assert kwargs["duration"] == 15.0


def test_inpaint_region_without_audio_fails():
    with pytest.raises(SystemExit):
        _run(["-p", "test", "--inpaint-start", "2.0", "--inpaint-end", "5.0"])


def test_inpaint_audio_without_region_fails():
    with pytest.raises(SystemExit):
        _run(["-p", "test", "--inpaint-audio", FAKE_AUDIO_PATH])


def test_inpaint_start_without_end_fails():
    with pytest.raises(SystemExit):
        _run(
            ["-p", "test", "--inpaint-audio", FAKE_AUDIO_PATH, "--inpaint-start", "2.0"]
        )


def test_inpaint_mismatched_region_count_fails():
    with pytest.raises(SystemExit):
        _run(
            [
                "-p",
                "test",
                "--inpaint-audio",
                FAKE_AUDIO_PATH,
                "--inpaint-start",
                "1.0",
                "--inpaint-start",
                "5.0",
                "--inpaint-end",
                "3.0",
            ]
        )


# ---------------------------------------------------------------------------
# Chunked decode
# ---------------------------------------------------------------------------


def test_chunked_decode_on(mock_model):
    _run(["-p", "test", "--chunked-decode"])
    assert mock_model.generate.call_args.kwargs["chunked_decode"] is True


def test_chunked_decode_off(mock_model):
    _run(["-p", "test", "--no-chunked-decode"])
    assert mock_model.generate.call_args.kwargs["chunked_decode"] is False


def test_chunked_decode_default(mock_model):
    _run(["-p", "test"])
    assert mock_model.generate.call_args.kwargs["chunked_decode"] is None


def test_chunked_decode_flags_mutually_exclusive():
    with pytest.raises(SystemExit):
        _run(["-p", "test", "--chunked-decode", "--no-chunked-decode"])


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def test_lora_single(mock_model):
    _run(["-p", "test", "--lora-ckpt-path", "lora.safetensors"])
    mock_model.load_lora.assert_called_once_with(["lora.safetensors"])


def test_lora_stacked(mock_model):
    _run(
        [
            "-p",
            "test",
            "--lora-ckpt-path",
            "a.safetensors",
            "--lora-ckpt-path",
            "b.safetensors",
        ]
    )
    mock_model.load_lora.assert_called_once_with(["a.safetensors", "b.safetensors"])


def test_lora_strength(mock_model):
    _run(
        ["-p", "test", "--lora-ckpt-path", "lora.safetensors", "--lora-strength", "0.5"]
    )
    mock_model.set_lora_strength.assert_called_once_with(0.5, lora_index=None)


def test_lora_strength_with_index(mock_model):
    _run(
        [
            "-p",
            "test",
            "--lora-ckpt-path",
            "a.safetensors",
            "--lora-ckpt-path",
            "b.safetensors",
            "--lora-strength",
            "0.3",
            "--lora-index",
            "1",
        ]
    )
    mock_model.set_lora_strength.assert_called_once_with(0.3, lora_index=1)


def test_no_lora_no_load(mock_model):
    _run(["-p", "test"])
    mock_model.load_lora.assert_not_called()
    mock_model.set_lora_strength.assert_not_called()
