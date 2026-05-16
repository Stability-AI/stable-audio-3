import re
from pathlib import Path

import pytest
import torch
import torchaudio

from stable_audio_3 import AutoencoderModel, StableAudioModel
from stable_audio_3.model_configs import ae_models

# ---------------------------------------------------------------------------
# Hardware detection — used by fixtures and tests to gate GPU-only paths
# ---------------------------------------------------------------------------
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()
HAS_ACCEL = HAS_CUDA or HAS_MPS
ACCEL_DEVICE = "cuda" if HAS_CUDA else ("mps" if HAS_MPS else "cpu")


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--save-audio",
        action="store_true",
        default=False,
        help="Save generated audio to disk. Files are written to test_audio_outputs/.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    """Best available compute device for this session."""
    return ACCEL_DEVICE


@pytest.fixture(scope="session", params=["small-music", "small-sfx", "medium"])
def sa3_model(request):
    """Session-scoped model fixture parametrized over model sizes.

    small-music — loads via from_pretrained("small-music"); runs on CPU or accelerator.
    small-sfx   — loads via from_pretrained("small-sfx"); runs on CPU or accelerator.
    medium      — requires a CUDA GPU; skipped otherwise.
    """
    name = request.param

    if name in ("small-music", "small-sfx"):
        return StableAudioModel.from_pretrained(name, device=ACCEL_DEVICE)

    if name == "medium":
        if not HAS_CUDA:
            pytest.skip("Medium model requires a CUDA GPU — none detected")
        return StableAudioModel.from_pretrained("medium", device=ACCEL_DEVICE)


@pytest.fixture(scope="session", params=list(ae_models))
def autoencoder(request):
    """Session-scoped autoencoder model fixture parametrized over AE model sizes.

    same-l requires a CUDA GPU; skipped otherwise.
    """
    name = request.param
    if name == "same-l" and not HAS_CUDA:
        pytest.skip(f"{name} requires a CUDA GPU — none detected")

    return AutoencoderModel.from_pretrained(name, device=ACCEL_DEVICE)


@pytest.fixture
def maybe_save_audio(request):
    """Return a callable that saves audio to disk when --save-audio is passed.

    Usage in tests:
        def test_foo(sa3_model, maybe_save_audio):
            audio = sa3_model.generate(prompt="drums", ...)
            maybe_save_audio(audio, sr, "drums")

    Files are written to test_audio_outputs/{test_name[param]}_{prompt_slug}.wav.
    Does nothing when --save-audio is not set.
    """
    enabled = request.config.getoption("--save-audio")

    def _save(audio: torch.Tensor, sample_rate: int, prompt: str) -> None:
        if not enabled:
            return
        out_dir = Path("test_audio_outputs")
        out_dir.mkdir(exist_ok=True)
        slug = re.sub(r"[^\w]+", "_", prompt).strip("_")[:40] or "audio"
        test_name = request.node.name  # e.g. test_text_to_audio[small]
        filename = out_dir / f"{test_name}_{slug}.wav"
        torchaudio.save(str(filename), audio.squeeze(0).cpu(), sample_rate)

    return _save


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_ACCEL, reason="Flash attention check requires a GPU/accelerator"
)
def test_flash_attention_available(sa3_model, request):
    """Verify flash_attn is importable on GPU environments (medium model only)."""
    if request.node.callspec.params.get("sa3_model") != "medium":
        pytest.skip("Flash attention check is medium-model only")

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        pytest.fail(
            "flash_attn is not installed. Install via: uv sync --extra flash-attn"
        )
