import re
from pathlib import Path

import pytest
import torch
import torchaudio

from stable_audio_3 import StableAudioPipeline

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


@pytest.fixture(scope="session", params=["small", "medium"])
def model_pipe(request):
    """Session-scoped pipeline fixture parametrized over model sizes.

    small  — loads via from_pretrained("small"); runs on CPU or accelerator.
    medium — requires a GPU/accelerator; skipped otherwise.
             Update from_pretrained to point at the HuggingFace medium checkpoint
             when it becomes available.
    """
    name = request.param

    if name == "small":
        return StableAudioPipeline.from_pretrained("small")

    if name == "medium":
        if not HAS_ACCEL:
            pytest.skip("Medium model requires a GPU/accelerator — none detected")
        # TODO: wire up once medium checkpoint is available on HuggingFace
        # pytest.skip(
        #     "Medium model not yet configured — update from_pretrained call when ready"
        # )
        return StableAudioPipeline.from_pretrained("medium")


@pytest.fixture
def maybe_save_audio(request):
    """Return a callable that saves audio to disk when --save-audio is passed.

    Usage in tests:
        def test_foo(model_pipe, maybe_save_audio):
            audio = pipe.generate(prompt="drums", ...)
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
def test_flash_attention_available(model_pipe, request):
    """Verify flash_attn is importable on GPU environments (medium model only)."""
    if request.node.callspec.params.get("model_pipe") != "medium":
        pytest.skip("Flash attention check is medium-model only")

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        pytest.fail(
            "flash_attn is not installed. Install via: uv sync --extra flash-attn"
        )
