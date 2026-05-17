import pytest
import torch
from functools import partial

from stable_audio_3 import StableAudioModel
from stable_audio_3.models.lora import (
    LoRAParametrization,
    add_lora,
    get_lora_state_dict,
    save_lora_safetensors,
)
from tests.conftest import ACCEL_DEVICE

_LORA_RANK = 4
_LORA_ALPHA = 4.0


def _lora_cfg(rank, alpha):
    return {
        torch.nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=alpha,
                adapter_type="lora",
            ),
        },
        torch.nn.Conv1d: {
            "weight": partial(
                LoRAParametrization.from_conv1d,
                rank=rank,
                lora_alpha=alpha,
                adapter_type="lora",
            ),
        },
    }


@pytest.fixture(scope="session")
def synthetic_lora_ckpt(tmp_path_factory):
    """Build and save a minimal LoRA checkpoint from the small-music model for testing."""
    model = StableAudioModel.from_pretrained("small-music", device=ACCEL_DEVICE)
    add_lora(model.model.model, _lora_cfg(_LORA_RANK, _LORA_ALPHA))
    add_lora(model.model.conditioner, _lora_cfg(_LORA_RANK, _LORA_ALPHA))
    state_dict = {
        **get_lora_state_dict(model.model.model),
        **get_lora_state_dict(model.model.conditioner),
    }
    config = {"rank": _LORA_RANK, "alpha": _LORA_ALPHA, "adapter_type": "lora"}
    ckpt_path = tmp_path_factory.mktemp("lora") / "synthetic.safetensors"
    save_lora_safetensors(state_dict, config, ckpt_path)
    del model
    return ckpt_path


@pytest.fixture(scope="session")
def lora_model(synthetic_lora_ckpt):
    """Small-music model with a synthetic LoRA checkpoint loaded."""
    model = StableAudioModel.from_pretrained("small-music", device=ACCEL_DEVICE)
    model.load_lora([str(synthetic_lora_ckpt)])
    return model


def test_lora_inference(lora_model, maybe_save_audio):
    audio = lora_model.generate(prompt="drums", duration=1.0, steps=4, seed=42)

    assert isinstance(audio, torch.Tensor)
    assert audio.ndim == 3  # (batch, channels, samples)
    assert audio.shape[0] == 1
    assert audio.shape[-1] > 0

    maybe_save_audio(audio, lora_model.model.sample_rate, "lora_drums")
