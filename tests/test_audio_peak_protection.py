import wave
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch

from optimized.mlx.scripts.wav_io import protect_audio_peak as protect_numpy_peak
from optimized.mlx.scripts.wav_io import save_wav
from stable_audio_3.audio_output import (
    PCM16_CEILING,
    audio_peak,
    protect_audio_peak,
)
from stable_audio_3.model import StableAudioModel


def test_torch_peak_protection_attenuates_batch_items_independently():
    audio = torch.tensor(
        [
            [[0.0, 0.5, 1.0], [0.0, -0.5, -1.0]],
            [[0.0, 1.25, 1.75], [0.0, -1.25, -1.75]],
        ],
        dtype=torch.float32,
    )

    with pytest.warns(RuntimeWarning, match="no-boost attenuation"):
        protected = protect_audio_peak(audio, batch_dim=0)

    assert torch.equal(protected[0], audio[0])
    assert protected[1].abs().max() == 1.0
    ratio = (protected[1, 0, 1] / protected[1, 0, 2]).item()
    assert ratio == pytest.approx(1.25 / 1.75)


def test_torch_peak_protection_rejects_non_finite_audio():
    with pytest.raises(RuntimeError, match="1 non-finite"):
        protect_audio_peak(torch.tensor([0.0, torch.nan]))


def test_torch_peak_protection_supports_unbounded_int32_pcm():
    pcm = torch.tensor([[0, 40959, 57342], [0, -40959, -57342]], dtype=torch.int32)

    with pytest.warns(RuntimeWarning, match="peak 57342.000"):
        protected = protect_audio_peak(pcm, ceiling=PCM16_CEILING)

    narrowed = protected.to(torch.int16)
    assert narrowed.abs().max() == 32767
    ratio = (narrowed[0, 1].float() / narrowed[0, 2].float()).item()
    assert ratio == pytest.approx(40959 / 57342, abs=1e-4)


def test_torch_peak_protection_has_capture_safe_branchless_mode():
    audio = torch.tensor([0.0, 1.25, 1.75])
    peak = audio_peak(audio)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        protected = protect_audio_peak(
            audio,
            peak=peak,
            validate_nonfinite=False,
            emit_warning=False,
        )

    assert protected.max() == 1.0
    assert (protected[1] / protected[2]).item() == pytest.approx(1.25 / 1.75)


def test_numpy_peak_protection_does_not_boost_quiet_audio():
    audio = np.array([[0.0, 0.25, -0.75]], dtype=np.float32)

    protected = protect_numpy_peak(audio)

    assert protected is audio


def test_mlx_wav_serializer_attenuates_instead_of_clipping(tmp_path):
    audio = np.array(
        [
            [0.0, 0.5, 1.0, 1.25, 1.75],
            [0.0, -0.5, -1.0, -1.25, -1.75],
        ],
        dtype=np.float32,
    )
    output = tmp_path / "out.wav"

    with pytest.warns(RuntimeWarning, match="peak 1.750"):
        save_wav(str(output), audio, 44100)

    with wave.open(str(output), "rb") as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        pcm = pcm.reshape(-1, wav.getnchannels()).T

    assert pcm[0, -1] == 32767
    assert pcm[0, -2] < pcm[0, -1]
    assert pcm[0, -2] == pytest.approx(32767 * 1.25 / 1.75, abs=1)
    assert pcm[1, -2] == pytest.approx(-32767 * 1.25 / 1.75, abs=1)


class _FakePipeline:
    sample_rate = 1
    io_channels = 2
    pretransform = None
    diffusion_objective = None
    sampling_dist_shift = None

    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)

    @staticmethod
    def get_conditioning_inputs(_conditioning, negative=False):
        return {}


@pytest.mark.parametrize("discarded_tail", [10.0, float("nan")])
def test_generate_trims_decoder_padding_before_peak_protection(discarded_tail):
    model = StableAudioModel.__new__(StableAudioModel)
    model.model = _FakePipeline()
    model.device = "cpu"
    decoded = torch.tensor(
        [[[0.5, -0.5, discarded_tail], [0.25, -0.25, discarded_tail]]]
    )

    with (
        patch("stable_audio_3.model.sample_diffusion", return_value=decoded),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        result = model.generate(
            conditioning_tensors={},
            duration=2,
            sample_size=3,
            batch_size=1,
        )

    assert torch.equal(result, decoded[..., :2])
