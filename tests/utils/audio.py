import math

import torch

AMPLITUDE = 0.8


def sine_wave(
    duration: float,
    sample_rate: int,
    channels: int = 2,
    freq: float = 440.0,
    device: str = "cpu",
    half: bool = False,
) -> torch.Tensor:
    """Return a [channels, samples] sine-wave tensor in [-AMPLITUDE, AMPLITUDE]."""
    t = torch.linspace(0, duration, int(duration * sample_rate), device=device)
    wave = AMPLITUDE * torch.sin(2 * math.pi * freq * t)
    if half:
        wave = wave.half()
    return wave.expand(channels, -1).clone()


def assert_audio_valid(
    audio: torch.Tensor, expected_duration: float, sample_rate: int
) -> None:
    """Assert that an audio tensor looks like valid, non-trivial audio output.

    Args:
        audio: Output tensor of shape [B, C, T].
        expected_duration: Expected length in seconds.
        sample_rate: Sample rate in Hz.
    """
    assert audio is not None, "Audio output is None"
    assert audio.ndim == 3, f"Expected 3D tensor [B, C, T], got shape {audio.shape}"

    expected_samples = int(expected_duration * sample_rate)
    actual_samples = audio.shape[-1]
    tolerance = 0.15
    relative_error = abs(actual_samples - expected_samples) / expected_samples
    assert relative_error < tolerance, (
        f"Audio length {actual_samples} samples deviates more than {tolerance * 100:.0f}% "
        f"from expected {expected_samples} samples ({expected_duration}s @ {sample_rate}Hz)"
    )

    assert not torch.isnan(audio).any(), "Audio output contains NaN values"

    abs_max = audio.abs().max().item()
    assert abs_max > 0.001, f"Audio output appears silent (abs_max={abs_max:.5f})"
    assert abs_max <= 1.0, f"Audio output is clipping (abs_max={abs_max:.5f})"
