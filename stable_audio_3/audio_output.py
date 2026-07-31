"""Backend-neutral peak protection and PCM16 WAV serialization."""

from __future__ import annotations

import math
import warnings
import wave
from typing import Any, Literal


PCM_FLOAT_CEILING = 1.0
PCM16_CEILING = 32767.0
OutputPeakPolicy = Literal["attenuate", "raw"]


def dbfs_to_amplitude(ceiling_dbfs: float) -> float:
    """Convert a finite, non-positive dBFS ceiling to linear amplitude."""
    if not math.isfinite(ceiling_dbfs):
        raise ValueError(f"ceiling_dbfs must be finite, got {ceiling_dbfs}")
    if ceiling_dbfs > 0:
        raise ValueError(f"ceiling_dbfs must be <= 0, got {ceiling_dbfs}")
    amplitude = 10.0 ** (ceiling_dbfs / 20.0)
    if amplitude == 0.0:
        raise ValueError(f"ceiling_dbfs is too small to represent, got {ceiling_dbfs}")
    return amplitude


def _backend_name(audio: Any) -> str:
    module = type(audio).__module__.partition(".")[0]
    if module in {"numpy", "torch"}:
        return module
    raise TypeError(
        "audio must be a numpy.ndarray or torch.Tensor, "
        f"got {type(audio).__module__}.{type(audio).__qualname__}"
    )


def _reduce_dims(audio: Any, batch_dim: int | None) -> tuple[int, ...] | None:
    if batch_dim is None:
        return None
    if audio.ndim == 0:
        raise ValueError("batch_dim cannot be used with scalar audio")
    if not -audio.ndim <= batch_dim < audio.ndim:
        raise ValueError(
            f"batch_dim must be in [-{audio.ndim}, {audio.ndim - 1}], got {batch_dim}"
        )
    batch_dim %= audio.ndim
    return tuple(dim for dim in range(audio.ndim) if dim != batch_dim)


def _absolute_audio(audio: Any, backend: str) -> Any:
    """Take an overflow-safe absolute value, promoting integer PCM first."""
    if backend == "numpy":
        import numpy as np

        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float64)
        return np.abs(audio)

    import torch

    if not audio.dtype.is_floating_point and not audio.dtype.is_complex:
        audio = audio.to(torch.float64 if audio.dtype == torch.int64 else torch.float32)
    return audio.abs()


def _validate_finite(audio: Any, backend: str) -> None:
    if backend == "numpy":
        import numpy as np

        finite = np.isfinite(audio)
        if finite.all():
            return
        n_bad = int((~finite).sum())
    else:
        import torch

        finite = torch.isfinite(audio)
        if bool(finite.all()):
            return
        n_bad = int((~finite).sum().item())
    raise RuntimeError(
        f"refusing to process audio containing {n_bad} non-finite samples (NaN/Inf)"
    )


def audio_peak(audio: Any, batch_dim: int | None = None) -> Any:
    """Return the absolute peak, optionally reduced independently per batch item."""
    backend = _backend_name(audio)
    reduce_dims = _reduce_dims(audio, batch_dim)
    if backend == "numpy":
        if audio.size == 0:
            return 0.0
        absolute = _absolute_audio(audio, backend)
        if reduce_dims is None:
            return absolute.max()
        return absolute.max(axis=reduce_dims, keepdims=True)

    if audio.numel() == 0:
        return 0.0
    absolute = _absolute_audio(audio, backend)
    if reduce_dims is None:
        return absolute.amax()
    return absolute.amax(dim=reduce_dims, keepdim=True)


def report_peak_protection(
    peak: float,
    ceiling: float = PCM_FLOAT_CEILING,
    *,
    n_affected: int = 1,
    stacklevel: int = 2,
) -> None:
    """Raise for a non-finite peak or warn when attenuation is required."""
    if not math.isfinite(peak):
        raise RuntimeError("refusing to process audio with a non-finite peak (NaN/Inf)")
    if peak <= ceiling:
        return

    item_label = "item" if n_affected == 1 else "items"
    warnings.warn(
        f"audio peak {peak:.3f} exceeds the {ceiling:.3f} PCM ceiling; "
        f"applying no-boost attenuation to {n_affected} {item_label} to prevent clipping",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


def protect_audio_peak(
    audio: Any,
    ceiling: float = PCM_FLOAT_CEILING,
    batch_dim: int | None = None,
    *,
    peak: Any | None = None,
    validate_nonfinite: bool = True,
    emit_warning: bool = True,
) -> Any:
    """Attenuate out-of-range audio without boosting or hard clipping it.

    NumPy arrays and Torch tensors are supported. When ``batch_dim`` is
    provided, each batch item is attenuated independently. Passing both
    ``validate_nonfinite=False`` and ``emit_warning=False`` avoids host-side
    branching, which makes the Torch path safe to capture in a CUDA graph.
    """
    if ceiling <= 0:
        raise ValueError(f"ceiling must be positive, got {ceiling}")

    backend = _backend_name(audio)
    is_empty = audio.size == 0 if backend == "numpy" else audio.numel() == 0
    if is_empty:
        return audio

    if backend == "numpy":
        import numpy as np

        if validate_nonfinite:
            _validate_finite(audio, backend)
        peaks = audio_peak(audio, batch_dim) if peak is None else peak
        over_ceiling = peaks > ceiling
        has_over_ceiling = (
            bool(np.any(over_ceiling)) if (validate_nonfinite or emit_warning) else None
        )
        if emit_warning and has_over_ceiling:
            report_peak_protection(
                float(np.max(peaks)),
                ceiling,
                n_affected=int(np.count_nonzero(over_ceiling)),
                stacklevel=3,
            )
        if has_over_ceiling is False:
            return audio
        scale = np.maximum(peaks / ceiling, 1.0)
        return audio / scale

    if validate_nonfinite:
        _validate_finite(audio, backend)
    peaks = audio_peak(audio, batch_dim) if peak is None else peak
    over_ceiling = peaks > ceiling
    has_over_ceiling = (
        bool(over_ceiling.any()) if (validate_nonfinite or emit_warning) else None
    )
    if emit_warning and has_over_ceiling:
        report_peak_protection(
            float(peaks.max().item()),
            ceiling,
            n_affected=int(over_ceiling.sum().item()),
            stacklevel=3,
        )
    if has_over_ceiling is False:
        return audio
    scale = (peaks / ceiling).clamp(min=1.0)
    return audio / scale


def apply_output_peak_policy(
    audio: Any,
    policy: OutputPeakPolicy = "attenuate",
    *,
    ceiling_dbfs: float = 0.0,
    batch_dim: int | None = None,
    emit_warning: bool = True,
) -> Any:
    """Apply the public output policy while always rejecting non-finite audio."""
    if policy not in {"attenuate", "raw"}:
        raise ValueError(
            f"output peak policy must be 'attenuate' or 'raw', got {policy!r}"
        )
    backend = _backend_name(audio)
    if policy == "raw":
        _validate_finite(audio, backend)
        return audio
    return protect_audio_peak(
        audio,
        ceiling=dbfs_to_amplitude(ceiling_dbfs),
        batch_dim=batch_dim,
        validate_nonfinite=True,
        emit_warning=emit_warning,
    )


def zero_audio_padding_(audio: Any, valid_sample_mask: Any, sample_dim: int) -> Any:
    """Zero invalid samples in-place using a 1-D boolean validity mask."""
    backend = _backend_name(audio)
    if _backend_name(valid_sample_mask) != backend:
        raise TypeError("audio and valid_sample_mask must use the same backend")
    if not -audio.ndim <= sample_dim < audio.ndim:
        raise ValueError(
            f"sample_dim must be in [-{audio.ndim}, {audio.ndim - 1}], got {sample_dim}"
        )
    sample_dim %= audio.ndim
    if valid_sample_mask.ndim != 1:
        raise ValueError("valid_sample_mask must be one-dimensional")
    if valid_sample_mask.shape[0] != audio.shape[sample_dim]:
        raise ValueError(
            "valid_sample_mask length must equal the audio sample dimension"
        )

    mask_shape = [1] * audio.ndim
    mask_shape[sample_dim] = valid_sample_mask.shape[0]
    valid_sample_mask = valid_sample_mask.reshape(mask_shape)
    if backend == "numpy":
        import numpy as np

        np.copyto(audio, 0, where=~valid_sample_mask)
    else:
        audio.masked_fill_(~valid_sample_mask, 0)
    return audio


def save_wav(
    path: str,
    audio: Any,
    sample_rate: int = 44100,
    *,
    peak_ceiling_dbfs: float = 0.0,
) -> None:
    """Write channel-first NumPy floating-point audio as 16-bit PCM WAV."""
    if _backend_name(audio) != "numpy":
        raise TypeError("save_wav expects a channel-first numpy.ndarray")

    import numpy as np

    audio = protect_audio_peak(audio, ceiling=dbfs_to_amplitude(peak_ceiling_dbfs))
    pcm = (audio * PCM16_CEILING).astype(np.int16).T
    with wave.open(path, "wb") as wav:
        wav.setnchannels(audio.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
