"""Backend-neutral peak protection and PCM16 WAV serialization."""

from __future__ import annotations

import math
import warnings
import wave
from typing import Any


PCM_FLOAT_CEILING = 1.0
PCM16_CEILING = 32767.0


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
    batch_dim %= audio.ndim
    return tuple(dim for dim in range(audio.ndim) if dim != batch_dim)


def audio_peak(audio: Any, batch_dim: int | None = None) -> Any:
    """Return the absolute peak, optionally reduced independently per batch item."""
    backend = _backend_name(audio)
    reduce_dims = _reduce_dims(audio, batch_dim)
    if backend == "numpy":
        import numpy as np

        if audio.size == 0:
            return 0.0
        if reduce_dims is None:
            return np.abs(audio).max()
        return np.abs(audio).max(axis=reduce_dims, keepdims=True)

    if audio.numel() == 0:
        return 0.0
    if reduce_dims is None:
        return audio.abs().amax()
    return audio.abs().amax(dim=reduce_dims, keepdim=True)


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
            finite = np.isfinite(audio)
            if not finite.all():
                n_bad = int((~finite).sum())
                raise RuntimeError(
                    f"refusing to process audio containing {n_bad} non-finite samples (NaN/Inf)"
                )
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

    import torch

    if validate_nonfinite:
        finite = torch.isfinite(audio)
        if not finite.all():
            n_bad = int((~finite).sum().item())
            raise RuntimeError(
                f"refusing to process audio containing {n_bad} non-finite samples (NaN/Inf)"
            )
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


def save_wav(
    path: str,
    audio: Any,
    sample_rate: int = 44100,
) -> None:
    """Write channel-first NumPy floating-point audio as 16-bit PCM WAV."""
    if _backend_name(audio) != "numpy":
        raise TypeError("save_wav expects a channel-first numpy.ndarray")

    import numpy as np

    audio = protect_audio_peak(audio)
    pcm = (audio * PCM16_CEILING).astype(np.int16).T
    with wave.open(path, "wb") as wav:
        wav.setnchannels(audio.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
