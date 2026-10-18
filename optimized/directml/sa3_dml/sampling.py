"""Rectified-flow pingpong sampler, timestep schedule and guidance (host-side, float32)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .specs import DOWNSAMPLING_RATIO, DURATION_PADDING_SEC, LATENT_ALIGN, SAMPLE_RATE

LOGSNR_ANCHOR = -6.2
LOGSNR_END = 2.0


def latent_length(seconds: float, sample_size: int, padding: float = DURATION_PADDING_SEC) -> int:
    """Latent frames of the reference sample: duration plus headroom, aligned and capped."""
    target = int((seconds + padding) * SAMPLE_RATE)
    target = -(-target // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO
    align = DOWNSAMPLING_RATIO * LATENT_ALIGN
    target = -(-target // align) * align
    return min(target, sample_size) // DOWNSAMPLING_RATIO


def valid_length(seconds: float, total: int, padding: float = DURATION_PADDING_SEC) -> int:
    """Unmasked latent frames of the reference sample.

    The reference sampler excludes frames past this length from self-attention, and every other
    DiT layer is per-frame, so sampling exactly these frames gives the same result.
    """
    effective = -(-int(seconds * SAMPLE_RATE) // DOWNSAMPLING_RATIO)
    return min(effective + int(padding * SAMPLE_RATE / DOWNSAMPLING_RATIO), total)


def keep_mask(length: int, regions: Sequence[tuple[float, float]], duration: float) -> np.ndarray:
    """Inpainting mask [1, 1, length] of the reference: 1 keeps the source latent, 0 regenerates it.

    Regions are cut at sample resolution and everything past `duration` is regenerated; latent
    frame i takes the value at sample i * 4096 (nearest-neighbour downsampling).
    """
    positions = np.arange(length, dtype=np.int64) * DOWNSAMPLING_RATIO
    total = length * DOWNSAMPLING_RATIO
    keep = np.ones(length, np.float32)
    for start, end in regions:
        lo, hi = min(int(start * SAMPLE_RATE), total), min(int(end * SAMPLE_RATE), total)
        keep[(positions >= lo) & (positions < hi)] = 0.0
    keep[positions >= int(duration * SAMPLE_RATE)] = 0.0
    return keep.reshape(1, 1, length)


def schedule(steps: int, sigma_max: float = 1.0) -> np.ndarray:
    """Linear grid warped uniformly in log-SNR (sequence-length invariant), first point at sigma_max."""
    t = np.linspace(sigma_max, 0.0, steps + 1, dtype=np.float32)
    logsnr = LOGSNR_END - t * (LOGSNR_END - LOGSNR_ANCHOR)
    out = (1.0 / (1.0 + np.exp(logsnr))).astype(np.float32)
    out = np.where(t <= 0, 0.0, np.where(t >= 1, 1.0, out)).astype(np.float32)
    out[0] = sigma_max
    return out


def guided_velocity(x: np.ndarray, sigma: float, v_cond: np.ndarray, v_uncond: np.ndarray,
                    cfg_scale: float, apg_scale: float) -> np.ndarray:
    """Classifier-free guidance in denoised space with adaptive projected guidance."""
    cond = x - v_cond * sigma
    uncond = x - v_uncond * sigma
    diff = cond - uncond
    if apg_scale != 0.0:
        unit = cond / max(float(np.sqrt((cond * cond).sum())), 1e-8)
        orthogonal = diff - (diff * unit).sum() * unit
        diff = orthogonal if apg_scale == 1.0 else apg_scale * orthogonal + (1 - apg_scale) * diff
    guided = cond + (cfg_scale - 1.0) * diff
    return (x - guided) / sigma


def pingpong(velocity: Callable[[np.ndarray, float], np.ndarray], noise: np.ndarray,
             sigmas: np.ndarray, rng: np.random.Generator,
             on_step: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Denoise at each sigma, then re-noise to the next one; the last step returns the denoised latent."""
    x = noise
    steps = len(sigmas) - 1
    for i in range(steps):
        t_cur, t_next = float(sigmas[i]), float(sigmas[i + 1])
        denoised = x - t_cur * velocity(x, t_cur)
        if t_next > 0:
            x = (1.0 - t_next) * denoised + t_next * rng.standard_normal(x.shape, dtype=np.float32)
        else:
            x = denoised
        if on_step is not None:
            on_step(i + 1, steps)
    return x.astype(np.float32)
