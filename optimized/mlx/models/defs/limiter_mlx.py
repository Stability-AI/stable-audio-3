"""Output limiter in MLX — the shipped SA3 limiter (`dsp.limit_ship`) for the Apple-Silicon backend.

Same setting as the TensorRT graft and the TFLite baked limiter: **sample-peak detector, ceiling
0.977 (-0.2021 dBFS), 5.8 ms window** (LIMITER_SPEC.md §1). Applied to the decoder's float audio
output on-device (before the int16 conversion), so the MLX backend emits already-limited audio.

Algorithm (§2, factor=1): channel-linked sample-peak envelope -> block MAX decimate by D=64 ->
required gain min(1, ceiling/env) -> running MIN over 2*Ld+1=9 taps (the brickwall) -> Hann smooth
-> half-pixel linear upsample -> multiply. At factor=1 there is no polyphase conv — only abs/max/
reshape/min/gather, all native MLX ops.

⚠ NOT run on the repo's CI box (MLX needs Apple Silicon / Metal). The `limit_numpy` twin below uses
the identical op sequence and is verified against `dsp.limit_ship` to max|Δ|=3.0e-7 across lengths;
the MLX ops mirror it 1:1, so it should match on-device — but run `limit_numpy`-vs-`limit_mlx` and an
ear check on a Mac before trusting it end to end.
"""
import math
import numpy as np

CEILING = 0.977      # -0.2021 dBFS sample-peak ceiling
D = 64               # gain decimation
LD = 4               # ceil(5.0ms * 44100 / D) -> 5.80 ms half-window


def _hann(n):
    w = np.hanning(n).astype(np.float32)
    return w / w.sum()


def _upsample_index(T, Tb):
    """Half-pixel linear-upsample indices/weights (== F.interpolate align_corners=False)."""
    src = (np.arange(T, dtype=np.float32) + 0.5) / D - 0.5
    i0 = np.clip(np.floor(src), 0, Tb - 1).astype(np.int32)
    frac = np.clip(src - i0, 0.0, 1.0).astype(np.float32)
    i1 = np.minimum(i0 + 1, Tb - 1).astype(np.int32)
    return i0, i1, frac


def limit_mlx(x, ceiling=CEILING):
    """x: mx.array (2, T) float32. Returns (2, T) with |y| <= ceiling. ceiling=inf bypasses exactly."""
    import mlx.core as mx
    if not math.isfinite(ceiling) or ceiling > 1e6:
        return x
    C, T = x.shape
    Tb = (T + D - 1) // D
    env = mx.max(mx.abs(x), axis=0)                                       # (T,) channel-linked sample peak
    pad = Tb * D - T
    if pad:
        env = mx.concatenate([env, mx.repeat(env[-1:], pad)])            # replicate pad to a whole block
    envd = mx.max(env.reshape(Tb, D), axis=1)                            # (Tb,) block MAX
    g = mx.minimum(ceiling / mx.maximum(envd, 1e-12), 1.0)
    gp = mx.concatenate([mx.ones((LD,), g.dtype), g, mx.ones((LD,), g.dtype)])   # running MIN, pad gain 1.0
    gmin = gp[LD:LD + Tb]
    for s in range(1, LD + 1):
        gmin = mx.minimum(gmin, mx.minimum(gp[LD - s:LD - s + Tb], gp[LD + s:LD + s + Tb]))
    hw = _hann(2 * LD + 1)                                               # Hann smooth as shifted weighted sum
    gpad = mx.concatenate([mx.repeat(gmin[:1], LD), gmin, mx.repeat(gmin[-1:], LD)])
    gs = mx.zeros((Tb,), g.dtype)
    for j in range(2 * LD + 1):
        gs = gs + float(hw[j]) * gpad[j:j + Tb]
    i0, i1, frac = _upsample_index(T, Tb)                                # half-pixel linear upsample (gather + lerp)
    gu = mx.take(gs, mx.array(i0)) * (1.0 - mx.array(frac)) + mx.take(gs, mx.array(i1)) * mx.array(frac)
    return x * mx.minimum(gu, 1.0)[None, :]


def limit_numpy(x, ceiling=CEILING):
    """Pure-numpy twin — host fallback and the tested reference (verified vs dsp.limit_ship to 3e-7)."""
    if not math.isfinite(ceiling) or ceiling > 1e6:
        return x.astype(np.float32)
    x = np.asarray(x, np.float32)
    C, T = x.shape
    Tb = (T + D - 1) // D
    env = np.abs(x).max(0)
    pad = Tb * D - T
    if pad:
        env = np.concatenate([env, np.repeat(env[-1:], pad)])
    envd = env.reshape(Tb, D).max(1)
    g = np.minimum(ceiling / np.maximum(envd, 1e-12), 1.0)
    gp = np.concatenate([np.ones(LD, np.float32), g, np.ones(LD, np.float32)])
    gmin = gp[LD:LD + Tb].copy()
    for s in range(1, LD + 1):
        gmin = np.minimum(gmin, np.minimum(gp[LD - s:LD - s + Tb], gp[LD + s:LD + s + Tb]))
    hw = _hann(2 * LD + 1)
    gpad = np.concatenate([np.repeat(gmin[:1], LD), gmin, np.repeat(gmin[-1:], LD)])
    gs = np.zeros(Tb, np.float32)
    for j in range(2 * LD + 1):
        gs = gs + float(hw[j]) * gpad[j:j + Tb]
    i0, i1, frac = _upsample_index(T, Tb)
    gu = gs[i0] * (1.0 - frac) + gs[i1] * frac
    return (x * np.minimum(gu, 1.0)).astype(np.float32)


if __name__ == "__main__":
    # Run on a Mac: checks limit_mlx == limit_numpy (the reference verified vs dsp.limit_ship to 3e-7),
    # that the ceiling holds, and prints on-device timing. `python models/defs/limiter_mlx.py`
    import time
    try:
        import mlx.core as mx
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"MLX unavailable ({e}) — this self-check needs Apple Silicon.")
    rng = np.random.RandomState(0)
    worst = 0.0
    for T in (4096 * 8, 4096 * 128, 4096 * 323):
        x = (rng.standard_normal((2, T)) * 1.6).astype(np.float32)
        ref = limit_numpy(x)
        y = np.array(limit_mlx(mx.array(x)).astype(mx.float32))
        d = float(np.abs(y - ref).max()); worst = max(worst, d)
        t0 = time.time(); _ = np.array(limit_mlx(mx.array(x)).astype(mx.float32)); dt = (time.time() - t0) * 1e3
        print(f"  T={T:7d}  peak={np.abs(y).max():.4f}  |mlx-numpy|={d:.3e}  {dt:.2f} ms")
    print(f"PARITY {'OK' if worst < 1e-4 else 'DIFF %.2e' % worst}  (ceiling {CEILING})")

