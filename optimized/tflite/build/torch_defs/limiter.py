"""Output limiter, baked into the SAME decoder tflite graph.

This is the shipped SA3 limiter (`dsp.limit_ship` == `dsp.limit_edge(factor=1)` from the
limiter spec) as a torch module, so ai_edge_torch converts it into the decoder graph and the
tflite decoder emits already-limited float audio (peak <= ceiling), matching the TensorRT graft.

Algorithm (LIMITER_SPEC.md §2, sample-peak / factor=1 config, decided 2026-08-20):
  channel-linked sample-peak envelope -> block MAX decimate by D=64 -> required gain
  min(1, ceiling/env) -> running MIN over 2*Ld+1=9 taps (the brickwall) -> Hann smooth ->
  linear upsample -> multiply. ceiling 0.977 (-0.2021 dBFS), Ld=4 (5.80 ms half-window).

Baked per-rung: each fixed-size decode window limits its own chunk. Because the rung tiling
keeps an overlap of TRIM latents (>= 12 -> >= 49152 samples) which is block-aligned and far
exceeds the limiter's 512-sample dependency radius (SPEC §10), per-rung limiting is bit-exact
to whole-signal limiting in the center-stitched (kept) region. T is always a multiple of D
(T = latents * 4096 = latents * 64 * D), so no block padding is needed.

Verified: torch module == dsp.limit_ship (max|d|=0); tflite via ai_edge_torch == dsp.limit_ship
(max|d|=1.8e-7, well under the spec's 1e-4 acceptance bar).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CEILING = 0.977          # -0.2021 dBFS, sample-peak ceiling (LIMITER_SPEC.md §1)
D = 64                   # gain decimation: computed at sr/64, linearly upsampled
LD = 4                   # ceil(5.0ms * 44100 / D) -> 5.80 ms half-window


class OutputLimiter(nn.Module):
    """x: [1, 2, T] raw decoder audio (may exceed +-1), T % 64 == 0. Returns x * g, |y| <= ceiling."""

    def __init__(self, ceiling: float = CEILING, d: int = D, ld: int = LD):
        super().__init__()
        self.ceiling, self.d, self.ld = float(ceiling), int(d), int(ld)
        w = torch.hann_window(2 * ld + 1, periodic=False)
        self.register_buffer("w", (w / w.sum()).view(1, 1, -1))

    def forward(self, x):
        d, ld, c = self.d, self.ld, self.ceiling
        T = x.shape[-1]
        env = x.abs().amax(1, keepdim=True)                       # [1,1,T] channel-linked sample peak
        envd = F.max_pool1d(env, d, stride=d)                     # [1,1,T/D] block MAX (never under-reads)
        g_req = torch.clamp(c / envd.clamp_min(1e-12), max=1.0)
        g_min = -F.max_pool1d(-g_req, 2 * ld + 1, stride=1, padding=ld)   # running MIN = brickwall
        gd = F.conv1d(F.pad(g_min, (ld, ld), mode="replicate"), self.w)   # Hann smooth (edge pad)
        # Linear upsample via a 2D bilinear resize -> ai_edge_torch emits a native RESIZE_BILINEAR.
        # On fixed-size rungs the output size is static, so litert delegates it to XNNPACK (fast) and
        # nothing ∝T is baked as a constant. A 1D F.interpolate(mode="linear") instead lowers to a
        # gather + per-output index/weight arrays of length T (~21 MB baked per big rung). Same result
        # to ~2e-7, ~21 MB smaller/rung, and slightly faster.
        g = F.interpolate(gd.unsqueeze(2), size=(1, (T // d) * d),
                          mode="bilinear", align_corners=False).squeeze(2)
        return x * g.clamp(max=1.0)


class LimitedDecoder(nn.Module):
    """Wraps a decoder (latent -> float audio) so its output passes through the baked limiter."""

    def __init__(self, decoder, ceiling: float = CEILING):
        super().__init__()
        self.decoder = decoder
        self.limiter = OutputLimiter(ceiling)

    def forward(self, z):
        return self.limiter(self.decoder(z))


def maybe_wrap(decoder):
    """Wrap `decoder` with the limiter unless SA3_BAKE_LIMITER=0 (env). Default: baked in."""
    import os
    if os.environ.get("SA3_BAKE_LIMITER", "1") == "0":
        return decoder
    return LimitedDecoder(decoder)
