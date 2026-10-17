#!/usr/bin/env python3
"""SAME-S Decoder — PyTorch (for CoreML export).

Mirrors `SAMEDecoder` + decoder-mode `TransformerResamplingBlock` + `TransformerBlock`
from stable-audio-tools-dev for the sa3-sm-music config:

  channels=128, c_mults=[6], strides=[16], transformer_depths=[6],
  dim_heads=64, latent_dim=256, out_channels=512,
  differential=True, dyt=True, variable_stride=True,
  chunk_size=32, chunk_midpoint_shift=True,
  conv_mapping=True, sinusoidal_blocks=[0]

This is a clean rewrite optimized for `torch.jit.trace` → CoreML:
  - static shapes (single-token sinusoidal broadcast, fixed effective_chunk=34)
  - plain F.scaled_dot_product_attention (no flash/flex/chunk-halo fallback ladder)
  - precomputed RoPE cos/sin buffers for the 34-token chunk
  - running_std absorbed as a scalar buffer (softnorm bottleneck.decode)
  - WNConv1d output mapping with weight_norm pre-fused at load time

Input:  [B, 256, T_lat]    channels-first latents from softnorm space
Output: [B, 512, T_lat*16] channels-first audio patches (still needs unpatching),
        OR [B, 2, T_lat*4096] finished stereo audio when output_audio=True (unpatch
        baked in-graph — matches the ONNX decoder convention: self-contained).
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture constants ─────────────────────────────────────────────
LATENT_DIM       = 256
DIM              = 768            # backbone width
NUM_HEADS        = 12
HEAD_DIM         = 64
ROPE_DIMS        = 32
NUM_BLOCKS       = 6
FF_INNER         = 2304           # ff_mult=3, inner=DIM*3=2304
FF_PROJ_OUT      = FF_INNER * 2   # GLU input projection: 2 * inner
OUT_CHANNELS     = 512            # patched audio channels (256 * 2 stereo)
STRIDE           = 16
SUB_CHUNK_SIZE   = STRIDE + 1     # 17 = 1 latent + 16 new-token positions
CHUNK_SIZE_LAT   = 32             # in latent-frame space
EFFECTIVE_CHUNK  = CHUNK_SIZE_LAT + CHUNK_SIZE_LAT // STRIDE  # 32 + 2 = 34
SHIFT            = EFFECTIVE_CHUNK // 2                       # 17
PAD_MODULO       = CHUNK_SIZE_LAT // STRIDE                   # 2
SIN_PER_POS      = SUB_CHUNK_SIZE - 1                         # 16 new tokens per latent
QKV_NORM_EPS     = 1e-3
DYT_NORM_EPS     = 1e-3           # unused; DyT has no eps but keep for parity


# ── RoPE ────────────────────────────────────────────────────────────────

def _build_rope_cache(seq_len: int, dim: int = ROPE_DIMS, base: float = 10000.0):
    """Cos/sin tables for RoPE applied to first `dim` of head_dim.

    Half-half pairing (MLX/upstream convention): dims [0..dim//2-1] pair with
    [dim//2..dim-1] and share the same frequency. Returns cos, sin of shape
    [seq_len, dim//2].
    """
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, half]
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to first ROPE_DIMS of x[..., seq, head_dim]. cos/sin: [seq, half].

    Computed in FP32 then cast back to x.dtype. Matches the upstream pattern
    (transformer.py:_apply_rotary_pos_emb): RoPE is FP16-fragile because cos/sin
    span [-1, 1] and the rotated values can be much larger than the inputs,
    so accumulation in FP32 keeps PSNR high under coremltools FP16 conversion.
    """
    in_dtype = x.dtype
    rope_dim = ROPE_DIMS
    half = rope_dim // 2
    x_rot = x[..., :rope_dim].to(torch.float32)
    x_pass = x[..., rope_dim:]

    first = x_rot[..., :half]    # 0..15
    second = x_rot[..., half:]   # 16..31
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    out_first  = first  * cos - second * sin
    out_second = second * cos + first  * sin

    rotated = torch.cat([out_first, out_second], dim=-1).to(in_dtype)
    return torch.cat([rotated, x_pass], dim=-1)


# ── Norms / FF / Attention ──────────────────────────────────────────────

class DyT(nn.Module):
    """DynamicTanh: gamma * tanh(alpha * x) + beta. `alpha` is scalar, gamma/beta vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class GLU_FF(nn.Module):
    """SwiGLU feedforward (ff_mult=3, inner=2304).

    Upstream `FeedForward(glu=True)` wraps a GLU module whose input projection is
    `proj` and an outer `Linear(inner→dim)` — naming kept flat as glu_proj/proj_out.
    """
    def __init__(self, dim: int = DIM, inner: int = FF_INNER):
        super().__init__()
        self.glu_proj = nn.Linear(dim, inner * 2, bias=True)
        self.proj_out = nn.Linear(inner, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.glu_proj(x)
        value, gate = x.chunk(2, dim=-1)
        return self.proj_out(value * F.silu(gate))


class DifferentialAttention(nn.Module):
    """Differential SDPA — two attention paths with shared V, subtract outputs.

    Upstream `Attention(differential=True)` does:
        q, k, v, q_diff, k_diff = to_qkv(x).chunk(5)
        out = SDPA(q,k,v) - SDPA(q_diff, k_diff, v)

    qk_norm='dyt' applies DyT on head_dim=64 to all four of q, k, q_diff, k_diff.
    """
    def __init__(self, dim: int = DIM, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, 5 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = DyT(head_dim)
        self.k_norm = DyT(head_dim)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.to_qkv(x)
        # Upstream chunk(5) order: q, k, v, q_diff, k_diff  (transformer.py:738)
        q, k, v, q_diff, k_diff = qkv.chunk(5, dim=-1)

        # [B, T, C] → [B, H, T, D]
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, D).transpose(1, 2)
        q      = to_heads(q)
        k      = to_heads(k)
        v      = to_heads(v)
        q_diff = to_heads(q_diff)
        k_diff = to_heads(k_diff)

        # DyT on q/k for both paths
        q      = self.q_norm(q)
        k      = self.k_norm(k)
        q_diff = self.q_norm(q_diff)
        k_diff = self.k_norm(k_diff)

        # RoPE on first 32 of 64 head dims
        q      = _apply_rope(q,      rope_cos, rope_sin)
        k      = _apply_rope(k,      rope_cos, rope_sin)
        q_diff = _apply_rope(q_diff, rope_cos, rope_sin)
        k_diff = _apply_rope(k_diff, rope_cos, rope_sin)

        # Inside a 34-token chunk: full attention (no mask, no SWA)
        out_main = F.scaled_dot_product_attention(q,      k,      v, scale=self.scale)
        out_diff = F.scaled_dot_product_attention(q_diff, k_diff, v, scale=self.scale)
        # Differential subtract in FP32 to avoid catastrophic cancellation
        # when out_main and out_diff have similar magnitudes (common case).
        out = (out_main.to(torch.float32) - out_diff.to(torch.float32)).to(out_main.dtype)

        # [B, H, T, D] → [B, T, C]
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x + attn(dyt(x)), x + ff(dyt(x))."""
    def __init__(self):
        super().__init__()
        self.pre_norm = DyT(DIM)
        self.attn     = DifferentialAttention()
        self.ff_norm  = DyT(DIM)
        self.ff       = GLU_FF()

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.pre_norm(x), rope_cos, rope_sin)
        x = x + self.ff(self.ff_norm(x))
        return x


# ── Decoder ─────────────────────────────────────────────────────────────

class SAMESDecoder(nn.Module):
    """SAME-S decoder: latents → audio patches.

    Input:  [B, 256, T_lat]
    Output: [B, 512, T_lat*16]

    `T_lat` must be a multiple of `PAD_MODULO`=2 — caller is responsible for
    padding (CoreML traces at fixed seq_len, so this is enforced at export time).

    If `output_audio=True`, the unpatch (patches → stereo waveform) is baked into
    the forward, so the output is finished audio [B, 2, T_lat*4096] instead of
    patches [B, 512, T_lat*16]. This makes the model self-contained (ONNX convention).
    """

    def __init__(self, output_audio: bool = False):
        super().__init__()
        self.output_audio = output_audio
        # Bottleneck softnorm decode: x *= running_std (scalar)
        self.register_buffer("running_std", torch.ones(1))

        # latent → backbone width
        self.project_in = nn.Linear(LATENT_DIM, DIM, bias=True)

        # Single learnable new-token (broadcast over 16 positions per latent)
        self.new_tokens = nn.Parameter(torch.zeros(1, 1, DIM))

        # 6 transformer blocks (3 pre-shift, 3 post-shift)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS)])

        # WNConv1d(768→512, k=3, padding=1) — weight_norm pre-fused at load time
        self.mapping = nn.Conv1d(DIM, OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

        # Precompute RoPE for one 34-token chunk (positions 0..33 → cos/sin [34, 16])
        cos, sin = _build_rope_cache(EFFECTIVE_CHUNK)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B, _C, T_lat = latents.shape

        # Bottleneck softnorm decode (scalar multiply)
        x = latents * self.running_std

        # Project: [B, 256, T_lat] → [B, T_lat, 256] → [B, T_lat, 768]
        x = self.project_in(x.transpose(1, 2))

        # Expand each latent slot with 16 copies of the single new-token:
        #   [B, T_lat, 1, D] cat [1, 1, 16, D] → [B, T_lat, 17, D] → [B, T_lat*17, D]
        x = x.unsqueeze(2)                                            # [B, T_lat, 1, D]
        nt = self.new_tokens.unsqueeze(0)                             # [1, 1, 1, D]
        nt = nt.expand(B, T_lat, SIN_PER_POS, DIM)                    # [B, T_lat, 16, D]
        x = torch.cat([x, nt], dim=2)                                 # [B, T_lat, 17, D]
        x = x.reshape(B, T_lat * SUB_CHUNK_SIZE, DIM)                 # [B, T_lat*17, D]

        # ── First half (blocks 0..2): independent 34-token chunks ──
        internal_T = T_lat * SUB_CHUNK_SIZE
        nc1 = internal_T // EFFECTIVE_CHUNK
        x = x.reshape(B * nc1, EFFECTIVE_CHUNK, DIM)
        x = self.blocks[0](x, self.rope_cos, self.rope_sin)
        x = self.blocks[1](x, self.rope_cos, self.rope_sin)
        x = self.blocks[2](x, self.rope_cos, self.rope_sin)
        x = x.reshape(B, internal_T, DIM)

        # ── Shift by 17 on both ends, then second half (blocks 3..5) ──
        # Replicate the first 17 tokens on the left and last 17 on the right.
        left  = x[:, :SHIFT, :]
        right = x[:, -SHIFT:, :]
        x = torch.cat([left, x, right], dim=1)                        # [B, internal_T + 34, D]
        nc2 = (internal_T + EFFECTIVE_CHUNK) // EFFECTIVE_CHUNK
        x = x.reshape(B * nc2, EFFECTIVE_CHUNK, DIM)
        x = self.blocks[3](x, self.rope_cos, self.rope_sin)
        x = self.blocks[4](x, self.rope_cos, self.rope_sin)
        x = self.blocks[5](x, self.rope_cos, self.rope_sin)
        x = x.reshape(B, internal_T + EFFECTIVE_CHUNK, DIM)
        x = x[:, SHIFT:-SHIFT, :]                                     # [B, internal_T, D]

        # ── Drop the latent-slot position (index 0 of each 17-block), keep 16 new tokens ──
        x = x.reshape(B * T_lat, SUB_CHUNK_SIZE, DIM)
        x = x[:, 1:, :]                                               # [B*T_lat, 16, D]
        x = x.reshape(B, T_lat * SIN_PER_POS, DIM)

        # Channels-first → output mapping (WNConv1d-fused)
        x = x.transpose(1, 2)                                         # [B, 768, T_lat*16]
        x = self.mapping(x)                                           # [B, 512, T_lat*16]

        if self.output_audio:
            # Unpatch: [B, 512, T_lat*16] → stereo waveform [B, 2, T_lat*4096].
            # Matches tflite_pipeline.unpatch (patch_size=256, channels=2):
            #   reshape [B, 2, 256, T_lat*16] → transpose(0,1,3,2) → reshape [B, 2, L*256]
            L = x.shape[2]                                            # T_lat*16
            x = x.reshape(B, 2, 256, L)
            x = x.permute(0, 1, 3, 2)                                 # [B, 2, L, 256]
            x = x.reshape(B, 2, L * 256)                              # [B, 2, T_lat*4096]
        return x


# ── Weight loading ──────────────────────────────────────────────────────

def load_model(weights_path: str | None = None, dtype: torch.dtype = torch.float32,
               output_audio: bool = False) -> SAMESDecoder:
    """Load SAMESDecoder from .npz produced by scripts/export_same_s_weights.py.

    .npz key naming is flat and matches PyTorch module hierarchy directly except:
      - `running_std` → buffer
      - `new_tokens`  → parameter (1,1,768)
      - `mapping.{weight,bias}` already weight_norm-fused into plain Conv1d
    """
    if weights_path is None:
        candidates = [
            Path(__file__).parent.parent / "mlx" / "same_s_decoder_f32.npz",
            Path(__file__).parent.parent / "weights" / "same_s_decoder_f32.npz",
        ]
        for p in candidates:
            if p.exists():
                weights_path = str(p)
                break
        else:
            raise FileNotFoundError(
                f"Weights not found; searched {[str(p) for p in candidates]}"
            )

    model = SAMESDecoder(output_audio=output_audio)
    raw = dict(np.load(weights_path))

    state = {}
    for k, arr in raw.items():
        state[k] = torch.from_numpy(arr.astype(np.float32))

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Filter precomputed RoPE buffers from missing
    real_missing = [m for m in missing if "rope_" not in m]
    if real_missing:
        print(f"  load_state_dict missing: {real_missing}")
    if unexpected:
        print(f"  load_state_dict unexpected: {unexpected}")

    model = model.to(dtype)
    model.eval()
    return model


# ── Convenience: build state_dict suitable for upstream SAMEDecoder ─────

def state_dict_for_upstream(npz_path: str) -> dict:
    """Return a state_dict that loads into upstream `SAMEDecoder` (sa-tools-dev).

    Used for cross-validation: build upstream model, load via this state dict,
    compare against our SAMESDecoder.
    """
    raw = dict(np.load(npz_path))
    sd: dict[str, torch.Tensor] = {}
    sd["layers.1.weight"]    = torch.from_numpy(raw["project_in.weight"])
    sd["layers.1.bias"]      = torch.from_numpy(raw["project_in.bias"])
    sd["layers.3.new_tokens"] = torch.from_numpy(raw["new_tokens"])
    for i in range(NUM_BLOCKS):
        bp = f"layers.3.transformers.{i}"
        bm = f"blocks.{i}"
        for k in ("alpha", "gamma", "beta"):
            sd[f"{bp}.pre_norm.{k}"] = torch.from_numpy(raw[f"{bm}.pre_norm.{k}"])
            sd[f"{bp}.ff_norm.{k}"]  = torch.from_numpy(raw[f"{bm}.ff_norm.{k}"])
        sd[f"{bp}.self_attn.to_qkv.weight"] = torch.from_numpy(raw[f"{bm}.attn.to_qkv.weight"])
        sd[f"{bp}.self_attn.to_out.weight"] = torch.from_numpy(raw[f"{bm}.attn.to_out.weight"])
        for nm in ("q_norm", "k_norm"):
            for k in ("alpha", "gamma", "beta"):
                sd[f"{bp}.self_attn.{nm}.{k}"] = torch.from_numpy(raw[f"{bm}.attn.{nm}.{k}"])
        sd[f"{bp}.ff.ff.0.proj.weight"] = torch.from_numpy(raw[f"{bm}.ff.glu_proj.weight"])
        sd[f"{bp}.ff.ff.0.proj.bias"]   = torch.from_numpy(raw[f"{bm}.ff.glu_proj.bias"])
        sd[f"{bp}.ff.ff.2.weight"]      = torch.from_numpy(raw[f"{bm}.ff.proj_out.weight"])
        sd[f"{bp}.ff.ff.2.bias"]        = torch.from_numpy(raw[f"{bm}.ff.proj_out.bias"])
    sd["layers.3.mapping.weight"] = torch.from_numpy(raw["mapping.weight"])
    sd["layers.3.mapping.bias"]   = torch.from_numpy(raw["mapping.bias"])
    return sd
