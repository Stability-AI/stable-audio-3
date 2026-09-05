"""SAME-L Decoder — PyTorch (for CoreML export).

Architecturally similar to TAAEv2: 12 blocks, dim=1536, SWA attention with
window=±17 tokens, sin gate from block 5. But uses SAME-style single learnable
new_tokens (broadcast) and a Conv1d kernel=1 mapping (vs TAAEv2's Linear).

Designed for trace-friendly CoreML export at fixed T_lat. The SWA mask is
materialized as a buffer at __init__ time so it's a constant in the trace.

Input:  [B, 256, T_lat]
Output: [B, 512, T_lat*16]  (patches),
        OR [B, 2, T_lat*4096] finished stereo audio when output_audio=True
        (unpatch baked in-graph — self-contained, ONNX decoder convention).
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants (SAME-L from sa3-medium)
LATENT_DIM       = 256
DIM              = 1536
NUM_HEADS        = 24
HEAD_DIM         = 64
ROPE_DIMS        = 32
NUM_BLOCKS       = 12
FF_INNER         = 4608           # ff_mult=3
SIN_START_BLOCK  = 5
OUT_CHANNELS     = 512
STRIDE           = 16
SUB_CHUNK_SIZE   = STRIDE + 1     # 17
BLOCK_SIZE       = SUB_CHUNK_SIZE
SIN_PER_POS      = SUB_CHUNK_SIZE - 1  # 16


# ── RoPE ─────────────────────────────────────────────────────────────

def _build_rope_cache(max_seq: int, dim: int = ROPE_DIMS, base: float = 10000.0):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq, half]
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to first ROPE_DIMS of x. x: [..., seq, head_dim]."""
    T = x.shape[-2]
    half = ROPE_DIMS // 2
    x_rot = x[..., :ROPE_DIMS]
    x_pass = x[..., ROPE_DIMS:]
    cos_t = cos[:T]
    sin_t = sin[:T]
    first = x_rot[..., :half]
    second = x_rot[..., half:]
    out_first = first * cos_t - second * sin_t
    out_second = second * cos_t + first * sin_t
    return torch.cat([out_first, out_second, x_pass], dim=-1)


def _make_swa_mask(T: int, half_w: int = BLOCK_SIZE) -> torch.Tensor:
    """[T, T] additive SWA mask: 0 where |i-j| <= half_w, -inf elsewhere."""
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    valid = (j - i).abs() <= half_w
    return torch.where(valid, torch.tensor(0.0), torch.tensor(float("-inf")))


class DyT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class DifferentialSWA(nn.Module):
    """5x QKV self-attn with SWA mask, two SDPA paths concatenated as 2H heads."""

    def __init__(self):
        super().__init__()
        self.scale = HEAD_DIM ** -0.5
        self.to_qkv = nn.Linear(DIM, 5 * DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)
        self.q_norm = DyT(HEAD_DIM)
        self.k_norm = DyT(HEAD_DIM)

    def forward(self, x, cos, sin, attn_mask=None):
        B, T, _ = x.shape
        H, D = NUM_HEADS, HEAD_DIM
        qkv = self.to_qkv(x)
        q1, k1, v, q2, k2 = qkv.chunk(5, dim=-1)

        def to_heads(t):
            return t.view(B, T, H, D).transpose(1, 2)

        q1, k1, v, q2, k2 = [to_heads(t) for t in (q1, k1, v, q2, k2)]

        q1 = self.q_norm(q1); k1 = self.k_norm(k1)
        q2 = self.q_norm(q2); k2 = self.k_norm(k2)

        q1 = _apply_rope(q1, cos, sin); k1 = _apply_rope(k1, cos, sin)
        q2 = _apply_rope(q2, cos, sin); k2 = _apply_rope(k2, cos, sin)

        Q = torch.cat([q1, q2], dim=1)
        K = torch.cat([k1, k2], dim=1)
        V = torch.cat([v, v], dim=1)

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, scale=self.scale)
        out1, out2 = out.chunk(2, dim=1)
        diff = out1 - out2
        return self.to_out(diff.transpose(1, 2).reshape(B, T, DIM))


class FeedForward(nn.Module):
    """GLU FF; optional sin(πx) gate (blocks 5..11)."""
    def __init__(self, use_sin=False):
        super().__init__()
        self.use_sin = use_sin
        self.glu_proj = nn.Linear(DIM, FF_INNER * 2, bias=True)
        self.proj_out = nn.Linear(FF_INNER, DIM, bias=True)

    def forward(self, x):
        projected = self.glu_proj(x)
        value, gate = projected.chunk(2, dim=-1)
        if self.use_sin:
            activated = value * torch.sin(gate * math.pi)
        else:
            activated = value * F.silu(gate)
        return self.proj_out(activated)


class TransformerBlock(nn.Module):
    def __init__(self, block_idx):
        super().__init__()
        self.pre_norm = DyT(DIM)
        self.attn = DifferentialSWA()
        self.ff_norm = DyT(DIM)
        self.ff = FeedForward(use_sin=(block_idx >= SIN_START_BLOCK))

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(self.pre_norm(x), cos, sin, attn_mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class SAMELDecoder(nn.Module):
    """SAME-L decoder.

    For trace, T_lat can be fixed at construction (bake SWA mask as a buffer)
    OR set max_T_lat to None for dynamic-shape inference (mask computed in fwd).
    """

    def __init__(self, T_lat: int | None = 320, output_audio: bool = False):
        super().__init__()
        self.T_lat = T_lat
        self.output_audio = output_audio
        self.register_buffer("running_std", torch.ones(1))
        self.project_in = nn.Linear(LATENT_DIM, DIM, bias=True)
        # Single learnable new_token, broadcast over 16 positions per latent
        self.new_tokens = nn.Parameter(torch.zeros(1, 1, DIM))
        self.blocks = nn.ModuleList([TransformerBlock(i) for i in range(NUM_BLOCKS)])
        # WNConv1d kernel=1 in upstream — fused weight_norm at load time
        self.mapping = nn.Conv1d(DIM, OUT_CHANNELS, kernel_size=1, bias=True)

        if T_lat is not None:
            # RoPE buffer covers internal_T = T_lat * 17
            cos, sin = _build_rope_cache(T_lat * BLOCK_SIZE)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            # SWA mask buffer for fixed internal_T
            internal_T = T_lat * BLOCK_SIZE
            if internal_T <= BLOCK_SIZE:
                self.register_buffer("swa_mask", torch.zeros(0), persistent=False)
            else:
                self.register_buffer("swa_mask", _make_swa_mask(internal_T), persistent=False)
        else:
            # Dynamic-shape mode: compute RoPE/mask in forward
            self.register_buffer("rope_cos", torch.zeros(0), persistent=False)
            self.register_buffer("rope_sin", torch.zeros(0), persistent=False)
            self.register_buffer("swa_mask", torch.zeros(0), persistent=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B, _, T_lat = latents.shape

        # Bottleneck softnorm decode
        x = latents * self.running_std

        # [B, 256, T_lat] → [B, T_lat, 256] → [B, T_lat, DIM]
        x = self.project_in(x.transpose(1, 2))

        # Broadcast single new_token to 16 positions per latent
        x_e = x.unsqueeze(2)                                # [B, T_lat, 1, DIM]
        nt = self.new_tokens.unsqueeze(0).expand(B, T_lat, SIN_PER_POS, DIM)
        x = torch.cat([x_e, nt], dim=2)                     # [B, T_lat, 17, DIM]
        x = x.reshape(B, T_lat * SUB_CHUNK_SIZE, DIM)

        # RoPE & SWA mask: use baked buffers if available, else compute on the fly
        internal_T = T_lat * SUB_CHUNK_SIZE
        if self.rope_cos.numel() == 0 or self.rope_cos.shape[0] < internal_T:
            cos, sin = _build_rope_cache(internal_T)
            cos = cos.to(x.device, x.dtype); sin = sin.to(x.device, x.dtype)
        else:
            cos = self.rope_cos; sin = self.rope_sin

        if internal_T <= BLOCK_SIZE:
            attn_mask = None
        else:
            if self.swa_mask.numel() == 0 or self.swa_mask.shape[0] != internal_T:
                attn_mask = _make_swa_mask(internal_T).to(x.device, x.dtype)
            else:
                attn_mask = self.swa_mask

        for layer in self.blocks:
            x = layer(x, cos, sin, attn_mask)

        # Drop the latent-slot at index 0 of each 17-block, keep 16 new positions
        x = x.reshape(B, T_lat, SUB_CHUNK_SIZE, DIM)[:, :, 1:]
        x = x.reshape(B, T_lat * SIN_PER_POS, DIM)

        # mapping: nn.Conv1d expects [B, C, T]. We have [B, T, C].
        x = x.transpose(1, 2)
        x = self.mapping(x)  # [B, OUT_CHANNELS, T_lat*16]

        if self.output_audio:
            # Unpatch: [B, 512, T_lat*16] → stereo waveform [B, 2, T_lat*4096].
            # Matches tflite_pipeline.unpatch (patch_size=256, channels=2):
            #   reshape [B, 2, 256, T_lat*16] → transpose(0,1,3,2) → reshape [B, 2, L*256]
            L = x.shape[2]                    # T_lat*16
            x = x.reshape(B, 2, 256, L)
            x = x.permute(0, 1, 3, 2)         # [B, 2, L, 256]
            x = x.reshape(B, 2, L * 256)      # [B, 2, T_lat*4096]
        return x


# ── Weight loading ──────────────────────────────────────────────────

def load_model(weights_path: str | None = None, T_lat: int = 320,
               dtype: torch.dtype = torch.float32,
               output_audio: bool = False) -> SAMELDecoder:
    """Load SAMELDecoder from .npz produced by scripts/export_same_l_weights.py.

    .npz mapping.weight has PyTorch Conv1d layout [out=512, in=1536, k=1] — matches
    nn.Conv1d directly, no permute needed.
    """
    if weights_path is None:
        weights_path = str(Path(__file__).parent.parent / "mlx" / "same_l_decoder_f32.npz")

    model = SAMELDecoder(T_lat=T_lat, output_audio=output_audio).eval()
    raw = dict(np.load(weights_path))
    state = {}
    for k, arr in raw.items():
        state[k] = torch.from_numpy(arr.astype(np.float32))
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "rope_" not in m and "swa_mask" not in m]
    if real_missing:
        print(f"  missing: {real_missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")
    return model.to(dtype)


if __name__ == "__main__":
    m = load_model(T_lat=32, dtype=torch.float32)
    nparams = sum(p.numel() for p in m.parameters())
    print(f"Params: {nparams:,} ({nparams*2/1e6:.1f} MB FP16)")
    x = torch.randn(1, 256, 32) * 0.5
    with torch.no_grad():
        y = m(x)
    print(f"y: {y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")
