"""SAME-S Encoder — PyTorch (mirror of same_s_decoder_torch.SAMESDecoder, downsampling direction).

Reuses the decoder's modules verbatim (DyT / GLU_FF / DifferentialAttention / TransformerBlock /
RoPE) — the transformer stack is identical. SAME-S attention is BLOCK-LOCAL (non-overlapping
34-token chunks, full attn inside; midpoint-shift by 17 between blocks 2<->3) -> O(L) natively,
so NO windowing surgery needed (unlike SAME-L). Requires even T_lat (PAD_MODULO=2).

Flow (input_audio=True):
  audio [B,2,L*4096] -> patchify [B,512,L*16] -> mapping WNConv1d(512->768,k=1) -> [B,L*16,768]
   -> summary token appended at index 16 of each 17-group -> [B,L*17,768]
   -> blocks 0..2 (34-chunks) -> shift 17 -> blocks 3..5 -> unshift
   -> keep summary (index 16) -> [B,L,768] -> project_out Linear(768->256) -> [B,256,L]
   -> bottleneck (z*scaling_factor+bias)/running_std
(summary position + bottleneck to be pinned vs shipped same-s enc tflite, as for SAME-L.)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import same_s_decoder_torch as D   # shared modules (must be on sys.path)

LATENT_DIM = D.LATENT_DIM          # 256
DIM = D.DIM                        # 768
NUM_BLOCKS = D.NUM_BLOCKS          # 6
IN_CHANNELS = D.OUT_CHANNELS       # 512
SUB = D.SUB_CHUNK_SIZE             # 17
ECHUNK = D.EFFECTIVE_CHUNK         # 34
SHIFT = D.SHIFT                    # 17
PATCH = 256


class SAMESEncoder(nn.Module):
    def __init__(self, input_audio: bool = True, bottleneck: str = "canonical", summ_idx: int = SUB - 1):
        super().__init__()
        self.input_audio = input_audio
        self.bottleneck_mode = bottleneck
        self.summ_idx = summ_idx
        self.mapping = nn.Conv1d(IN_CHANNELS, DIM, kernel_size=1, bias=True)   # 512->768 k1
        self.new_tokens = nn.Parameter(torch.zeros(1, 1, DIM))
        self.blocks = nn.ModuleList([D.TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.project_out = nn.Linear(DIM, LATENT_DIM, bias=True)
        self.register_buffer("running_std", torch.ones(1))
        self.register_buffer("scaling_factor", torch.ones(1, LATENT_DIM, 1))
        self.register_buffer("bias", torch.zeros(1, LATENT_DIM, 1))
        cos, sin = D._build_rope_cache(ECHUNK)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _bottleneck(self, z):
        if self.bottleneck_mode == "raw":
            return z
        if self.bottleneck_mode == "canonical":
            return (z * self.scaling_factor + self.bias) / self.running_std
        raise ValueError(self.bottleneck_mode)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        B = audio.shape[0]
        if self.input_audio:
            Lp = audio.shape[2] // PATCH
            x = audio.reshape(B, 2, Lp, PATCH).permute(0, 1, 3, 2).reshape(B, 2 * PATCH, Lp)
        else:
            x = audio
        Lp = x.shape[2]                                  # L*16
        T_lat = Lp // (SUB - 1)                          # L
        x = self.mapping(x).transpose(1, 2)              # [B,L*16,768]
        x = x.reshape(B, T_lat, SUB - 1, DIM)
        nt = self.new_tokens.unsqueeze(0).expand(B, T_lat, 1, DIM)
        if self.summ_idx == SUB - 1:
            x = torch.cat([x, nt], dim=2)
        else:
            x = torch.cat([nt, x], dim=2)
        x = x.reshape(B, T_lat * SUB, DIM)

        internal_T = T_lat * SUB
        rc, rs = self.rope_cos, self.rope_sin
        # first half: independent 34-token chunks
        nc1 = internal_T // ECHUNK
        x = x.reshape(B * nc1, ECHUNK, DIM)
        x = self.blocks[0](x, rc, rs); x = self.blocks[1](x, rc, rs); x = self.blocks[2](x, rc, rs)
        x = x.reshape(B, internal_T, DIM)
        # shift 17 both ends, second half, unshift
        left = x[:, :SHIFT, :]; right = x[:, -SHIFT:, :]
        x = torch.cat([left, x, right], dim=1)
        nc2 = (internal_T + ECHUNK) // ECHUNK
        x = x.reshape(B * nc2, ECHUNK, DIM)
        x = self.blocks[3](x, rc, rs); x = self.blocks[4](x, rc, rs); x = self.blocks[5](x, rc, rs)
        x = x.reshape(B, internal_T + ECHUNK, DIM)
        x = x[:, SHIFT:-SHIFT, :]

        # keep summary token, downsample 16->1
        x = x.reshape(B, T_lat, SUB, DIM)[:, :, self.summ_idx]
        x = self.project_out(x).transpose(1, 2)          # [B,256,L]
        return self._bottleneck(x)


def load_model(weights_path: str | None = None, dtype: torch.dtype = torch.float32,
               input_audio: bool = True, bottleneck: str = "canonical", summ_idx: int = SUB - 1):
    if weights_path is None:
        weights_path = str(Path(__file__).parent.parent / "mlx" / "same_s_encoder_f32.npz")
    model = SAMESEncoder(input_audio=input_audio, bottleneck=bottleneck, summ_idx=summ_idx).eval()
    raw = dict(np.load(weights_path))
    state = {}
    for k, arr in raw.items():
        key = k[len("bottleneck."):] if k.startswith("bottleneck.") else k
        state[key] = torch.from_numpy(arr.astype(np.float32))
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "rope_" not in m]
    if real_missing:
        print(f"  missing: {real_missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")
    return model.to(dtype)
