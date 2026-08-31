"""SAME-L Encoder — PyTorch (mirror of same_l_decoder_torch.SAMELDecoder, downsampling direction).

Audio -> latent. The transformer stack is IDENTICAL to the decoder's (12 blocks, dim=1536,
5x-QKV differential SWA window=+-17, DyT norms, GLU FF with sin gate from block 5) — it reuses
the decoder module classes verbatim (DyT / DifferentialSWA / FeedForward / TransformerBlock),
so windowed_decoder.patch() windows this encoder too.

Flow (input_audio=True — matches the shipped baked-I/O encoder):
  audio [B,2,L*4096]
   -> patchify           [B,512,L*16]           (inverse of decoder.unpatch)
   -> mapping WNConv1d(512->1536,k1) -> [B,L*16,1536]
   -> interleave: 1 learnable summary token at index 0 of each 17-group -> [B,L*17,1536]
   -> 12x transformer (SWA over L*17)
   -> keep summary token (index 0 of each 17-group) -> [B,L,1536]      (downsample 16->1)
   -> project_out Linear(1536->256) -> [B,256,L]
   -> bottleneck encode  -> softnorm latent [B,256,L]

Bottleneck: the decoder's only op is `x = latents * running_std`; the encoder inverts it,
`latents = z / running_std` (scaling_factor/bias are the VAE affine, absorbed like the decoder).
The exact form is pinned by matching the shipped enc tflite on real music (see validate script).
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import same_l_decoder_torch as D   # shared modules (must be on sys.path)

LATENT_DIM = D.LATENT_DIM        # 256
DIM = D.DIM                      # 1536
NUM_BLOCKS = D.NUM_BLOCKS        # 12
IN_CHANNELS = D.OUT_CHANNELS     # 512 (patch channels)
BLOCK_SIZE = D.BLOCK_SIZE        # 17
SUB = D.SUB_CHUNK_SIZE           # 17
PATCH = 256                      # samples per patch channel-group


class SAMELEncoder(nn.Module):
    def __init__(self, T_lat: int | None = 320, input_audio: bool = True,
                 bottleneck: str = "canonical"):
        super().__init__()
        self.T_lat = T_lat
        self.input_audio = input_audio
        self.bottleneck_mode = bottleneck
        # input mapping: WNConv1d(512->1536, k=1) — fused weight_norm at load
        self.mapping = nn.Conv1d(IN_CHANNELS, DIM, kernel_size=1, bias=True)
        self.new_tokens = nn.Parameter(torch.zeros(1, 1, DIM))
        self.blocks = nn.ModuleList([D.TransformerBlock(i) for i in range(NUM_BLOCKS)])
        # ENCODER has NO sin gate (unlike decoder blocks 5..11) — pinned vs shipped tflite
        for blk in self.blocks:
            blk.ff.use_sin = False
        self.project_out = nn.Linear(DIM, LATENT_DIM, bias=True)
        self.register_buffer("running_std", torch.ones(1))
        self.register_buffer("scaling_factor", torch.ones(1, LATENT_DIM, 1))
        self.register_buffer("bias", torch.zeros(1, LATENT_DIM, 1))

        if T_lat is not None:
            internal_T = T_lat * SUB
            cos, sin = D._build_rope_cache(internal_T)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            if internal_T <= BLOCK_SIZE:
                self.register_buffer("swa_mask", torch.zeros(0), persistent=False)
            else:
                self.register_buffer("swa_mask", D._make_swa_mask(internal_T), persistent=False)
        else:
            self.register_buffer("rope_cos", torch.zeros(0), persistent=False)
            self.register_buffer("rope_sin", torch.zeros(0), persistent=False)
            self.register_buffer("swa_mask", torch.zeros(0), persistent=False)

    def _bottleneck(self, z):
        # z: [B,256,L] raw project_out. PINNED bit-exact vs shipped tflite (cos=1, -91..-116 dB):
        #   latent = (z * scaling_factor + bias) / running_std
        m = self.bottleneck_mode
        if m == "raw":
            return z
        if m == "canonical":
            return (z * self.scaling_factor + self.bias) / self.running_std
        raise ValueError(m)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        B = audio.shape[0]
        if self.input_audio:
            # patchify: [B,2,L*4096] -> [B,512,L*16]  (inverse of decoder.unpatch)
            Lp = audio.shape[2] // PATCH                 # L*16
            x = audio.reshape(B, 2, Lp, PATCH).permute(0, 1, 3, 2).reshape(B, 2 * PATCH, Lp)
        else:
            x = audio                                    # already [B,512,L*16]
        Lp = x.shape[2]                                  # L*16
        T_lat = Lp // (SUB - 1)                          # L
        x = self.mapping(x)                              # [B,1536,L*16]
        x = x.transpose(1, 2)                            # [B,L*16,1536]

        # append summary token at END (index 16) of each 17-group (audio at 0..15)
        x = x.reshape(B, T_lat, SUB - 1, DIM)            # [B,L,16,1536]
        nt = self.new_tokens.unsqueeze(0).expand(B, T_lat, 1, DIM)   # [B,L,1,1536]
        x = torch.cat([x, nt], dim=2)                    # [B,L,17,1536]
        x = x.reshape(B, T_lat * SUB, DIM)               # [B,L*17,1536]

        internal_T = T_lat * SUB
        if self.rope_cos.numel() == 0 or self.rope_cos.shape[0] < internal_T:
            cos, sin = D._build_rope_cache(internal_T); cos = cos.to(x); sin = sin.to(x)
        else:
            cos, sin = self.rope_cos, self.rope_sin
        if internal_T <= BLOCK_SIZE:
            attn_mask = None
        elif self.swa_mask.numel() == 0 or self.swa_mask.shape[0] != internal_T:
            attn_mask = D._make_swa_mask(internal_T).to(x)
        else:
            attn_mask = self.swa_mask

        for layer in self.blocks:
            x = layer(x, cos, sin, attn_mask)

        # keep summary token (index 16, the appended new_token): downsample 16->1
        x = x.reshape(B, T_lat, SUB, DIM)[:, :, SUB - 1]  # [B,L,1536]
        x = self.project_out(x)                          # [B,L,256]
        x = x.transpose(1, 2)                            # [B,256,L]
        return self._bottleneck(x)


def load_model(weights_path: str | None = None, T_lat: int | None = None,
               dtype: torch.dtype = torch.float32, input_audio: bool = True,
               bottleneck: str = "canonical") -> SAMELEncoder:
    if weights_path is None:
        weights_path = str(Path(__file__).parent.parent / "mlx" / "same_l_encoder_f32.npz")
    model = SAMELEncoder(T_lat=T_lat, input_audio=input_audio, bottleneck=bottleneck).eval()
    raw = dict(np.load(weights_path))
    state = {}
    for k, arr in raw.items():
        # bottleneck.{running_std,scaling_factor,bias} -> top-level buffers
        key = k[len("bottleneck."):] if k.startswith("bottleneck.") else k
        state[key] = torch.from_numpy(arr.astype(np.float32))
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "rope_" not in m and "swa_mask" not in m]
    if real_missing:
        print(f"  missing: {real_missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")
    return model.to(dtype)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    m = load_model(T_lat=None)
    n = sum(p.numel() for p in m.parameters())
    print(f"SAME-L encoder params: {n:,} ({n*4/1e6:.0f} MB fp32)")
    x = torch.randn(1, 2, 64 * 4096) * 0.1
    with torch.no_grad():
        y = m(x)
    print(f"audio {tuple(x.shape)} -> latent {tuple(y.shape)} std={y.std():.3f}")
