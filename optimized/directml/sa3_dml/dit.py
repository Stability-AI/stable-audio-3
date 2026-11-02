"""Diffusion transformer runner with the reference conditioning and CFG branches."""

from __future__ import annotations

import numpy as np

from .sampling import guided_velocity
from .session import Session
from .specs import LATENT_DIM
from .surgery import CROSS_ATTN_SCALE_INPUT, CROSS_ATTN_TOKENS
from .text import PromptEmbedding


class DiT:
    """Velocity model v(x, t) for one generation of fixed latent length.

    cfg_scale == 1 runs one forward per step. Otherwise each step runs the conditional branch
    and an unconditional one, which uses an all-zero cross-attention sequence or, with a
    negative prompt, its embeddings with padded tokens zeroed, as in the reference sampler.
    `local_add_cond` [1, 257, L] carries the inpainting mask and masked source latents and is
    shared by both branches.
    """

    def __init__(self, session: Session, length: int, prompt: PromptEmbedding, seconds: float,
                 negative: PromptEmbedding | None = None, cfg_scale: float = 1.0, apg_scale: float = 1.0,
                 local_add_cond: np.ndarray | None = None):
        self.session = session
        self.cfg_scale = float(cfg_scale)
        self.apg_scale = float(apg_scale)
        self.forwards = 0
        if local_add_cond is None:
            local_add_cond = np.zeros((1, LATENT_DIM + 1, length), np.float32)
        common = {
            "seconds_total": np.array([seconds], np.float32),
            "local_add_cond": np.asarray(local_add_cond, np.float32),
        }
        ones = np.ones((1, CROSS_ATTN_TOKENS, 1), np.float32)
        self.cond = {**common, "t5_hidden": prompt.hidden, "t5_mask": prompt.mask,
                     CROSS_ATTN_SCALE_INPUT: ones}
        if negative is None:
            self.uncond = {**self.cond, CROSS_ATTN_SCALE_INPUT: np.zeros_like(ones)}
        else:
            keep = np.concatenate([negative.mask, np.ones((1, 1), np.float32)], axis=1)[..., None]
            self.uncond = {**common, "t5_hidden": negative.hidden, "t5_mask": negative.mask,
                           CROSS_ATTN_SCALE_INPUT: keep.astype(np.float32)}

    def _forward(self, feeds: dict, x: np.ndarray, t: float) -> np.ndarray:
        self.forwards += 1
        out = self.session.run({**feeds, "x": x, "t": np.array([t], np.float32)})[0]
        return out.astype(np.float32)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        v_cond = self._forward(self.cond, x, t)
        if self.cfg_scale == 1.0:
            return v_cond
        v_uncond = self._forward(self.uncond, x, t)
        return guided_velocity(x, t, v_cond, v_uncond, self.cfg_scale, self.apg_scale)
