"""SAME latent encoder and decoder with the reference overlapped chunking."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .assets import ENCODER_INPUT, ENCODER_SAMPLES
from .session import Session, SessionPool
from .specs import DOWNSAMPLING_RATIO, LATENT_DIM, CodecSpec
from .surgery import TOKEN_NOISE_INPUT, TOKEN_ROWS

BOTTLENECK_DITHER = 1e-3
DECODER_INPUT = "latent"


def chunk_starts(total: int, chunk: int, overlap: int) -> list[int]:
    hop = chunk - overlap
    starts = list(range(0, total - chunk + 1, hop))
    if starts[-1] != total - chunk:
        starts.append(total - chunk)
    return starts


def stitch(total: int, chunk: int, overlap: int, scale: int,
           run: Callable[[int], np.ndarray]) -> np.ndarray:
    """Run `run(start)` on windows of `chunk` latent frames and keep the centre of each.

    Inner window edges drop half the overlap and the last window is anchored to the end, as in
    the reference `encode_audio` / `decode_audio`. Outputs have `scale` samples per frame.
    """
    starts = chunk_starts(total, chunk, overlap)
    half, size = (overlap // 2) * scale, chunk * scale
    out = None
    for i, start in enumerate(starts):
        part = run(start)
        if out is None:
            out = np.zeros(part.shape[:-1] + (total * scale,), np.float32)
        last = i == len(starts) - 1
        offset = total * scale - size if last else start * scale
        left = 0 if i == 0 else half
        right = size if last else size - half
        out[..., offset + left:offset + right] = part[..., left:right]
    return out


def _align(frames: int, align: int) -> int:
    return -(-frames // align) * align


class _Codec:
    role = ""

    def __init__(self, graph: Path, pool: SessionPool, codec: CodecSpec, chunk: int | None = None,
                 overlap: int | None = None):
        self.graph, self.pool, self.codec = graph, pool, codec
        self.chunk = codec.chunk if chunk is None else chunk
        self.overlap = codec.overlap if overlap is None else overlap
        if self.chunk and (self.chunk % max(2, codec.frame_align) or self.overlap % 2
                           or not 0 <= self.overlap < self.chunk):
            raise ValueError("chunk and overlap must be even, with overlap < chunk "
                             "(chunk 0 processes the whole sequence at once)")

    def _token_noise(self, session: Session, rows: int, std: float,
                     rng: np.random.Generator | None) -> dict[str, np.ndarray]:
        spec = session.inputs.get(TOKEN_NOISE_INPUT)
        if spec is None:
            return {}
        shape = tuple(rows if isinstance(d, str) or d is None else int(d) for d in spec.shape)
        if rng is None or not std:
            return {TOKEN_NOISE_INPUT: np.zeros(shape, np.float32)}
        return {TOKEN_NOISE_INPUT: std * rng.standard_normal(shape, dtype=np.float32)}

    def _windowed(self, total: int, scale: int, run: Callable[[int, int], np.ndarray]) -> np.ndarray:
        if not self.chunk or total < self.chunk:
            return run(0, total)
        return stitch(total, self.chunk, self.overlap, scale, lambda start: run(start, self.chunk))


class Encoder(_Codec):
    """Encodes [2, N] audio at 44.1 kHz into [1, 256, ceil(N / 4096)] latents.

    Windows of `chunk` frames overlap by `overlap` frames as in the reference pretransform. Each
    window draws the encoder token noise of the reference SAME-L encoder from `rng`.
    """

    role = "encoder"

    def _encode(self, audio: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        frames = audio.shape[-1] // DOWNSAMPLING_RATIO
        padded = _align(frames, self.codec.frame_align)
        if padded != frames:
            audio = np.pad(audio, ((0, 0), (0, (padded - frames) * DOWNSAMPLING_RATIO)))
        session = self.pool.get(self.role, self.graph, {ENCODER_SAMPLES: padded * DOWNSAMPLING_RATIO,
                                                        TOKEN_ROWS: padded})
        feeds = {ENCODER_INPUT: audio[None], **self._token_noise(session, padded, self.codec.encoder_token_noise, rng)}
        return session.run(feeds)[0].astype(np.float32)[..., :frames]

    def __call__(self, audio: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        audio = np.asarray(audio, np.float32)
        frames = -(-audio.shape[-1] // DOWNSAMPLING_RATIO)
        audio = np.pad(audio, ((0, 0), (0, frames * DOWNSAMPLING_RATIO - audio.shape[-1])))
        spl = DOWNSAMPLING_RATIO
        return self._windowed(frames, 1, lambda start, n: self._encode(audio[:, start * spl:(start + n) * spl], rng))


class Decoder(_Codec):
    """Decodes [1, 256, L] latents to [1, 2, L * 4096] audio in [-1, 1].

    Each window adds the reference bottleneck dither (1e-3 of the latent scale) and Gaussian
    noise of std `token_noise` on the learned upsampling tokens, both drawn from `rng`.
    """

    role = "decoder"

    def _decode(self, latents: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        frames = latents.shape[-1]
        rows = _align(frames, self.codec.frame_align)
        session = self.pool.get(self.role, self.graph, {"L": frames, TOKEN_ROWS: rows})
        if rng is not None:
            latents = latents + BOTTLENECK_DITHER * rng.standard_normal(latents.shape, dtype=np.float32)
        feeds = {DECODER_INPUT: latents, **self._token_noise(session, rows, self.codec.token_noise, rng)}
        return session.run(feeds)[0].astype(np.float32)

    def __call__(self, latents: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        latents = np.asarray(latents, np.float32).reshape(1, LATENT_DIM, -1)
        return self._windowed(latents.shape[-1], DOWNSAMPLING_RATIO,
                              lambda start, n: self._decode(latents[..., start:start + n], rng))
