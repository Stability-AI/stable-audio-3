"""WAV input/output, resampling and channel conversion."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WAV_FORMATS = ("float32", "pcm16")

_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE
_RESAMPLE_WIDTH = 6
_RESAMPLE_ROLLOFF = 0.99


@dataclass
class Audio:
    """Waveform [channels, samples] in float32 and its sample rate."""

    samples: np.ndarray
    sample_rate: int

    @classmethod
    def load(cls, path: Path | str) -> Audio:
        return cls(*read_wav(path))

    @property
    def seconds(self) -> float:
        return self.samples.shape[-1] / self.sample_rate


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int, fmt: str = "float32") -> Path:
    """Write [channels, samples] audio in [-1, 1] as IEEE float32 or 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, np.float32), -1.0, 1.0)
    channels, frames = audio.shape
    if fmt == "pcm16":
        data = np.rint(audio.T * 32767.0).astype("<i2").tobytes()
        tag, bits, extra = 1, 16, b""
    elif fmt == "float32":
        data = audio.T.astype("<f4").tobytes()
        tag, bits, extra = 3, 32, struct.pack("<H", 0)
    else:
        raise ValueError(f"unknown WAV format {fmt!r}")
    block = channels * bits // 8
    fmt_chunk = struct.pack("<HHIIHH", tag, channels, sample_rate, sample_rate * block, block, bits) + extra
    chunks = [b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk]
    if tag == 3:
        chunks.append(b"fact" + struct.pack("<II", 4, frames))
    chunks.append(b"data" + struct.pack("<I", len(data)) + data)
    body = b"WAVE" + b"".join(chunks)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a PCM (8/16/24/32-bit) or IEEE float WAV file as float32 [channels, samples].

    Integer samples are scaled by 2^(bits-1), as torchaudio and libsndfile do.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    fmt = data = None
    pos = 12
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path} has no fmt or data chunk")
    tag, channels, rate, _, block, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    width = bits // 8
    if channels < 1 or block != channels * width:
        raise ValueError(f"{path}: unsupported WAV layout ({channels} channels, {bits} bits)")
    frames = len(data) // block
    data = data[:frames * block]
    if tag == _FLOAT and bits in (32, 64):
        samples = np.frombuffer(data, f"<f{width}").astype(np.float32)
    elif tag == _PCM and bits == 8:
        samples = (np.frombuffer(data, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == _PCM and bits in (16, 32):
        samples = np.frombuffer(data, f"<i{width}").astype(np.float32) / float(2 ** (bits - 1))
    elif tag == _PCM and bits == 24:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (b[:, 0] << 8 | b[:, 1] << 16 | b[:, 2] << 24) >> 8
        samples = ints.astype(np.float32) / float(2 ** 23)
    else:
        raise ValueError(f"{path}: unsupported WAV encoding (format {tag}, {bits} bits)")
    return np.ascontiguousarray(samples.reshape(frames, channels).T), rate


def _resample_kernel(orig: int, new: int) -> tuple[np.ndarray, int]:
    base = min(orig, new) * _RESAMPLE_ROLLOFF
    width = math.ceil(_RESAMPLE_WIDTH * orig / base)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx) * base
    t = np.clip(t, -_RESAMPLE_WIDTH, _RESAMPLE_WIDTH)
    window = np.cos(t * math.pi / _RESAMPLE_WIDTH / 2) ** 2
    t *= math.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.where(t == 0, 1.0, np.sin(t) / t)
    return (kernel * window * (base / orig)).astype(np.float32), width


def resample(audio: np.ndarray, orig_rate: int, new_rate: int) -> np.ndarray:
    """Band-limited sinc resampling with a Hann window, equal to torchaudio's default Resample."""
    if orig_rate == new_rate:
        return np.asarray(audio, np.float32)
    gcd = math.gcd(int(orig_rate), int(new_rate))
    orig, new = int(orig_rate) // gcd, int(new_rate) // gcd
    kernel, width = _resample_kernel(orig, new)
    audio = np.asarray(audio, np.float32)
    length = audio.shape[-1]
    padded = np.pad(audio.reshape(-1, length), ((0, 0), (width, width + orig)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape[1], axis=-1)[:, ::orig]
    out = (frames @ kernel.T).reshape(padded.shape[0], -1)[:, :math.ceil(new * length / orig)]
    return np.ascontiguousarray(out.reshape(*audio.shape[:-1], -1), np.float32)


def to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Mono is duplicated, extra channels are dropped (stereo) or averaged (mono), as in the reference."""
    audio = np.asarray(audio, np.float32)
    if audio.ndim == 1:
        audio = audio[None]
    if channels == 1:
        return audio.mean(0, keepdims=True)
    if audio.shape[0] == 1:
        return np.repeat(audio, channels, 0)
    return audio[:channels]


def fit_length(audio: np.ndarray, samples: int) -> np.ndarray:
    """Crop or zero-pad [channels, n] audio to `samples`, keeping the start."""
    out = np.zeros((audio.shape[0], samples), np.float32)
    n = min(samples, audio.shape[-1])
    out[:, :n] = audio[:, :n]
    return out


def prepare(audio: Audio, sample_rate: int, channels: int, samples: int | None = None) -> np.ndarray:
    """Resample, convert channels and optionally pad/crop, as the reference prepares audio inputs."""
    out = to_channels(resample(audio.samples, audio.sample_rate, sample_rate), channels)
    return out if samples is None else fit_length(out, samples)
