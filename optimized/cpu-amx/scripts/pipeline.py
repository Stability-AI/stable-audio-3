"""SA3 text-to-audio pipeline — numpy host side, self-contained (torch-free).

This is the cpu-amx sibling of optimized/mlx/models/defs/sa3_pipeline.py and the
`tflite_pipeline` module: the fp32 pre/post and the pingpong rectified-flow
sampler are pure numpy, so the heavy compute (T5Gemma, DiT, decoder) can be any
pluggable backend. Here the backends are the torch-free C++ AMX engines
(see backends.py); this module never imports torch, mlx or tflite.

Pipeline:
    prompt -> SentencePiece -> T5Gemma -> conditioning (cross_attn + global)
           -> DiT pingpong (rectified-flow) -> SAME-S / SAME-L decoder -> WAV

The DiT is a plug-in callable ``cond_backend(x, t, cross, gcond) -> velocity``.
The C++ ``DiTCppAmx`` instance is exactly such a callable, so wiring it into
``sample`` / ``sample_cfg`` gives every feature (CFG, APG, negative prompt,
audio-to-audio, inpaint paste-back) for free.

Numerics are a verbatim port of tflite_pipeline.py (which matches sa3_mlx.py /
the TensorRT release to ~90 dB), so a cpu-amx generation is the same music as
the other releases for a given prompt/seed (up to backend precision).
"""
from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np

# ── constants (shared across the whole SA3 family) ──────────────────────────
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096          # decoder upsample (256 patch x 16)
COND_TOKENS = 256                  # T5Gemma sequence length

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TOKENIZER_NPZ = ASSETS / "t5gemma_f16.npz"     # SentencePiece model lives in here
COND_NPZ = ASSETS / "cond_medium.npz"          # learned padding + seconds embedder


# ── WAV I/O ─────────────────────────────────────────────────────────────────
def save_wav(path, audio, sample_rate: int = SAMPLE_RATE):
    """audio: (channels, T) float32 in [-1, 1]. Writes 16-bit PCM WAV."""
    audio = np.asarray(audio, np.float32)
    if not np.isfinite(audio).all():
        n_bad = int((~np.isfinite(audio)).sum())
        raise RuntimeError(f"refusing to write WAV — {n_bad} non-finite samples (NaN/Inf)")
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).T  # (T, channels) interleaved
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path) -> np.ndarray:
    """Read a WAV. Returns (2, T) float32 in [-1, 1].

    16-bit PCM @ 44.1 kHz is read natively; any other format (24/32-bit,
    48 kHz, mp3/flac) falls back to ffmpeg. Mono is duplicated to stereo."""
    path = str(path)
    try:
        with wave.open(path, "rb") as w:
            nch, sw, sr, nframes = (w.getnchannels(), w.getsampwidth(),
                                    w.getframerate(), w.getnframes())
            if sr == SAMPLE_RATE and sw == 2:
                raw = np.frombuffer(w.readframes(nframes), np.int16).astype(np.float32) / 32767.0
                if nch == 1:
                    return np.stack([raw, raw], axis=0)
                return raw.reshape(-1, nch).T[:2]
    except wave.Error:
        pass
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-"],
            capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"{path}: unsupported WAV format and ffmpeg is not installed.\n"
            f"Install ffmpeg, or convert to 16-bit/44.1 kHz stereo first.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{path}: ffmpeg failed — {e.stderr.decode().strip()}")
    raw = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32767.0
    return raw.reshape(-1, 2).T


# ── Tokenizer (SentencePiece from the bundled npz) ──────────────────────────
class Tokenizer:
    def __init__(self, npz_path=TOKENIZER_NPZ):
        import sentencepiece as spm
        arrs = np.load(npz_path)
        if "TOKENIZER_MODEL" not in arrs.files:
            raise ValueError(f"{npz_path} missing TOKENIZER_MODEL")
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromSerializedProto(arrs["TOKENIZER_MODEL"].tobytes())
        self.pad = 0

    def __call__(self, prompt: str, max_len: int = COND_TOKENS):
        ids = np.full((1, max_len), self.pad, np.int32)
        mask = np.zeros((1, max_len), np.int32)
        toks = self.sp.Encode(prompt or "")[:max_len]
        ids[0, :len(toks)] = toks
        mask[0, :len(toks)] = 1
        return ids, mask


# ── Conditioner (numpy port of sa3_pipeline) ────────────────────────────────
def _expo_fourier(norm, dim=256, min_freq=0.5, max_freq=10000.0):
    norm = np.asarray(norm, np.float32).reshape(-1, 1)
    half = dim // 2
    ramp = np.arange(half, dtype=np.float32) / max(half - 1, 1)
    freqs = np.exp(ramp * (math.log(max_freq) - math.log(min_freq)) + math.log(min_freq))
    args = norm * freqs * 2 * math.pi
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1).astype(np.float32)


class Conditioner:
    """Loads cond.{padding_embedding,seconds_total_weight,seconds_total_bias}."""
    def __init__(self, npz=COND_NPZ):
        z = np.load(npz)
        self.pad = z["cond.padding_embedding"].astype(np.float32)     # (768,)
        self.W = z["cond.seconds_total_weight"].astype(np.float32)    # (768, 256)
        self.b = z["cond.seconds_total_bias"].astype(np.float32)      # (768,)

    def seconds_embed(self, seconds, min_val=0.0, max_val=384.0):
        s = np.clip(np.float32(seconds), min_val, max_val)
        norm = (s - min_val) / (max_val - min_val)
        ff = _expo_fourier([norm], dim=256)
        return (ff @ self.W.T + self.b)[:, None, :].astype(np.float32)  # (1,1,768)

    def build(self, last_hidden, mask, seconds):
        """last_hidden (1,256,768), mask (1,256) -> cross (1,257,768), global (1,768)."""
        m = mask.astype(np.float32)[..., None]
        padded = last_hidden * m + self.pad.reshape(1, 1, -1) * (1 - m)
        se = self.seconds_embed(seconds)                               # (1,1,768)
        cross = np.concatenate([padded, se], axis=1).astype(np.float32)
        gcond = se[:, 0, :].astype(np.float32)
        return cross, gcond


# ── Pingpong schedule (numpy port; monotonic a2a rebuild for sigma_max<1) ────
def _logsnr_shift(t, anchor=-6.2, end=2.0):
    t = t.astype(np.float32)
    logsnr = end - t * (end - anchor)
    out = 1.0 / (1.0 + np.exp(logsnr))
    out = np.where(t <= 0, 0.0, out)
    out = np.where(t >= 1, 1.0, out)
    return out.astype(np.float32)


def build_pingpong_schedule(steps, sigma_max=1.0):
    """(steps+1) sigmas from sigma_max down to 0. Warp the normalized [1->0] grid
    through the logSNR shift, then scale by sigma_max — monotonic, first step
    exactly at sigma_max. sigma_max=1.0 is plain text-to-audio (bit-identical to
    upstream); sigma_max<1.0 is the audio-to-audio start."""
    t = _logsnr_shift(np.linspace(1.0, 0.0, steps + 1).astype(np.float32)) * np.float32(sigma_max)
    t[0] = np.float32(sigma_max)
    return t


# ── patch/unpatch (encoder patch grid) ──────────────────────────────────────
def patch_audio(audio: np.ndarray, patch_size: int = 256) -> np.ndarray:
    """Patched-pretransform encode: (B, 2, T_audio) -> (B, 512, T_audio/256)."""
    B, C, T = audio.shape
    assert T % patch_size == 0, f"audio length {T} not a multiple of {patch_size}"
    L = T // patch_size
    x = audio.reshape(B, C, L, patch_size).transpose(0, 1, 3, 2)
    return x.reshape(B, C * patch_size, L)


# ── Sampler (shared, numpy) ─────────────────────────────────────────────────
def make_noise(T_lat, steps, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((1, 256, T_lat)).astype(np.float32)
    step_noise = [rng.standard_normal((1, 256, T_lat)).astype(np.float32) for _ in range(steps)]
    return x0, step_noise


def _cfg_velocity(x, tc, cond_v, uncond_v, cfg_scale, apg=0.0):
    """Combine cond/uncond velocities in denoised space (RF), guide, map back.
    Mirrors sa3_mlx.model_fn. apg>0 = Adaptive Projected Guidance (project the
    cond-uncond diff orthogonal to cond_denoised). All fp32."""
    x = x.astype(np.float32)
    sigma = np.float32(tc)
    cond_d = x - cond_v.astype(np.float32) * sigma
    uncond_d = x - uncond_v.astype(np.float32) * sigma
    diff = cond_d - uncond_d
    if apg <= 0.0 or cfg_scale < 1.0:
        # cfg<1 is the interpolation regime (0 = pure uncond/neg branch); APG's
        # orthogonal projection only applies to the extrapolation regime cfg>1.
        cfg_diff = diff
    else:
        norm = np.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
        unit = cond_d / np.maximum(norm, 1e-8)
        parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
        diff_orth = diff - parallel
        cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
    cfg_d = cond_d + (cfg_scale - 1.0) * cfg_diff
    if sigma == 0:
        return np.zeros_like(x)
    return ((x - cfg_d) / sigma).astype(np.float32)


def sample(dit_forward, x0, step_noise, sigmas, cross, gcond,
           on_step=None, paste_back=None):
    """Rectified-flow pingpong. dit_forward(x,t,cross,gcond)->v.

    paste_back=(init_lat, keep_mask): after every step restore the preserved
    region (keep_mask 1=keep init, 0=regenerate) so inpainting leaves untouched
    regions bit-exact."""
    steps = len(sigmas) - 1
    x = x0.copy()
    for i in range(steps):
        tc, tn = float(sigmas[i]), float(sigmas[i + 1])
        v = dit_forward(x, tc, cross, gcond)
        denoised = x - tc * v
        if i < steps - 1 and tn > 0:
            x = (1 - tn) * denoised + tn * step_noise[i]
        else:
            x = denoised
        if paste_back is not None:
            init_lat, keep_mask = paste_back
            x = init_lat * keep_mask + x * (1.0 - keep_mask)
        if on_step:
            on_step(i + 1, steps)
    return x


def sample_cfg(cond_backend, x0, step_noise, sigmas, cross, gcond, null_cross,
               cfg_scale, apg=0.0, batched=False, on_step=None, paste_back=None):
    """Rectified-flow pingpong WITH classifier-free guidance.

    cfg_scale == 1.0 -> no uncond branch (identical to sample()).
    cfg_scale != 1.0 -> per step, evaluate cond and uncond velocities and guide.

    batched is accepted for signature parity but must be False here — there is no
    batch-2 C++ DiT, so CFG is a sequential dual-pass (two batch=1 forwards)."""
    if cfg_scale == 1.0:
        return sample(cond_backend, x0, step_noise, sigmas, cross, gcond,
                      on_step=on_step, paste_back=paste_back)
    if batched:
        raise ValueError("cpu-amx has no batch-2 DiT; call sample_cfg(..., batched=False)")

    steps = len(sigmas) - 1
    x = x0.copy()
    for i in range(steps):
        tc, tn = float(sigmas[i]), float(sigmas[i + 1])
        cond_v = cond_backend(x, tc, cross, gcond)
        uncond_v = cond_backend(x, tc, null_cross, gcond)
        v = _cfg_velocity(x, tc, cond_v, uncond_v, cfg_scale, apg)
        denoised = x - tc * v
        if i < steps - 1 and tn > 0:
            x = (1 - tn) * denoised + tn * step_noise[i]
        else:
            x = denoised
        if paste_back is not None:
            init_lat, keep_mask = paste_back
            x = init_lat * keep_mask + x * (1.0 - keep_mask)
        if on_step:
            on_step(i + 1, steps)
    return x
