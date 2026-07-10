"""SA3 text-to-audio inference pipeline for TFLite / LiteRT (CPU) — trimmed to just
what the baked-I/O CLI (scripts/sa3_tflite.py) needs.

Pipeline:  prompt → (SentencePiece) → T5Gemma TFLite → conditioning (baked in-graph)
           → DiT pingpong (8 steps, rectified-flow, host-side numpy) → decoder → WAV

The baked-I/O DiT bakes the conditioner (seconds embedder + prompt padding) and
patch/unpatch into the graph, so this module only needs: the tokenizer, the T5Gemma
front-end, the pingpong schedule, the noise maker, the host-side sampler, and a WAV
writer. The research-only backends (MLX DiT A/B, per-precision model dicts, the numpy
Conditioner / unpatch, encode_prompt / decode_with helpers) live in the speed-metal
repo's tflite_pipeline.py and are intentionally dropped here.
"""
from __future__ import annotations
import wave
from pathlib import Path
from typing import Callable
import numpy as np

# This file lives in <project>/models/defs/. The bundled SentencePiece model sits at
# <project>/models/tokenizer.model — resolve it relative to this file so the tokenizer
# works regardless of the caller's cwd.
DEFS_DIR = Path(__file__).resolve().parent
MODELS_DIR = DEFS_DIR.parent
TOKENIZER_MODEL = MODELS_DIR / "tokenizer.model"

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096          # decoder upsample (256 patch × 16)
COND_TOKENS = 256                  # T5Gemma seq len


# ───────────────────────── WAV ─────────────────────────
def save_wav(path, audio):  # audio: (2, T) float32 in [-1,1]
    audio = np.clip(np.asarray(audio, np.float32), -1, 1)
    pcm = (audio * 32767.0).astype(np.int16).T  # (T, 2) interleaved
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0]); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


# ───────────────────────── Tokenizer (SentencePiece, bundled) ─────────────────────────
class Tokenizer:
    """SentencePiece front-end. Loads the model from the BUNDLED models/tokenizer.model
    (the .tflite T5Gemma is encoder-only and carries no tokenizer). Encode(prompt)[:256],
    pad=0 — reproduces the SA3 groundtruth input_ids exactly."""
    def __init__(self, model_path=TOKENIZER_MODEL):
        import sentencepiece as spm
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"tokenizer model not found at {model_path}. It should ship with this "
                f"release under models/tokenizer.model."
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromFile(str(model_path))
        self.pad = 0

    def __call__(self, prompt: str, max_len: int = COND_TOKENS):
        ids = np.full((1, max_len), self.pad, np.int32)
        mask = np.zeros((1, max_len), np.int32)
        toks = self.sp.Encode(prompt)[:max_len]
        ids[0, :len(toks)] = toks
        mask[0, :len(toks)] = 1
        return ids, mask


# ───────────────────────── Pingpong schedule (numpy port) ─────────────────────────
def _logsnr_shift(t, anchor=-6.2, end=2.0):
    t = t.astype(np.float32)
    logsnr = end - t * (end - anchor)
    out = 1.0 / (1.0 + np.exp(logsnr))            # sigmoid(-logsnr)
    out = np.where(t <= 0, 0.0, out)
    out = np.where(t >= 1, 1.0, out)
    return out.astype(np.float32)


def build_pingpong_schedule(steps, sigma_max=1.0):
    # LogSNR pingpong schedule of (steps+1) points from sigma_max down to 0: warp the normalized
    # [1→0] grid through the logSNR shift, then scale by sigma_max. Decreases monotonically
    # sigma_max→0 across all steps, with the first step exactly at sigma_max to match the init
    # mix. sigma_max=1.0 is plain text-to-audio; sigma_max<1.0 is the audio-to-audio start.
    t = _logsnr_shift(np.linspace(1.0, 0.0, steps + 1).astype(np.float32)) * np.float32(sigma_max)
    t[0] = np.float32(sigma_max)
    return t


# ───────────────────────── TFLite helper ─────────────────────────
def _interp(path, threads=8):
    from ai_edge_litert import interpreter as tfl
    it = tfl.Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


class T5GemmaTFLite:
    """T5Gemma encoder (fixed 256 text tokens). ids/mask int32 [1,256] → last_hidden [1,256,768] fp32."""
    def __init__(self, path, threads=8):
        self.it = _interp(path, threads)
        det = sorted(self.it.get_input_details(), key=lambda d: d["name"])  # args_0=ids, args_1=mask
        self.i_ids, self.i_mask = det[0]["index"], det[1]["index"]
        self.out = self.it.get_output_details()[0]["index"]

    def __call__(self, ids, mask):
        self.it.set_tensor(self.i_ids, ids.astype(np.int32))
        self.it.set_tensor(self.i_mask, mask.astype(np.int32))
        self.it.invoke()
        return self.it.get_tensor(self.out).copy()                         # (1,256,768) fp32


# ───────────────────────── Sampler (shared, numpy) ─────────────────────────
def make_noise(T_lat, steps, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((1, 256, T_lat)).astype(np.float32)
    step_noise = [rng.standard_normal((1, 256, T_lat)).astype(np.float32) for _ in range(steps)]
    return x0, step_noise


def sample(dit_forward: Callable, x0, step_noise, sigmas, cross, gcond, on_step=None,
           paste_back=None):
    """Rectified-flow pingpong. dit_forward(x,t,cross,gcond)->v (the velocity; CFG,
    if any, is folded into dit_forward's return so this stays cfg-agnostic — matches
    sa3_trt_core.sample_flow_pingpong, where model_fn returns cfg_v). cross/gcond are
    passed through to dit_forward (the baked DiT ignores them — conditioning is in-graph).

    paste_back=(init_lat, keep_mask): after every step, restore the preserved region
    (keep_mask 1=keep init, 0=regenerate) so inpainting leaves untouched regions
    bit-exact. Mirrors sample_flow_pingpong's paste_back (applied post-renoise)."""
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
