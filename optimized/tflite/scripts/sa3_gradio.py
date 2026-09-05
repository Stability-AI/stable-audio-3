"""SA3 TFLite — gradio web UI (portable CPU / XNNPACK).

The TFLite sibling of optimized/mlx/scripts/sa3_gradio.py, driving the baked-I/O
varlen TFLite models (scripts/sa3_tflite.py) instead of MLX. Every generation mode
is wired, plus a precision picker the MLX/TRT UIs don't need:
  - Model picker: sm-music / sm-sfx / medium (hot-swap; the large DiT interpreter
    is cached — first use of a model/precision/length loads + XNNPACK-packs the
    weights, subsequent runs reuse it and only re-bind the conditioning).
  - Precision picker: fp32 / w16a32 / w8a32 / w8a8-dyn (one knob for DiT + codec +
    encoder, exactly like the CLI's --precision). fp32 is the CPU fast-and-accurate
    default; w16a32 is ≈lossless half-size; w8a32/w8a8-dyn are GPTQ int8 (¼ size).
  - CFG 0-10 next to seconds/steps (0 = negative prompt takes over, 0.5 = halfway
    between prompts, 1 = off, >1 = extrapolate) + negative prompt/APG under Advanced.
  - Audio-to-audio: guide audio + init_noise_level (whole clip starts from its
    latents); global sigma_max under Advanced for prompt-only generations.
  - Inpainting: separate reference audio + start/end range sliders (kept bit-exact
    outside the range). Combinable with a2a.
  - LoRA: file upload + local-folder pick (loras/<model>/) + strength, per-model
    memory, add/remove. Merged once into the DiT weights at load (the frozen graph
    has no per-step gating — so no Min/Max-step sliders, unlike the MLX UI). Only
    fp32 / w16a32 can be LoRA-merged; the panel disables itself under int8.
  - Spectrogram-as-player (numpy port, white playhead, click-to-seek), history
    panel, Output options (Auto-play / Auto-download / Infinite Radio / Loop /
    Hotswap), MP3 (via ffmpeg) or WAV saving.

Launch:
    ./sa3-gradio                  # share=True by default, sm-music + same-s, fp32
    ./sa3-gradio --dit medium --precision w8a32
    ./sa3-gradio --no-share       # local-only
"""
from __future__ import annotations
import argparse
import base64
import gc
import html as html_lib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import wave
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO))          # so `from models.defs.* import` resolves
sys.path.insert(0, str(SCRIPTS_DIR))   # so `from weights / lora_* / spec import` resolves

from sa3_tflite import (  # noqa: E402
    BakedDiT, RungEncoder, RungDecoder, read_wav,
    valid_T_lat, DEFAULT_DECODER, DIT_REL, DEC_REL, T5_REL,
    COND_TOKENS, COND_DIM, SAMPLE_RATE, SAMPLES_PER_LATENT,
    RUNG_TRIM, MIN_SIGMA,
)
from models.defs import tflite_pipeline as P   # noqa: E402
from lora_core import parse_lora_spec, LoraError  # noqa: E402
from lora_patch import get_patched_dit  # noqa: E402
from weights import ensure_local, PRECISIONS, dit_rel, dec_rel, enc_rel  # noqa: E402
from spec import render_spectrogram_png  # noqa: E402

OUTPUT_DIR = REPO / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Local LoRA library: drop .safetensors under loras/<model>/ and they appear in
# the per-slot dropdown (scanned per selected DiT; hidden when empty).
LORAS_DIR = REPO / "loras"
LORA_DIR_NAMES = {"medium": "sa3-medium", "sm-sfx": "sa3-sm-sfx",
                  "sm-music": "sa3-sm-music"}
for _d in LORA_DIR_NAMES.values():
    (LORAS_DIR / _d).mkdir(parents=True, exist_ok=True)
_DD_NONE = ""   # dropdown placeholder value ("--- Choose a LoRA ---")

# Precisions whose weights can be LoRA-merged (fp32 consts + w16a32 fp16-behind-
# DEQUANTIZE). Quantized-int8 graphs (w8a32 / w8a8-dyn) can't — see lora_patch.
LORA_PRECISIONS = ("fp32", "w16a32")
# Trained max clip length per model (repo README model table).
MAX_SECONDS = {"sm-music": 120, "sm-sfx": 120, "medium": 380}
DEFAULT_DECODERS = dict(DEFAULT_DECODER)
# XNNPACK CPU threads (set from --threads in main()).
_THREADS = 8


def _lora_dd_choices(dit_name: str) -> list:
    """Dropdown choices for ./loras/<model>/: placeholder + (name, abspath)."""
    d = LORAS_DIR / LORA_DIR_NAMES.get(dit_name, dit_name)
    hits = sorted(d.glob("*.safetensors")) if d.is_dir() else []
    return [("--- Choose a LoRA ---", _DD_NONE)] + [(p.name, str(p)) for p in hits]


# MP3 (V0) saving needs ffmpeg; without it we save WAV and hide the choice.
FFMPEG = shutil.which("ffmpeg") is not None
FORMAT_MP3, FORMAT_WAV = "Save to MP3 (V0)", "Save to WAV"

# The baked TFLite models are frozen FlatBuffer graphs, so unlike MLX there is no
# ThreadLocalStream hazard — but a single XNNPACK Interpreter must NOT be invoked
# from two threads at once, and gradio runs each handler on a rotating anyio worker
# thread. The categorical fix (same structure as the MLX UI): ALL model work
# (pre-warm + every generation) runs on one dedicated owner thread; handlers submit
# to it and wait. This also serializes generations.
import concurrent.futures as _cf  # noqa: E402
_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tfl")


def run_serial(fn, *args, **kwargs):
    """Run fn on the owner thread and return its result (re-raises errors)."""
    return _EXECUTOR.submit(fn, *args, **kwargs).result()


def read_audio_any(path: str) -> np.ndarray:
    """(2, T) float32 @ 44.1 kHz from any common upload format.

    Layered: read_wav handles 16-bit/44.1k natively and shells out to ffmpeg
    for the rest when installed; if that fails (no ffmpeg), soundfile's bundled
    libsndfile decodes mp3/flac/ogg/24-bit/48 kHz directly, followed by a
    linear resample to 44.1 kHz."""
    try:
        return read_wav(path)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)   # (T, ch)
        y = data.T
        y = np.stack([y[0], y[0]]) if y.shape[0] == 1 else y[:2]
        if sr != SAMPLE_RATE:
            in_len = y.shape[-1]
            new_len = int(round(in_len * SAMPLE_RATE / sr))
            scale = in_len / new_len
            pos = np.clip((np.arange(new_len) + 0.5) * scale - 0.5, 0, in_len - 1)
            y = np.stack([np.interp(pos, np.arange(in_len), ch) for ch in y])
        return np.ascontiguousarray(y, dtype=np.float32)


def condense_prompt(prompt: str) -> str:
    """Prompt → filename fragment (the main repo's verbose-naming rule):
    filesystem-special characters become hyphens, capped at 150 chars."""
    prompt = re.sub(r'[\\/:*?"<>|]', '-', prompt)[:150]
    return prompt or "_"


def verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed, precision) -> str:
    """prompt[.neg-…][.cfg{scale}][.smx{σ}][.{precision}].{seed} — the main repo's
    gradio 'verbose' file naming, plus a precision tag for non-fp32 runs so A/B
    runs across --precision don't overwrite each other."""
    base = condense_prompt(prompt)
    if negative_prompt and negative_prompt.strip():
        base += ".neg-" + condense_prompt(negative_prompt.strip())
    if cfg != 1.0:
        base += f".cfg{cfg:g}"
    if sigma_max != 1.0:
        base += f".smx{sigma_max:g}"
    if precision and precision != "fp32":
        base += f".{precision}"
    return f"{base}.{seed}"


def _save_wav(pcm_int16, out_path):
    """pcm_int16: (T, 2) int16 interleaved."""
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_int16.tobytes())


# ── Model caches ─────────────────────────────────────────────────────────────
# The T5Gemma front-end is loaded once (fixed 256 tokens). The DiT interpreter is
# large (medium fp32 ≈ 5.8 GB) and holds its conditioning as RESIDENT tensors, so
# it's cached by FILE PATH (dit family + precision, or the LoRA-patched clone) and
# re-bound per generation via set_conditioning — resize+allocate only when the
# batch (cfg) or length changes. Codec enc/dec are cached by (name, precision).
_t5 = None
_tok = None
_dit_cache: dict[str, BakedDiT] = {}
_dit_lru: list[str] = []
_DIT_CACHE_MAX = 1          # one big DiT resident at a time (medium fp32 = 5.8 GB)
_dec_cache: dict[tuple, RungDecoder] = {}
_dec_lru: list[tuple] = []
_enc_cache: dict[tuple, RungEncoder] = {}
_enc_lru: list[tuple] = []
_CODEC_CACHE_MAX = 2
# Encoded-latent cache: rerunning a2a/inpainting with the same input audio skips
# the encoder. Keyed on the file's CONTENT hash (gradio re-uploads get fresh temp
# paths) + codec + precision + latent length.
_latent_cache: dict[tuple, np.ndarray] = {}
_LATENT_CACHE_MAX = 32


def get_t5():
    global _t5, _tok
    if _t5 is None:
        _t5 = P.T5GemmaTFLite(ensure_local(T5_REL), _THREADS)
        _tok = P.Tokenizer()
    return _t5, _tok


def get_dit(dit_path, T_lat, t5_hidden, t5_mask, seconds, cfg, apg,
            null_h, null_m, local_add_cond, batched):
    """Cache the (large) baked DiT interpreter by its file path; re-bind the
    per-generation conditioning via set_conditioning. Returns (model, load_ms)."""
    key = str(dit_path)
    if key in _dit_cache:
        if key in _dit_lru:
            _dit_lru.remove(key)
        _dit_lru.append(key)
        model = _dit_cache[key]
        model.set_conditioning(T_lat, t5_hidden, t5_mask, seconds, cfg=cfg, apg=apg,
                               null_hidden=null_h, null_mask=null_m,
                               local_add_cond=local_add_cond, batched=batched)
        return model, 0.0
    while len(_dit_cache) >= _DIT_CACHE_MAX:
        old = _dit_lru.pop(0)
        print(f"  ← LRU-evicting DiT {Path(old).name}")
        _dit_cache.pop(old, None)
        gc.collect()
    t0 = time.time()
    model = BakedDiT(dit_path, T_lat, t5_hidden, t5_mask, seconds, _THREADS,
                     cfg=cfg, apg=apg, null_hidden=null_h, null_mask=null_m,
                     local_add_cond=local_add_cond, batched=batched)
    load_ms = (time.time() - t0) * 1000
    _dit_cache[key] = model
    _dit_lru.append(key)
    return model, load_ms


def _codec_prec(precision: str) -> str:
    """Map the (DiT-style) precision from the UI to the codec's 2 rung tiers: fp32 -> fp32, any int8 -> w8a8."""
    return "fp32" if precision == "fp32" else "w8a8"


def get_decoder(name: str, precision: str) -> RungDecoder:
    cprec = _codec_prec(precision)
    key = (name, cprec)
    if key in _dec_cache:
        _dec_lru.remove(key)
        _dec_lru.append(key)
        return _dec_cache[key]
    while len(_dec_cache) >= _CODEC_CACHE_MAX:
        old = _dec_lru.pop(0)
        _dec_cache.pop(old, None)
        gc.collect()
    dec = RungDecoder(ensure_local(dec_rel(name, cprec)), threads=_THREADS, trim=RUNG_TRIM[name])
    _dec_cache[key] = dec
    _dec_lru.append(key)
    return dec


def get_encoder(name: str, precision: str) -> RungEncoder:
    cprec = _codec_prec(precision)
    key = (name, cprec)
    if key in _enc_cache:
        _enc_lru.remove(key)
        _enc_lru.append(key)
        return _enc_cache[key]
    while len(_enc_cache) >= _CODEC_CACHE_MAX:
        old = _enc_lru.pop(0)
        _enc_cache.pop(old, None)
        gc.collect()
    enc = RungEncoder(ensure_local(enc_rel(name, cprec)), threads=_THREADS, trim=RUNG_TRIM[name])
    _enc_cache[key] = enc
    _enc_lru.append(key)
    return enc


# ── One full generation (mirrors sa3_tflite.py main(), all modes) ─────────────
def run_generation(dit_name: str, decoder_name: str, precision: str, prompt: str,
                   negative_prompt: str, seconds: float, steps: int,
                   seed: int, cfg: float, apg: float, sigma_max: float,
                   a2a_audio_path: str | None = None,
                   inpaint_audio_path: str | None = None,
                   inpaint_range_sec=None,
                   lora_specs=None, cfg_batched: bool = True):
    """Returns (audio_np (2,T) float32, timings dict).

    a2a and inpainting are independent and combinable (same as the MLX UI):
      - a2a_audio_path: the whole generation STARTS from this audio's latents
        (x0 = lat*(1-σmax) + noise*σmax)
      - inpaint_audio_path + inpaint_range_sec: kept bit-exact outside the range
        (local_add_cond context + per-step paste-back)
      - both: the inpainted span regenerates FROM the a2a guide instead of pure noise
    """
    t = {}
    dec = decoder_name
    needs_even = (dec == "same-s")
    T_lat = valid_T_lat(seconds)

    # 1. T5Gemma (tokenize + encode; + negative prompt for CFG)
    t0 = time.time()
    t5, tok = get_t5()
    ids, mask = tok(prompt)
    t5_hidden = t5(ids, mask)
    null_h = null_m = None
    if cfg != 1.0:
        if negative_prompt and negative_prompt.strip():
            n_ids, n_mask = tok(negative_prompt.strip())
            null_h = t5(n_ids, n_mask)
            null_m = n_mask.astype(np.float32)
        else:
            # All-zero hidden+mask → in-graph conditioner emits learned padding
            # embeds for every position (the standard unconditional branch).
            null_h = np.zeros((1, COND_TOKENS, COND_DIM), np.float32)
            null_m = np.zeros((1, COND_TOKENS), np.float32)
    t["t5_ms"] = (time.time() - t0) * 1000

    # 2. Encode the provided audio inputs → latents (each optional). Memoized by
    #    (file content, codec, precision, T_lat) so re-running with the same audio
    #    skips the encoder.
    def encode_audio(path):
        import hashlib
        ck = (hashlib.sha1(Path(path).read_bytes()).hexdigest(), dec, precision, T_lat)
        if ck in _latent_cache:
            t["enc_cached"] = t.get("enc_cached", 0) + 1
            return _latent_cache[ck]
        enc = get_encoder(dec, precision)
        # SAME-S encoder needs even L; round the encode grid up, trim to T_lat.
        enc_L = T_lat + 1 if (needs_even and T_lat % 2 != 0) else T_lat
        target = enc_L * SAMPLES_PER_LATENT
        audio_np = read_audio_any(path)
        if audio_np.shape[-1] >= target:
            audio_np = audio_np[:, :target]
        else:
            audio_np = np.pad(audio_np, ((0, 0), (0, target - audio_np.shape[-1])))
        lat = enc.encode(audio_np[None])[:, :, :T_lat]   # (1,256,T_lat); RungEncoder auto-dispatches rungs
        while len(_latent_cache) >= _LATENT_CACHE_MAX:
            _latent_cache.pop(next(iter(_latent_cache)))
        _latent_cache[ck] = lat
        return lat

    a2a_latents = None      # start-point guide (whole clip)
    ctx_latents = None      # inpaint context (kept outside the range)
    t["enc_ms"] = 0.0
    if a2a_audio_path:
        t0 = time.time()
        a2a_latents = encode_audio(a2a_audio_path)
        t["enc_ms"] += (time.time() - t0) * 1000
    if inpaint_audio_path and inpaint_range_sec is not None:
        t0 = time.time()
        ctx_latents = encode_audio(inpaint_audio_path)
        t["enc_ms"] += (time.time() - t0) * 1000

    # 3. inpaint local_add_cond + paste-back (TRT channel layout: [1,257,L])
    local_add_cond = None
    paste_back = None
    if ctx_latents is not None:
        s0 = max(0, int(round(inpaint_range_sec[0] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(T_lat, int(round(inpaint_range_sec[1] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        keep = np.ones((1, 1, T_lat), np.float32)
        keep[:, :, s0:s1] = 0.0                       # 1=keep init, 0=regenerate
        masked = ctx_latents.astype(np.float32) * keep
        local_add_cond = np.concatenate([keep, masked], axis=1)   # (1,257,T_lat)
        paste_back = (ctx_latents.astype(np.float32), keep)

    # 4. initial noise + a2a start-point mix (independent of inpainting)
    x0, step_noise = P.make_noise(T_lat, steps, seed)
    if a2a_latents is not None:
        x0 = a2a_latents.astype(np.float32) * (1.0 - sigma_max) + x0 * sigma_max

    # 5. DiT load (+ optional LoRA patch) + pingpong sample
    dit_path = ensure_local(dit_rel(dit_name, precision))
    if lora_specs:
        # Raises LoraError for quantized-int8 precisions (caught upstream → error box).
        dit_path = get_patched_dit(dit_path, lora_specs, family=dit_name,
                                   precision=precision, log=lambda m: print(f"  {m}"))
    backend, t["dit_load_ms"] = get_dit(str(dit_path), T_lat, t5_hidden,
                                        mask.astype(np.float32), float(seconds),
                                        cfg, apg, null_h, null_m, local_add_cond,
                                        cfg_batched)
    sigmas = P.build_pingpong_schedule(steps, sigma_max)
    t0 = time.time()
    latents = P.sample(backend, x0, step_noise, sigmas, None, None, paste_back=paste_back)
    t["sample_ms"] = (time.time() - t0) * 1000
    t["n_fwd"] = backend.n_fwd

    # 6. Decode — RungDecoder auto-dispatches the optimal rung per length (no chunk knob)
    decoder = get_decoder(dec, precision)
    t0 = time.time()
    dec_lat = latents
    if needs_even and T_lat % 2:                  # SAME-S needs even L: pad one latent, trim the audio
        dec_lat = np.concatenate([latents, latents[:, :, -1:]], axis=2)
    audio = decoder.decode(dec_lat)[:, :, :T_lat * SAMPLES_PER_LATENT]
    t["decode_ms"] = (time.time() - t0) * 1000

    audio_np = audio[0]                               # (2, L*4096)
    requested = int(round(seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > requested:
        audio_np = audio_np[..., :requested]

    t["T_lat"] = T_lat
    t["samples"] = audio_np.shape[-1]
    t["inference_ms"] = sum(t.get(k, 0) for k in
                            ("t5_ms", "enc_ms", "dit_load_ms", "sample_ms", "decode_ms"))
    t["realtime"] = (audio_np.shape[-1] / SAMPLE_RATE) / max(t["inference_ms"] / 1000, 1e-9)
    return audio_np, t


# ── HTML rendering (inline handlers only — gradio HTML runs attributes, not
# <script> tags) ────────────────────────────────────────────────────────────
# One-at-a-time playback. Chrome does NOT stop a playing <audio> when it's
# removed from the DOM (verified via CDP: swapped-out elements keep playing as
# detached ghosts), and querySelectorAll can't see detached nodes — so every
# player registers itself in window._sa3All on play, and pause-others sweeps
# that registry (reaching ghosts) plus the document. Entries that are detached
# AND paused drop from the registry so ghosts can be garbage-collected.
# A DETACHED element's play event must never silence the living player (the
# swap can briefly leave a superseded copy whose pending play() resolves late).
# gradio's slider track fill lives in an inline CSS var (--range_progress) that
# is only written on user input / first mount — a slider REmounted by a
# visibility toggle (LoRA slots on model switch / add) comes back without it and
# renders a blank track. Recompute it client-side after any remount batch;
# idempotent for healthy sliders (same formula gradio uses).
_JS_FIX_SLIDERS = (
    "() => setTimeout(() => {"
    "document.querySelectorAll('input[type=range]').forEach(r => {"
    "const min = +r.min || 0, max = +r.max || 100, v = +r.value;"
    "if (max > min) r.style.setProperty('--range_progress',"
    "((v - min) / (max - min) * 100) + '%');"
    "});}, 150)"
)

_JS_PAUSE_OTHERS = (
    "if(this.isConnected){"
    "var t=this;var L=window._sa3All=(window._sa3All||[]);"
    "if(L.indexOf(t)<0)L.push(t);"
    "L.forEach(function(o){if(o!==t){try{o.pause()}catch(e){}}});"
    "document.querySelectorAll('audio').forEach(function(o){if(o!==t)o.pause();});"
    "window._sa3All=L.filter(function(o){return o===t||o.isConnected;});}"
    "else{this.pause();}")
_JS_PLAYHEAD = ("var p=this.closest('.blk').querySelector('.ph');"
                "if(p&&this.duration)p.style.left=(this.currentTime/this.duration*100)+'%';")
# Main-player position ledger (for Hotswap): every timeupdate/play/pause stamps
# the current position so a freshly swapped-in element can resume exactly there.
# Guards (all verified via CDP against the live page):
#  - isConnected: teardown fires a pause AND a final timeupdate on the removed
#    element — detached elements never write the ledger.
#  - unresumed hotswap candidates (data-hs set, hsd not yet) don't write either:
#    with autoplay, the NEW element's 'play' event fires BEFORE loadedmetadata
#    and would stamp t≈0 over the old clip's position right before the resume
#    handler reads it.
_JS_POS_RECORD = ("if(this.isConnected&&!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                  "{window._sa3Pos={t:this.currentTime,playing:!this.paused,ts:Date.now()};}")
# timeupdate variant: a PAUSED element only fires timeupdate on seeks — e.g. the
# resume handler's own currentTime assignment. Stamping playing:false there
# poisons the ledger for any second render of the same clip (gradio sometimes
# renders a component update twice), which froze the handoff chain. While
# paused, only real pause events may write.
_JS_POS_RECORD_TU = ("if(this.isConnected&&!this.paused&&"
                     "!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                     "{window._sa3Pos={t:this.currentTime,playing:true,ts:Date.now()};}")
# Resilient play: Chrome's autoplay policy can reject programmatic play() once
# the transient user activation from the Generate click has expired (seconds),
# leaving the clip correctly positioned but paused. Try, retry shortly after,
# and as a last resort arm a one-shot listener so the user's next click or
# keypress anywhere resumes playback.
# Every attempt re-checks isConnected: a superseded (detached) element must
# never revive itself from a queued retry and fight the current player.
_JS_TRY_PLAY = (
    "var A=this;A.play().catch(function(){setTimeout(function(){"
    "if(!A.isConnected)return;"
    "A.play().catch(function(){console.warn('play blocked by autoplay policy — "
    "will resume on next interaction');"
    "var f=function(){if(A.isConnected)A.play();"
    "document.removeEventListener('pointerdown',f,true);"
    "document.removeEventListener('keydown',f,true);};"
    "document.addEventListener('pointerdown',f,true);"
    "document.addEventListener('keydown',f,true);});},150);});")
# Hotswap resume: if the previous main audio was playing when this one arrived,
# jump to its position (+ the split-second since the last stamp) and keep going;
# beyond the new clip's duration -> start at zero. Guarded (hsd) so it applies
# once, on whichever of loadedmetadata/canplay fires first.
_JS_HOTSWAP = ("if(this.isConnected&&this.dataset.hs==='1'&&!this.dataset.hsd&&this.duration){"
               "this.dataset.hsd=1;var s=window._sa3Pos;"
               "this.dataset.dbg=s?(s.playing?'playing@'+s.t.toFixed(2):'notplaying@'+s.t.toFixed(2)):'noledger';"
               "if(s&&s.playing){var tt=s.t+(Date.now()-s.ts)/1000;"
               "this.currentTime=(tt<this.duration)?tt:0;" + _JS_TRY_PLAY + "}}")
_JS_SEEK = ("var a=this.closest('.blk').querySelector('audio');"
            "var r=this.getBoundingClientRect();"
            "if(a&&a.duration){a.currentTime=(event.clientX-r.left)/r.width*a.duration;a.play();}")
_JS_PROMOTE = ("var b=document.getElementById('sa3-promote');"
               "if(b){(b.tagName==='BUTTON'?b:b.querySelector('button')||b).click();}")
# Scroll anchoring for the history panel: onscroll continuously records which
# item sits at the viewport top (+offset); after every re-render a hidden
# bootstrap <img onerror> restores that anchor — stick-to-top when at top,
# otherwise keep hovering over the same old item as new ones prepend.
_JS_SCROLL_RECORD = (
    "var c=this,k=null,off=0,ch=c.querySelectorAll('[data-key]');"
    "for(var i=0;i<ch.length;i++){if(ch[i].offsetTop+ch[i].offsetHeight>c.scrollTop)"
    "{k=ch[i].getAttribute('data-key');off=ch[i].offsetTop-c.scrollTop;break}}"
    "window._sa3S={atTop:c.scrollTop<8,key:k,off:off};")
_JS_SCROLL_RESTORE = (
    "var c=document.getElementById('sa3-hist');var s=window._sa3S||{};"
    "if(c){var el=s.key?c.querySelector('[data-key=&quot;'+s.key+'&quot;]'):null;"
    "if(el&&!s.atTop){c.scrollTop=el.offsetTop-(s.off||0)}else{c.scrollTop=0}}"
    "this.remove();")


def _ago(ts) -> str:
    if not ts:
        return ""
    d = max(0.0, time.time() - ts)
    if d < 10:
        return "just now"
    if d < 60:
        return f"{int(d)}s ago"
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < 86400:
        return f"{int(d // 3600)}h ago"
    return f"{int(d // 86400)}d ago"


def _lora_specs_from_ui(vals, notes):
    """Turn the 3 LoRA slots' flat values (file, dropdown, strength)×3 into
    lora_core specs. A slot's adapter is the dropped file or, failing that, the
    dropdown pick from ./loras/<model>/; empty slots are skipped. Unlike the MLX
    UI there is no per-step range (the frozen TFLite graph merges once at load).
    A bad spec gets a note and is skipped (never an error — matches the app's
    permissive-by-design generation policy)."""
    specs = []
    for i in range(3):
        path, picked, strength = vals[3 * i: 3 * i + 3]
        if not path:
            path = picked if picked and picked != _DD_NONE else None
        if not path:
            continue
        try:
            specs.append(parse_lora_spec([path, f"strength={float(strength)}"]))
        except LoraError as e:
            notes.append(f"LoRA {i + 1}: {e}")
    return specs or None


def _lora_disp(specs) -> str:
    """Short human tag per adapter: 'myband×0.8, rain'."""
    tags = []
    for s in specs or []:
        name = Path(s["path"]).stem
        if len(name) > 25:
            name = name[:24] + "…"
        tag = name
        if s["strength"] != 1.0:
            tag += f"×{s['strength']:g}"
        tags.append(tag)
    return ", ".join(tags)


def _meta_suffix(entry) -> str:
    """' · sm-mus · w8a32 · cfg 1.5 · noise 0.92 · audio2audio · inpainting' —
    only the non-defaults. The sigma tag reads 'noise' for a2a runs (it IS the
    init_noise_level there), 'smx' otherwise."""
    parts = []
    cfg = entry.get("cfg", 1.0)
    smx = entry.get("smx", 1.0)
    mode = entry.get("mode", "")
    is_a2a = "a2a" in mode or mode == "audio-to-audio"
    dit = entry.get("dit", "")
    prec = entry.get("prec", "fp32")
    if dit:
        parts.append({"medium": "med", "sm-music": "sm-mus", "sm-sfx": "sm-sfx"}.get(dit, dit))
    if prec and prec != "fp32":
        parts.append(prec)
    if cfg != 1.0:
        parts.append(f"cfg {cfg:g}")
    if smx != 1.0:
        parts.append(f"{'noise' if is_a2a else 'smx'} {smx:g}")
    if is_a2a:
        parts.append("audio2audio")
    if "inpaint" in mode:
        parts.append("inpainting")
    if entry.get("lora"):
        parts.append(f"lora {entry['lora']}")
    return "".join(f" · {p}" for p in parts)


def _neg_disp(entry) -> str:
    """Labeled negative prompt, shown only when it actually acted (cfg != 1)."""
    neg = (entry.get("neg") or "").strip()
    if neg and entry.get("cfg", 1.0) != 1.0:
        return f' · <span style="opacity:0.75">neg: {html_lib.escape(neg)}</span>'
    return ""


def render_player(entry, *, small=False, autoplay=False, autodl=False, radio=False,
                  bg=None, advance=False, loop=False, hotswap=False):
    """One self-contained player block: audio + caption + seekable spectrogram
    with playhead. Global one-at-a-time playback via onplay pause-others.
    small: audio + spectrogram side by side (half width each), 'Xm ago' caption.
    advance: on ended, hop to the next history item's audio (Auto-play).
    loop: native loop attribute — finished audio restarts (suppresses ended).
    hotswap: main slot only — resume at the previous clip's position on arrival."""
    # assemble per-event handler bodies (a duplicated attribute name would
    # silently drop one handler, so each event is emitted exactly once)
    on_canplay = []
    if small:
        attrs = f'onplay="{_JS_PAUSE_OTHERS}" ontimeupdate="{_JS_PLAYHEAD}"'
    else:
        attrs = (f'onplay="{_JS_PAUSE_OTHERS}{_JS_POS_RECORD}" '
                 f'onpause="{_JS_POS_RECORD}" '
                 f'ontimeupdate="{_JS_PLAYHEAD}{_JS_POS_RECORD_TU}"')
    if hotswap and not small:
        attrs += f' data-hs="1" onloadedmetadata="{_JS_HOTSWAP}"'
        on_canplay.append(_JS_HOTSWAP)   # fallback if loadedmetadata already passed
    if autoplay and not small:
        # the autoplay attribute is subject to the same policy — rescue a
        # policy-paused clip (radio transitions, long generations)
        on_canplay.append("if(this.paused&&!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                          "{" + _JS_TRY_PLAY + "}")
    if autodl:
        js_name = entry["name"].replace("\\", "").replace("'", "\\'")
        on_canplay.append("if(!this.dataset.dld){this.dataset.dld=1;"
                          "var l=document.createElement('a');l.href=this.src;"
                          f"l.download='{js_name}';l.click();}}")
    if on_canplay:
        attrs += f' oncanplay="{"".join(on_canplay)}"'
    if loop:
        attrs += " loop"
    elif radio:
        attrs += f' onended="{_JS_PROMOTE}"'
    elif advance:
        attrs += (' onended="var b=this.closest(\'.blk\');'
                  "var n=b?b.nextElementSibling:null;"
                  "while(n&&!n.classList.contains('blk'))n=n.nextElementSibling;"
                  "if(n){var a=n.querySelector('audio');if(a)a.play();}\"")
    auto = "autoplay " if autoplay else ""
    # Serve audio via gradio's file route instead of a data: URI — the URL ends
    # with the real (verbose) filename, so right-click "Save audio as…" offers
    # it instead of download.mp3, and the page stays light as history grows.
    src = "gradio_api/file=" + urllib.parse.quote(entry["path"], safe="/")
    audio_el = (f'<audio controls {auto}style="width:100%" {attrs} '
                f'src="{src}"></audio>')
    prompt_disp = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"

    spec_core = ""
    if entry.get("spec_b64"):
        height = "height:56px;" if small else ""
        tip = ("3-band tinted stereo mel · red=bass / green=mid / blue=high · "
               "L top, R bottom · click to seek")
        spec_core = (f'<div style="position:relative; cursor:pointer" title="{tip}" '
                     f'onclick="{_JS_SEEK}">'
                     f'<img src="data:image/png;base64,{entry["spec_b64"]}" '
                     f'style="width:100%; {height} display:block; image-rendering:pixelated; '
                     f'border:1px solid #333" alt="spectrogram"/>'
                     f'<div class="ph" style="position:absolute; top:0; bottom:0; left:0%; '
                     f'width:2px; background:#fff; pointer-events:none; '
                     f'box-shadow:0 0 4px rgba(0,0,0,.8)"></div>'
                     f'</div>')

    if small:
        row = (f'<div style="display:flex; gap:8px; align-items:center">'
               f'<div style="flex:1; min-width:0">{audio_el}</div>'
               f'<div style="flex:1; min-width:0">{spec_core}</div></div>')
        cap = (f'<div style="font-size:0.8em; margin:2px 0; color:#888">'
               f'{_ago(entry.get("ts"))} · {prompt_disp}{_neg_disp(entry)} · seed {entry["seed"]}'
               f'{_meta_suffix(entry)}</div>')
        style = f"padding:6px 8px;{' background:' + bg + ';' if bg else ''}"
        return (f'<div class="blk" data-key="{entry["key"]}" style="{style}">'
                f'{row}{cap}</div>')

    return (f'<div class="blk" data-key="{entry["key"]}">'
            f'{audio_el}{spec_core}</div>')


def render_history(hist, advance=False, loop=False):
    if not hist:
        return ""
    # zebra striping instead of separators: light grey vs medium grey rows
    items = "".join(
        render_player(e, small=True, advance=advance, loop=loop,
                      bg="rgba(127,127,127,0.24)" if i % 2 else "rgba(127,127,127,0.08)")
        for i, e in enumerate(hist))
    boot = f'<img src="data:," style="display:none" onerror="{_JS_SCROLL_RESTORE}"/>'
    return (f'<div style="font-weight:600; margin-top:14px">Previous generations ({len(hist)})</div>'
            f'<div id="sa3-hist" onscroll="{_JS_SCROLL_RECORD}" '
            f'style="max-height:480px; overflow-y:auto; position:relative; margin-top:6px; '
            f'padding-right:6px">{boot}{items}</div>')


def render_queue_status(entry=None, generating=False):
    """Subdued status chip for the Infinite Radio queue — no audio/spectrogram,
    just Generating…/Ready (the swap uses the entry held in server state)."""
    if generating:
        body = "generating…"
    elif entry is not None:
        p = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"
        body = f"ready — {p}{_neg_disp(entry)} · seed {entry['seed']}{_meta_suffix(entry)}"
    else:
        return ""
    return (f'<div style="margin-top:2px; padding:6px 10px; '
            f'background:rgba(127,127,127,0.12); border-radius:6px; '
            f'color:#888; font-size:0.85em">'
            f'<b>Queued next</b> · {body}</div>')


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui(initial_dit: str, initial_decoder: str, initial_precision: str, *,
             share: bool, default_seconds: float, default_steps: int,
             server_port: int | None = None):
    import gradio as gr
    import random as _random

    # Pre-warm the initial pipeline so the first click is fast — ON the owner
    # thread, so all model state lives where generations run.
    warm_T = valid_T_lat(default_seconds)
    print(f"  pre-warming {initial_dit}+{initial_decoder} @ {initial_precision} (T_lat={warm_T})...")

    def _warm():
        t5, tok = get_t5()
        ids, mask = tok("")
        t5h = t5(ids, mask)
        get_dit(str(ensure_local(dit_rel(initial_dit, initial_precision))), warm_T,
                t5h, mask.astype(np.float32), float(default_seconds),
                1.0, 1.0, None, None, None, True)
        get_decoder(initial_decoder, initial_precision)
    try:
        run_serial(_warm)
    except Exception as e:
        print(f"  (warm-up skipped: {type(e).__name__}: {e})")

    def on_dit_change(dit_name, cur_seconds, prev, mem, vis, *slot_vals):
        """Model switch: pair the decoder, clamp seconds, and swap the LoRA
        panel to the new model — the outgoing model's slots (files, picks,
        strengths, visibility) are snapshotted into per-model memory and the
        incoming model's snapshot is restored (or an empty panel). Dropdowns
        rescan loras/<model>/ (hidden when empty)."""
        max_s = MAX_SECONDS.get(dit_name, 120)
        ch = _lora_dd_choices(dit_name)
        valid = {v for _, v in ch}
        lbl = f"…or pick from loras/{LORA_DIR_NAMES.get(dit_name, dit_name)}/"

        mem = dict(mem or {})
        mem[prev] = {"vis": list(vis),
                     "slots": [list(slot_vals[3 * i: 3 * i + 3]) for i in range(3)]}
        snap = mem.get(dit_name)
        if snap:
            nvis = list(snap["vis"])
            slots = [list(s) for s in snap["slots"]]
            for s in slots:
                if s[1] not in valid:
                    s[1] = _DD_NONE
            # a slot with an adapter must never be an invisible-but-active one
            nvis = [v or bool(s[0] or s[1] != _DD_NONE)
                    for v, s in zip(nvis, slots)]
        else:
            nvis = [False, False, False]
            slots = [[None, _DD_NONE, 1.0] for _ in range(3)]

        dd_ups = [gr.update(choices=ch, value=slots[i][1], visible=len(ch) > 1,
                            label=lbl) for i in range(3)]
        srow_ups = [gr.update(visible=bool(slots[i][0] or slots[i][1] != _DD_NONE))
                    for i in range(3)]
        row_ups = [gr.update(visible=v) for v in nvis]
        return (gr.update(value=DEFAULT_DECODERS.get(dit_name, "same-s")),
                gr.update(maximum=max_s, value=min(cur_seconds, max_s)),
                *dd_ups, *srow_ups, *row_ups,
                gr.update(visible=not all(nvis)), nvis, dit_name, mem,
                slots[0][0], slots[1][0], slots[2][0],
                slots[0][2], slots[1][2], slots[2][2])

    def _generate_entry(dit_name, precision, decoder_name, prompt, negative_prompt,
                        seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                        a2a_audio, inpaint_audio, inp_start, inp_end,
                        output_opts, file_format, *lora_vals):
        """Run one generation and package it as a history entry.
        Returns (entry, None) or (None, error_message)."""
        prompt = (prompt or "").strip()
        # Permissive by design: a generation should succeed with whatever IS set.
        # Half-configured features are ignored with a visible note, never an error.
        notes = []
        lora_specs = (_lora_specs_from_ui(lora_vals, notes) if lora_vals else None)
        if lora_specs and precision not in LORA_PRECISIONS:
            # UI already hides LoRA under int8; this is the belt-and-braces guard for
            # API callers / stale state. get_patched_dit would raise anyway — surface
            # it as a clean note + skip the LoRA rather than failing the generation.
            notes.append(f"LoRA ignored — precision {precision} can't be LoRA-merged "
                         f"(needs fp32 or w16a32)")
            lora_specs = None
        # blank or -1 → random seed, kept small (1-9999) for readability
        try:
            seed = int(seed_text.strip()) if seed_text and seed_text.strip() else -1
        except ValueError:
            seed = -1
            notes.append("seed wasn't an integer — used a random one")
        if seed == -1:
            seed = _random.randint(1, 9999)
        # backstop for API callers bypassing the slider maxima
        max_s = MAX_SECONDS.get(dit_name, 120)
        if seconds > max_s:
            notes.append(f"seconds clamped to {dit_name}'s trained max ({max_s:g}s)")
            seconds = float(max_s)
        # sigma_max governs every generation's schedule start; when guide audio is
        # present (a2a), init_noise_level overrides it (parent-repo gradio semantics).
        sigma_max = float(init_noise) if a2a_audio else float(sigma_max)
        if sigma_max < MIN_SIGMA:
            which = "init_noise_level" if a2a_audio else "sigma_max"
            notes.append(f"{which} 0 runs at {MIN_SIGMA} (model is undefined at t≈0)"
                         + (" — output ≈ the re-encoded input" if a2a_audio else ""))
            sigma_max = MIN_SIGMA

        inpaint_range = None
        if inpaint_audio and inp_end > inp_start and inp_start < seconds:
            if inp_end > seconds:
                notes.append(f"inpaint end clamped to the clip length ({seconds:g}s)")
            inpaint_range = (float(inp_start), float(min(inp_end, seconds)))
        elif inpaint_audio:
            notes.append("inpainting ignored — set the start/end range sliders")
        elif inp_end > inp_start:
            notes.append("inpaint range ignored — no reference audio uploaded")

        opts = output_opts or []
        if ("Infinite Radio" in opts and seed_text and seed_text.strip()
                and seed_text.strip() != "-1"):
            notes.append("Infinite Radio with a fixed seed repeats the same clip — "
                         "clear the seed for endless variety")

        mode = ("a2a+inpaint" if (a2a_audio and inpaint_range) else
                "inpaint" if inpaint_range else
                "audio-to-audio" if a2a_audio else "text-to-audio")
        try:
            audio_np, t = run_serial(
                run_generation,
                dit_name, decoder_name, precision, prompt, negative_prompt or "",
                float(seconds), int(steps), seed, float(cfg), float(apg),
                float(sigma_max), a2a_audio or None, inpaint_audio or None,
                inpaint_range, lora_specs)
        except LoraError as e:
            return None, f"LoRA error: {e}"
        except Exception as e:
            return None, f"error: {type(e).__name__}: {e}"
        if not np.isfinite(audio_np).all():
            return None, "error: model produced non-finite audio (try a higher σmax or different seed)"

        pcm = np.rint(np.clip(audio_np, -1, 1) * 32767.0).astype(np.int16).T   # (T, 2); round-to-nearest (matches TRT)
        basename = verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed, precision)
        out_path = OUTPUT_DIR / f"{basename}.wav"
        _save_wav(pcm, out_path)
        mime = "audio/wav"
        if FFMPEG and file_format == FORMAT_MP3:
            mp3_path = OUTPUT_DIR / f"{basename}.mp3"
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out_path),
                                "-codec:a", "libmp3lame", "-q:a", "0", str(mp3_path)],
                               capture_output=True)
            if r.returncode == 0 and mp3_path.exists():
                out_path.unlink()
                out_path, mime = mp3_path, "audio/mpeg"
            else:
                notes.append("mp3 encode failed — saved WAV instead")

        spec_b64 = None
        try:
            spec_png = render_spectrogram_png(pcm, sample_rate=SAMPLE_RATE,
                                              width=1200, height=240)
            spec_b64 = base64.b64encode(spec_png).decode("ascii")
        except Exception as e:
            notes.append(f"spectrogram failed: {type(e).__name__}: {e}")

        load_note = (f"DiT-load {t['dit_load_ms']:.0f} ms ·&nbsp; "
                     if t.get("dit_load_ms", 0) > 100 else "")
        cached_tag = " (cached)" if t.get("enc_cached") else ""
        enc_note = f"encode {t['enc_ms']:.0f} ms{cached_tag} ·&nbsp; " if t.get("enc_ms") else ""
        prec_note = f"<b>precision</b>: {precision} ·&nbsp; " if precision != "fp32" else ""
        apg_note = f" (apg {apg:g})" if apg != 1.0 else ""
        cfg_note = f"cfg {cfg:g}{apg_note} ·&nbsp; " if cfg != 1.0 else ""
        lora_note = (f"<b>lora</b>: {html_lib.escape(_lora_disp(lora_specs))} ·&nbsp; "
                     if lora_specs else "")
        prompt_disp = html_lib.escape(prompt) or "<i>(no prompt)</i>"
        neg_disp = (f' · <span style="opacity:0.75">neg: {html_lib.escape(negative_prompt.strip())}</span>'
                    if negative_prompt and negative_prompt.strip() and cfg != 1.0 else "")
        timing_html = (
            f"{prompt_disp}{neg_disp} ·&nbsp; "
            f"{prec_note}{load_note}{enc_note}{cfg_note}{lora_note}"
            f"<b>Inference</b>: {t['inference_ms']:.0f} ms "
            f"<span style='color:#888'>(t5={t['t5_ms']:.0f} · sample={t['sample_ms']:.0f} · "
            f"decode={t['decode_ms']:.0f})</span> ·&nbsp; "
            f"<b>{t['realtime']:.1f}× realtime</b> ·&nbsp; "
            f"<b>seed</b>: <code>{seed}</code> ·&nbsp; "
            f"<b>seq_len</b>: {t['T_lat']} ·&nbsp; <b>samples</b>: {t['samples']}"
        )
        if notes:
            timing_html = ("".join(f"<div style='color:#fa3; font-size:0.85em'>note: {n}</div>"
                                   for n in notes) + timing_html)

        entry = {
            "key": f"k{time.time_ns()}",
            "ts": time.time(),
            "dit": dit_name,
            "prec": precision,
            "neg": (negative_prompt or "").strip(),
            "cfg": float(cfg),
            "smx": float(sigma_max),
            "path": str(out_path),
            "mime": mime,
            "name": out_path.name,
            "size_mb": out_path.stat().st_size / 1e6,
            "mode": mode,
            "prompt": prompt,
            "seed": seed,
            "lora": _lora_disp(lora_specs),
            "spec_b64": spec_b64,
            "timing": timing_html,
        }
        return entry, None

    def _present(state, entry, opts, queued_panel, *, force_autoplay=False):
        """Make `entry` the current clip (previous current moves to history top)
        and render all output panels."""
        if state["current"] is not None:
            state["history"].insert(0, state["current"])
        state["current"] = entry
        loop = "Loop" in opts
        main = render_player(entry,
                             autoplay=force_autoplay or "Auto-play" in opts,
                             autodl="Auto-download" in opts,
                             radio="Infinite Radio" in opts and not loop,
                             loop=loop,
                             hotswap="Hotswap" in opts)
        return (main, entry["timing"], "",
                render_history(state["history"], advance="Auto-play" in opts, loop=loop),
                queued_panel, state)

    def generate(dit_name, precision, decoder_name, prompt, negative_prompt,
                 seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                 a2a_audio, inpaint_audio, inp_start, inp_end,
                 output_opts, file_format, *lora_and_state):
        *lora_vals, state = lora_and_state
        entry, err_msg = _generate_entry(
            dit_name, precision, decoder_name, prompt, negative_prompt, seconds, steps,
            seed_text, cfg, apg, sigma_max, init_noise, a2a_audio,
            inpaint_audio, inp_start, inp_end, output_opts, file_format,
            *lora_vals)
        if entry is None:
            return (gr.update(), gr.update(),
                    f"<span style='color:#f88'>{err_msg}</span>",
                    gr.update(), gr.update(), state)
        # a manual Generate makes any queued clip stale (settings may have changed);
        # the chained pregen refills it when Infinite Radio is on
        state["queued"] = None
        opts = output_opts or []
        queued_panel = render_queue_status(generating=True) if "Infinite Radio" in opts else ""
        return _present(state, entry, opts, queued_panel)

    def promote(dit_name, precision, decoder_name, prompt, negative_prompt,
                seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                a2a_audio, inpaint_audio, inp_start, inp_end,
                output_opts, file_format, *lora_and_state):
        """Infinite Radio: swap the pre-generated clip in when playback ends.
        Falls back to a full generate if the queue is empty. Either way the
        next track ALWAYS autoplays — Auto-play only governs whether a clip
        starts when nothing was playing; radio transitions are continuations."""
        *lora_vals, state = lora_and_state
        q = state.get("queued")
        if q is None:
            entry, err_msg = _generate_entry(
                dit_name, precision, decoder_name, prompt, negative_prompt, seconds, steps,
                seed_text, cfg, apg, sigma_max, init_noise, a2a_audio,
                inpaint_audio, inp_start, inp_end, output_opts, file_format,
                *lora_vals)
            if entry is None:
                return (gr.update(), gr.update(),
                        f"<span style='color:#f88'>{err_msg}</span>",
                        gr.update(), gr.update(), state)
            q = entry
        state["queued"] = None
        return _present(state, q, output_opts or [],
                        render_queue_status(generating=True), force_autoplay=True)

    def pregen(dit_name, precision, decoder_name, prompt, negative_prompt,
               seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
               a2a_audio, inpaint_audio, inp_start, inp_end,
               output_opts, file_format, *lora_and_state):
        """Chained after generate/promote: pre-generate the NEXT clip while the
        current one plays (Infinite Radio only)."""
        *lora_vals, state = lora_and_state
        if "Infinite Radio" not in (output_opts or []):
            state["queued"] = None
            return "", state
        entry, err_msg = _generate_entry(
            dit_name, precision, decoder_name, prompt, negative_prompt, seconds, steps,
            seed_text, cfg, apg, sigma_max, init_noise, a2a_audio,
            inpaint_audio, inp_start, inp_end, output_opts, file_format,
            *lora_vals)
        if entry is None:
            return (f"<div style='color:#f88; font-size:0.85em'>queue: {err_msg}</div>",
                    state)
        state["queued"] = entry
        return render_queue_status(entry), state

    # sa3-promote must stay MOUNTED for the onended click to find it —
    # visible=False would remove it from the DOM entirely, so hide via CSS.
    # sa3-out: collapse the column's flex gap + wrapper padding so the
    # spectrogram caption / timing line / queued status sit tightly together.
    _css = ("#sa3-promote{display:none !important}"
            "#sa3-out{gap:4px !important}"
            "#sa3-out .html-container{padding:0 !important; margin:0 !important}")
    with gr.Blocks(title="SA3 TFLite") as demo:
        gr.Markdown(
            "# SA3 TFLite — portable CPU (XNNPACK)\n"
            "Text-to-audio, CFG + negative prompt, audio-to-audio, inpainting, LoRA. "
            "Pick a precision (fp32 / w16a32 / w8a32 / w8a8-dyn). First use of a "
            "model/precision loads weights; subsequent runs are cached."
        )
        st = gr.State({"current": None, "queued": None, "history": []})
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    dit_dd = gr.Dropdown(label="DiT model", choices=list(DIT_REL.keys()),
                                         value=initial_dit, scale=1)
                    precision_dd = gr.Dropdown(label="Precision", choices=list(PRECISIONS),
                                               value=initial_precision, scale=1)
                    decoder_dd = gr.Dropdown(label="Decoder (codec)",
                                             choices=list(DEC_REL.keys()),
                                             value=initial_decoder, scale=1)
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=2, scale=6,
                                        placeholder="e.g. 'Impending tribal, epic orchestral buildup'")
                    seed = gr.Textbox(label="Seed (optional)", max_lines=1, value="",
                                      scale=1, min_width=80)
                with gr.Row():
                    seconds = gr.Slider(label="Seconds", minimum=1,
                                        maximum=MAX_SECONDS.get(initial_dit, 120),
                                        value=default_seconds, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16,
                                      value=default_steps, step=1)
                    cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0,
                                    value=1.0, step=0.1)

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        apg = gr.Slider(label="APG (only applies when CFG > 1)",
                                        minimum=0.0, maximum=1.0, value=1.0, step=0.05)
                        sigma_global = gr.Slider(label="sigma_max",
                                                 minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=1)

                with gr.Accordion("LoRA", open=False):
                    # Shown only under a quantized precision (LoRA disabled there).
                    lora_note = gr.Markdown(
                        "*LoRA needs precision **fp32** or **w16a32** — quantized-int8 "
                        "graphs (w8a32 / w8a8-dyn) can't be LoRA-merged.*",
                        visible=initial_precision not in LORA_PRECISIONS)
                    with gr.Group(visible=initial_precision in LORA_PRECISIONS) as lora_editor:
                        _dd0 = _lora_dd_choices(initial_dit)
                        lora_inputs, lora_rows, lora_rm_btns = [], [], []
                        lora_dds, lora_files, lora_srows = [], [], []
                        for _i in range(1, 4):
                            with gr.Group(visible=False) as _grp:
                                with gr.Row(equal_height=True):
                                    _lf = gr.File(label=f"LoRA {_i} (.safetensors)",
                                                  file_types=[".safetensors"],
                                                  type="filepath", scale=1, height=88)
                                    _ld = gr.Dropdown(
                                        label=f"…or pick from loras/{LORA_DIR_NAMES[initial_dit]}/",
                                        choices=_dd0, value=_DD_NONE, scale=1,
                                        visible=len(_dd0) > 1)
                                    _rm = gr.Button("✕", size="sm", scale=0,
                                                    min_width=36)
                                with gr.Row(visible=False) as _srow:
                                    _ls = gr.Slider(label="strength", minimum=0.0,
                                                    maximum=10.0, value=1.0, step=0.1,
                                                    scale=2, min_width=110)
                            lora_rows.append(_grp)
                            lora_rm_btns.append(_rm)
                            lora_dds.append(_ld)
                            lora_files.append(_lf)
                            lora_srows.append(_srow)
                            lora_inputs += [_lf, _ld, _ls]
                        lora_add_btn = gr.Button("+ Add LoRA", size="sm")
                    lora_vis = gr.State([False, False, False])
                    # Per-model LoRA memory: switching DiT saves the outgoing
                    # model's slots and restores the incoming model's.
                    lora_mem = gr.State({})
                    prev_dit = gr.State(initial_dit)

                with gr.Accordion("Audio-to-audio (guide the whole clip)", open=False):
                    a2a_audio = gr.Audio(label="Guide audio — generation starts from its latents",
                                         type="filepath")
                    sigma_slider = gr.Slider(
                        label="init_noise_level (1.0 = prompt, ~0.92 = fusion, 0 = input)",
                        minimum=0.0, maximum=1.0, value=0.92, step=0.01)

                with gr.Accordion("Inpainting (regenerate a span of reference audio)", open=False):
                    inpaint_audio = gr.Audio(label="Reference audio — kept bit-exact outside the range",
                                             type="filepath")
                    with gr.Row():
                        inp_start = gr.Slider(label="Start (s)", minimum=0, maximum=120,
                                              value=0, step=0.5)
                        inp_end = gr.Slider(label="End (s)", minimum=0, maximum=120,
                                            value=0, step=0.5)

                with gr.Accordion("Output", open=False):
                    output_opts = gr.CheckboxGroup(
                        ["Auto-play", "Auto-download", "Infinite Radio", "Loop", "Hotswap"],
                        value=["Auto-play"], label="Playback Options")
                    opts_state = gr.State(["Auto-play"])
                    file_format = gr.Radio(
                        [FORMAT_MP3, FORMAT_WAV] if FFMPEG else [FORMAT_WAV],
                        value=FORMAT_MP3 if FFMPEG else FORMAT_WAV,
                        label="Format", visible=FFMPEG)

            with gr.Column(scale=2, elem_id="sa3-out"):
                generate_btn = gr.Button("Generate", variant="primary", size="lg",
                                         elem_id="sa3-generate")
                promote_btn = gr.Button("", elem_id="sa3-promote")   # CSS-hidden, DOM-present
                gr.Markdown("**Output**")
                output_player = gr.HTML()
                timing = gr.HTML()
                error_box = gr.HTML()
                queued_html = gr.HTML()
                history_html = gr.HTML()

        def _sync_dd_vis(dit_name):
            """Second batch after any group (re)mount: gradio 6 drops visibility
            updates sent to components inside a hidden container (the component
            isn't in the DOM), so dropdown show/hide must be applied AFTER the
            slot groups are mounted — chained via .then() on every event that
            reveals them."""
            v = len(_lora_dd_choices(dit_name)) > 1
            return [gr.update(visible=v) for _ in range(3)]

        dit_dd.change(on_dit_change,
                      inputs=[dit_dd, seconds, prev_dit, lora_mem,
                              lora_vis] + lora_inputs,
                      outputs=[decoder_dd, seconds] + lora_dds + lora_srows
                              + lora_rows
                              + [lora_add_btn, lora_vis, prev_dit, lora_mem]
                              + lora_files
                              + [lora_inputs[3 * i + 2] for i in range(3)]
                      ).then(_sync_dd_vis, inputs=[dit_dd], outputs=lora_dds
                      ).then(None, js=_JS_FIX_SLIDERS)

        def on_precision_change(prec):
            """Quantized-int8 precisions can't be LoRA-merged — hide the editor
            and show the note. fp32 / w16a32 re-enable it."""
            disabled = prec not in LORA_PRECISIONS
            return gr.update(visible=disabled), gr.update(visible=not disabled)
        precision_dd.change(on_precision_change, inputs=[precision_dd],
                            outputs=[lora_note, lora_editor]
                            ).then(_sync_dd_vis, inputs=[dit_dd], outputs=lora_dds
                            ).then(None, js=_JS_FIX_SLIDERS)

        def on_seconds_change(sec):
            return gr.update(maximum=sec), gr.update(maximum=sec)
        seconds.change(on_seconds_change, inputs=[seconds], outputs=[inp_start, inp_end])

        def on_opts_change(opts, prev):
            """Loop and Infinite Radio are mutually exclusive — the one just
            ticked wins, the other unticks."""
            opts = opts or []
            if "Loop" in opts and "Infinite Radio" in opts:
                added = set(opts) - set(prev or [])
                drop = "Infinite Radio" if "Loop" in added else "Loop"
                opts = [o for o in opts if o != drop]
                return gr.update(value=opts), opts
            return gr.update(), opts
        output_opts.change(on_opts_change, inputs=[output_opts, opts_state],
                           outputs=[output_opts, opts_state])

        def lora_add(vis, dit_name, f1, d1, f2, d2, f3, d3):
            """Reveal the first hidden LoRA slot; hide the button at 3/3.
            Slider-row and dropdown visibility are recomputed from slot content
            and the current model's library on every add, so a re-opened slot
            can never inherit a stale visible state."""
            vis = list(vis)
            for i, v in enumerate(vis):
                if not v:
                    vis[i] = True
                    break
            srows = [gr.update(visible=bool(f or (d and d != _DD_NONE)))
                     for f, d in ((f1, d1), (f2, d2), (f3, d3))]
            return ([gr.update(visible=v) for v in vis] + srows
                    + [gr.update(visible=not all(vis)), vis])

        lora_add_btn.click(lora_add,
                           inputs=[lora_vis, dit_dd]
                           + [c for i in range(3)
                              for c in lora_inputs[3 * i: 3 * i + 2]],
                           outputs=lora_rows + lora_srows
                           + [lora_add_btn, lora_vis]
                           ).then(_sync_dd_vis, inputs=[dit_dd], outputs=lora_dds
                           ).then(None, js=_JS_FIX_SLIDERS)

        def _lora_remove(idx):
            def _rm(vis):
                vis = list(vis)
                vis[idx] = False
                return ([gr.update(visible=v) for v in vis]
                        + [gr.update(visible=True), vis,
                           # reset the slot: file, dropdown, strength (srow hidden)
                           None, gr.update(value=_DD_NONE), 1.0,
                           gr.update(visible=False)])
            _rm.__name__ = f"lora_remove_{idx + 1}"
            return _rm
        for _idx, _btn in enumerate(lora_rm_btns):
            _btn.click(_lora_remove(_idx), inputs=[lora_vis],
                       outputs=lora_rows + [lora_add_btn, lora_vis]
                       + lora_inputs[3 * _idx: 3 * _idx + 3]
                       + [lora_srows[_idx]])

        # A slot's strength slider appears once an adapter is loaded — via file
        # drop OR dropdown pick — and the two sources are exclusive (loading one
        # resets the other). All three events are user-driven (upload/clear/input),
        # so the programmatic resets here can't loop.
        def _lora_wire_slot(idx):
            lf, ld, srow = lora_files[idx], lora_dds[idx], lora_srows[idx]

            def _up(f):
                return gr.update(visible=bool(f)), gr.update(value=_DD_NONE)
            _up.__name__ = f"lora_file_{idx + 1}"
            lf.upload(_up, inputs=[lf], outputs=[srow, ld]
                      ).then(None, js=_JS_FIX_SLIDERS)

            def _cl(dd):
                return gr.update(visible=bool(dd and dd != _DD_NONE))
            _cl.__name__ = f"lora_file_clear_{idx + 1}"
            lf.clear(_cl, inputs=[ld], outputs=[srow])

            def _pick(dd, f):
                loaded = bool(dd and dd != _DD_NONE)
                return (gr.update(visible=loaded or bool(f)),
                        gr.update(value=None) if loaded else gr.update())
            _pick.__name__ = f"lora_pick_{idx + 1}"
            ld.input(_pick, inputs=[ld, lf], outputs=[srow, lf]
                     ).then(None, js=_JS_FIX_SLIDERS)
        for _idx in range(3):
            _lora_wire_slot(_idx)

        ctrl_inputs = [dit_dd, precision_dd, decoder_dd, prompt, negative_prompt,
                       seconds, steps, seed, cfg, apg, sigma_global,
                       sigma_slider, a2a_audio, inpaint_audio,
                       inp_start, inp_end, output_opts, file_format] + lora_inputs
        main_outputs = [output_player, timing, error_box, queued_html, history_html, st]

        # NB: _present returns (player, timing, err, history, queued, state) —
        # map onto components in that order.
        def _reorder(fn):
            def wrapped(*args):
                player, tim, err_, hist, queued, state = fn(*args)
                return player, tim, err_, queued, hist, state
            wrapped.__name__ = fn.__name__
            return wrapped

        generate_btn.click(_reorder(generate), inputs=ctrl_inputs + [st],
                           outputs=main_outputs
                           ).then(pregen, inputs=ctrl_inputs + [st],
                                  outputs=[queued_html, st])
        promote_btn.click(_reorder(promote), inputs=ctrl_inputs + [st],
                          outputs=main_outputs
                          ).then(pregen, inputs=ctrl_inputs + [st],
                                 outputs=[queued_html, st])

        gr.Markdown(
            "<p style='color:#888; font-size:0.85em'>"
            "WAVs saved under <code>output/gradio/</code>. "
            "Models run on CPU (XNNPACK). Precision applies to the DiT + codec "
            "(+ encoder for a2a/inpaint); T5Gemma is fp16.</p>"
        )

    # gradio 6 moved `css` from the Blocks constructor to launch().
    demo.queue(max_size=16).launch(share=share, server_name="0.0.0.0",
                                   server_port=server_port, css=_css,
                                   allowed_paths=[str(OUTPUT_DIR)],
                                   prevent_thread_lock=False, show_error=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", choices=list(DIT_REL.keys()), default="sm-music",
                    help="Initial DiT bundle (switchable at runtime)")
    ap.add_argument("--decoder", choices=list(DEC_REL.keys()), default=None,
                    help="Initial decoder. Default: pairs with --dit")
    ap.add_argument("--precision", choices=list(PRECISIONS), default="fp32",
                    help="Initial precision (switchable at runtime): fp32 (default, "
                         "CPU fast+accurate) | w16a32 (fp16, ≈lossless, half size) | "
                         "w8a32 / w8a8-dyn (GPTQ int8, ¼ size)")
    ap.add_argument("--default-seconds", type=float, default=30.0,
                    help="Length to pre-warm the initial DiT at")
    ap.add_argument("--default-steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8, help="XNNPACK CPU threads")
    ap.add_argument("--port", type=int, default=None,
                    help="Server port (default: gradio picks 7860+, auto-incrementing)")
    ap.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                    help="Create a public gradio.live URL (default on)")
    args = ap.parse_args()

    if args.decoder is None:
        args.decoder = DEFAULT_DECODERS[args.dit]

    global _THREADS
    _THREADS = args.threads

    print(f"\n━━━ SA3 TFLite — gradio ━━━")
    print(f"  initial dit:      {args.dit}")
    print(f"  initial decoder:  {args.decoder}")
    print(f"  initial precision:{args.precision}")
    print(f"  threads:          {args.threads}")
    print(f"  models:           {', '.join(DIT_REL.keys())}  (runtime-switchable)")
    build_ui(args.dit, args.decoder, args.precision, share=args.share,
             default_seconds=args.default_seconds, default_steps=args.default_steps,
             server_port=args.port)


if __name__ == "__main__":
    main()
