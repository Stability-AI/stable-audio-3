"""SA3 MLX — gradio web UI (Apple Silicon).

The MLX sibling of optimized/tensorRT/scripts/sa3_gradio.py, with every
generation mode wired (the TRT one only exposes text-to-audio):
  - Model picker: sm-music / sm-sfx / medium (hot-swap; models cache in unified
    memory, LRU-evicted — first switch loads weights, subsequent instant)
  - CFG 0-10 next to seconds/steps (0 = negative prompt takes over, 0.5 = halfway
    between prompts, 1 = off, >1 = extrapolate) + negative prompt/APG under Advanced
  - Audio-to-audio: guide audio + σmax slider (whole clip starts from its latents)
  - Inpainting: separate reference audio + start/end range sliders (kept bit-exact
    outside the range). Combinable with a2a: the regenerated span then starts
    from the guide audio instead of noise.
  - Spectrogram display: 3-band tinted stereo mel spectrogram (numpy port —
    no torch), rendered inline alongside the audio

Launch:
    ./sa3-gradio                  # share=True by default, sm-music + same-s
    ./sa3-gradio --dit medium
    ./sa3-gradio --no-share       # local-only
"""
from __future__ import annotations
import argparse
import base64
import math
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

import mlx.core as mx  # noqa: E402

from sa3_mlx import (  # noqa: E402
    DIT_CHOICES, DECODER_CHOICES, ENCODER_CHOICES, T5GEMMA_NPZ_REL,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
    load_dit, load_decoder, load_encoder, read_wav, patch_audio, _free_to_pool,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    apply_prompt_padding, build_pingpong_schedule, sample_flow_pingpong,
    patched_decode, load_conditioner_from_npz,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from weights import ensure_local  # noqa: E402
from spec import render_spectrogram_png  # noqa: E402

OUTPUT_DIR = REPO / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_SIGMA = 0.01
# MP3 (V0) saving needs ffmpeg; without it we save WAV and hide the choice.
FFMPEG = shutil.which("ffmpeg") is not None
FORMAT_MP3, FORMAT_WAV = "Save to MP3 (V0)", "Save to WAV"
DEFAULT_DECODERS = {name: cfg["default_decoder"] for name, cfg in DIT_CHOICES.items()}
# Trained max clip length per model (repo README model table).
MAX_SECONDS = {"sm-music": 120, "sm-sfx": 120, "medium": 380}

# MLX ≥0.31 registers GPU streams PER THREAD (ThreadLocalStream) while gradio
# runs each handler on a rotating anyio worker thread — cross-thread MLX use
# then dies with "There is no Stream(gpu, 0) in current thread". The categorical
# fix: ALL MLX work (pre-warm + every generation) runs on one dedicated owner
# thread; handlers submit to it and wait. This also serializes generations.
import concurrent.futures as _cf
_MLX_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def mlx_call(fn, *args, **kwargs):
    """Run fn on the MLX owner thread and return its result (re-raises errors)."""
    return _MLX_EXECUTOR.submit(fn, *args, **kwargs).result()


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


def verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed) -> str:
    """prompt[.neg-…].cfg{scale}[.smx{σ}].{seed} — matches the main repo's
    gradio 'verbose' file naming."""
    base = condense_prompt(prompt)
    if negative_prompt and negative_prompt.strip():
        base += ".neg-" + condense_prompt(negative_prompt.strip())
    if cfg != 1.0:
        base += f".cfg{cfg:g}"
    if sigma_max != 1.0:
        base += f".smx{sigma_max:g}"
    return f"{base}.{seed}"


def _save_wav(pcm_int16, out_path):
    """pcm_int16: (T, 2) int16 interleaved."""
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_int16.tobytes())


# ── Model caches (unified memory; ~0.9-2.8 GB per DiT, 0.2-1.7 GB per codec) ──
# The MLX DiT bakes RoPE/mask lengths at load, so the DiT cache key includes
# T_lat. Everything else is length-independent and cached by name.
_t5: T5Gemma | None = None
_cond_cache: dict[str, tuple] = {}                       # dit -> (padding_emb, secs_embedder)
_dit_cache: dict[tuple[str, int], object] = {}           # (dit, T_lat) -> model
_dit_lru: list[tuple[str, int]] = []
_DIT_CACHE_MAX = 2
_dec_cache: dict[str, tuple] = {}                        # decoder -> (model, chunk_fn, chunk_cfg)
_enc_cache: dict[str, tuple] = {}                        # decoder -> (model, pad_modulo)


def get_t5() -> T5Gemma:
    global _t5
    if _t5 is None:
        _t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    return _t5


def get_conditioner(dit_name: str):
    if dit_name not in _cond_cache:
        _cond_cache[dit_name] = load_conditioner_from_npz(
            str(ensure_local(DIT_CHOICES[dit_name]["ckpt"])), prefix="cond.")
    return _cond_cache[dit_name]


def get_dit(dit_name: str, T_lat: int, dtype):
    key = (dit_name, T_lat)
    if key in _dit_cache:
        if key in _dit_lru:
            _dit_lru.remove(key)
        _dit_lru.append(key)
        return _dit_cache[key], 0.0
    while len(_dit_cache) >= _DIT_CACHE_MAX:
        oldest = _dit_lru.pop(0)
        print(f"  ← LRU-evicting DiT {oldest}")
        _dit_cache.pop(oldest, None)
        _free_to_pool()
    t0 = time.time()
    model, _ = load_dit(dit_name, T_lat=T_lat, dtype=dtype)
    load_ms = (time.time() - t0) * 1000
    _dit_cache[key] = model
    _dit_lru.append(key)
    return model, load_ms


def get_decoder(decoder_name: str):
    if decoder_name not in _dec_cache:
        _dec_cache[decoder_name] = load_decoder(decoder_name, mx.float32)
    return _dec_cache[decoder_name]


def get_encoder(decoder_name: str):
    if decoder_name not in _enc_cache:
        _enc_cache[decoder_name] = load_encoder(decoder_name, mx.float32)
    return _enc_cache[decoder_name]


# ── One full generation (mirrors sa3_mlx.py main(), all modes) ─────────────
def run_generation(dit_name: str, decoder_name: str, prompt: str,
                   negative_prompt: str, seconds: float, steps: int,
                   seed: int, cfg: float, apg: float, sigma_max: float,
                   a2a_audio_path: str | None = None,
                   inpaint_audio_path: str | None = None,
                   inpaint_range_sec=None):
    """Returns (audio_np (2,T) float32, timings dict).

    a2a and inpainting are independent and combinable:
      - a2a_audio_path: the whole generation STARTS from this audio's latents
        (noise = lat*(1-σmax) + noise*σmax)
      - inpaint_audio_path + inpaint_range_sec: kept bit-exact outside the range
        (local_add_cond context + per-step paste-back)
      - both: the inpainted span regenerates FROM the a2a guide instead of pure noise
    """
    t = {}
    mx.set_default_device(mx.gpu)   # belt-and-braces; normally on the MLX owner thread
    dtype = mx.float16   # MLX canonical DiT dtype
    T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))

    # 1. T5Gemma
    t0 = time.time()
    enc = get_t5()
    embeds, mask = enc.encode([prompt], max_len=256)
    mx.eval(embeds, mask)
    t["t5_ms"] = (time.time() - t0) * 1000

    # 2. Conditioning
    t0 = time.time()
    padding_emb, secs_embedder = get_conditioner(dit_name)
    embeds = embeds.astype(dtype)
    embeds_padded = apply_prompt_padding(embeds, mask, padding_emb.astype(dtype))
    seconds_embed = secs_embedder(seconds).astype(dtype)
    cross_attn = mx.concatenate([embeds_padded, seconds_embed], axis=1)
    global_cond = seconds_embed[:, 0, :]
    null_cross_attn = None
    if cfg != 1.0:
        if negative_prompt and negative_prompt.strip():
            neg_embeds, neg_mask = enc.encode([negative_prompt.strip()], max_len=256)
            mx.eval(neg_embeds, neg_mask)
            neg_padded = apply_prompt_padding(neg_embeds.astype(dtype), neg_mask,
                                              padding_emb.astype(dtype))
            null_cross_attn = mx.concatenate([neg_padded, seconds_embed], axis=1)
        else:
            null_cross_attn = mx.zeros_like(cross_attn)
        mx.eval(null_cross_attn)
    mx.eval(cross_attn, global_cond)
    t["cond_ms"] = (time.time() - t0) * 1000

    # 3a. encode the provided audio inputs → latents (each is optional)
    def encode_audio(path):
        enc_model, pad_mod = get_encoder(decoder_name)
        enc_T_lat = T_lat
        if (T_lat * 16) % pad_mod != 0:
            enc_T_lat = math.ceil((T_lat * 16) / pad_mod) * pad_mod // 16
        target_samples = enc_T_lat * SAMPLES_PER_LATENT
        audio_np = read_audio_any(path)
        if audio_np.shape[-1] >= target_samples:
            audio_np = audio_np[:, :target_samples]
        else:
            audio_np = np.pad(audio_np, ((0, 0), (0, target_samples - audio_np.shape[-1])))
        patches_np = patch_audio(audio_np[None, ...], patch_size=256)
        lat = enc_model(mx.array(patches_np))[..., :T_lat]
        mx.eval(lat)
        return lat.astype(dtype)

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

    # 3b. DiT + pingpong sample
    dit_model, t["dit_load_ms"] = get_dit(dit_name, T_lat, dtype)
    sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max, use_logsnr_shift=True)

    key = mx.random.key(seed)
    pure_noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=key)
    # a2a start-point mix is independent of inpainting: when both are set, the
    # inpainted span regenerates FROM the guide audio instead of pure noise
    # (the kept span is pasted back from ctx_latents every step regardless).
    if a2a_latents is not None:
        noise = a2a_latents * (1.0 - sigma_max) + pure_noise * sigma_max
    else:
        noise = pure_noise
    mx.eval(noise)

    local_add_cond = None
    paste_back = None
    if ctx_latents is not None:
        s0 = max(0, int(round(inpaint_range_sec[0] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(T_lat, int(round(inpaint_range_sec[1] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        mask_np = np.ones((1, 1, T_lat), dtype=np.float32)
        mask_np[:, :, s0:s1] = 0.0
        keep = mx.array(mask_np)
        masked_input = ctx_latents.astype(mx.float32) * keep
        local_add_cond = mx.concatenate([keep, masked_input], axis=1).transpose(0, 2, 1).astype(dtype)
        paste_back = (ctx_latents, keep)

    def model_fn(x, tt):
        if cfg == 1.0:
            return dit_model(x, tt, cross_attn, global_cond, local_add_cond=local_add_cond)
        x2 = mx.concatenate([x, x], axis=0)
        t2 = mx.concatenate([tt, tt], axis=0)
        cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
        global2 = mx.concatenate([global_cond, global_cond], axis=0)
        lac2 = None if local_add_cond is None else mx.concatenate([local_add_cond, local_add_cond], axis=0)
        v_batched = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
        cond_v, uncond_v = mx.split(v_batched, 2, axis=0)
        sigma = tt.reshape(-1, 1, 1).astype(mx.float32)
        cond_d = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
        uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
        diff = cond_d - uncond_d
        # cfg < 1 is the INTERPOLATION regime: cfg_d = lerp(uncond_d, cond_d, cfg),
        # so 0 = pure negative/uncond branch, 0.5 = halfway between both prompts.
        # APG's orthogonal projection would bend that line — it only applies to the
        # extrapolation regime (cfg > 1) it was designed for.
        if apg <= 0.0 or cfg < 1.0:
            cfg_diff = diff
        else:
            norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
            unit = cond_d / mx.maximum(norm, 1e-8)
            parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
            diff_orth = diff - parallel
            cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
        cfg_d = cond_d + (cfg - 1.0) * cfg_diff
        cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
        return cfg_v.astype(x.dtype)

    t0 = time.time()
    latents = sample_flow_pingpong(model_fn, noise, sigmas, seed=seed + 1,
                                   paste_back=paste_back)
    mx.eval(latents)
    t["sample_ms"] = (time.time() - t0) * 1000

    # 4. Decode (fp32 codec; parity-aware chunk dispatch from sa3_mlx.py)
    t0 = time.time()
    decoder, chunk_fn, (chunk, ovl) = get_decoder(decoder_name)
    latents_fp32 = latents.astype(mx.float32)
    kernel = chunk + 2 * ovl
    if T_lat > kernel:
        patches = chunk_fn(decoder, latents_fp32, chunk, ovl)
    elif T_lat % 2 == 0:
        patches = decoder(latents_fp32)
    elif T_lat > 6:
        patches = chunk_fn(decoder, latents_fp32, 2, 2)
    else:
        latents_even = mx.concatenate([latents_fp32, latents_fp32[..., -1:]], axis=-1)
        patches = decoder(latents_even)[..., : T_lat * 16]
    mx.eval(patches)
    t["decode_ms"] = (time.time() - t0) * 1000

    # 5. Unpatch + trim
    audio = patched_decode(patches, patch_size=256, channels=2)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32))[0]
    requested = int(round(seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > requested:
        audio_np = audio_np[..., :requested]

    t["T_lat"] = T_lat
    t["samples"] = audio_np.shape[-1]
    t["inference_ms"] = sum(t.get(k, 0) for k in
                            ("t5_ms", "cond_ms", "enc_ms", "dit_load_ms", "sample_ms", "decode_ms"))
    t["realtime"] = (audio_np.shape[-1] / SAMPLE_RATE) / max(t["inference_ms"] / 1000, 1e-9)
    return audio_np, t


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui(initial_dit: str, initial_decoder: str, *, share: bool,
             default_seconds: float, default_steps: int):
    import gradio as gr
    import random as _random

    # Pre-warm the initial pipeline so the first click is fast — ON the MLX
    # owner thread, so all stream/model state lives where generations run.
    warm_T = max(1, math.ceil(default_seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    print(f"  pre-warming {initial_dit}+{initial_decoder} (T_lat={warm_T})...")

    def _warm():
        get_t5(); get_conditioner(initial_dit)
        get_dit(initial_dit, warm_T, mx.float16); get_decoder(initial_decoder)
    mlx_call(_warm)

    def on_dit_change(dit_name, cur_seconds):
        max_s = MAX_SECONDS.get(dit_name, 120)
        return (gr.update(value=DEFAULT_DECODERS.get(dit_name, "same-s")),
                gr.update(maximum=max_s, value=min(cur_seconds, max_s)))

    def generate(dit_name, decoder_name, prompt, negative_prompt,
                 seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                 a2a_audio, inpaint_audio, inp_start, inp_end,
                 output_opts, file_format):
        err = lambda m: ("", "", f"<span style='color:#f88'>{m}</span>")
        prompt = (prompt or "").strip()
        # Permissive by design: a generation should succeed with whatever IS set.
        # Half-configured features are ignored with a visible note, never an error.
        notes = []
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

        mode = ("a2a+inpaint" if (a2a_audio and inpaint_range) else
                "inpaint" if inpaint_range else
                "audio-to-audio" if a2a_audio else "text-to-audio")
        try:
            audio_np, t = mlx_call(
                run_generation,
                dit_name, decoder_name, prompt, negative_prompt or "",
                float(seconds), int(steps), seed, float(cfg), float(apg),
                float(sigma_max), a2a_audio or None, inpaint_audio or None,
                inpaint_range)
        except Exception as e:
            return err(f"error: {type(e).__name__}: {e}")
        if not np.isfinite(audio_np).all():
            return err("error: model produced non-finite audio (try a higher σmax or different seed)")

        pcm = (np.clip(audio_np, -1, 1) * 32767.0).astype(np.int16).T   # (T, 2)
        basename = verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed)
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
        b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
        # Audio + spectrogram in ONE block so inline handlers can couple them:
        # the audio's timeupdate drives the white playhead; clicking the
        # spectrogram seeks (gradio HTML runs attributes, not <script> tags).
        opts = output_opts or []
        autoplay = "autoplay " if "Auto-play" in opts else ""
        extra_attrs = ""
        if "Auto-download" in opts:
            js_name = out_path.name.replace("\\", "").replace("'", "\\'")
            extra_attrs += (' oncanplay="if(!this.dataset.dld){this.dataset.dld=1;'
                            "var l=document.createElement('a');l.href=this.src;"
                            f"l.download='{js_name}';l.click();}}\"")
        if "Infinite Radio" in opts:
            extra_attrs += (' onended="var b=document.getElementById(\'sa3-generate\');'
                            "if(b){(b.tagName==='BUTTON'?b:b.querySelector('button')||b).click();}\"")
            if seed_text and seed_text.strip() and seed_text.strip() != "-1":
                notes.append("Infinite Radio with a fixed seed repeats the same clip — "
                             "clear the seed for endless variety")
        audio_el = (
            f'<audio controls {autoplay}style="width:100%" '
            f'ontimeupdate="var p=this.parentNode.querySelector(\'.ph\');'
            f'if(p&&this.duration)p.style.left=(this.currentTime/this.duration*100)+\'%\';"'
            f'{extra_attrs} '
            f'src="data:{mime};base64,{b64}"></audio>'
            f'<div style="font-size:0.85em; margin:4px 0; color:#888">'
            f'{out_path.stat().st_size/1e6:.1f} MB · {mode} · saved as <code>{out_path.name}</code></div>'
        )
        try:
            spec_png = render_spectrogram_png(pcm, sample_rate=SAMPLE_RATE,
                                              width=1200, height=240)
            spec_b64 = base64.b64encode(spec_png).decode("ascii")
            spec_el = (
                f'<div style="position:relative; cursor:pointer" '
                f'onclick="var a=this.parentNode.querySelector(\'audio\');'
                f'var r=this.getBoundingClientRect();'
                f'if(a&&a.duration){{a.currentTime=(event.clientX-r.left)/r.width*a.duration;a.play();}}">'
                f'<img src="data:image/png;base64,{spec_b64}" '
                f'style="width:100%; display:block; image-rendering:pixelated; border:1px solid #333" '
                f'alt="spectrogram"/>'
                f'<div class="ph" style="position:absolute; top:0; bottom:0; left:0%; width:2px; '
                f'background:#fff; pointer-events:none; box-shadow:0 0 4px rgba(0,0,0,.8)"></div>'
                f'</div>'
                f'<div style="font-size:0.75em; color:#666; margin-top:2px">'
                f'3-band tinted stereo mel · red=bass / green=mid / blue=high · L top, R bottom · '
                f'click to seek</div>'
            )
        except Exception as e:
            spec_el = f"<span style='color:#fa3'>spectrogram failed: {type(e).__name__}: {e}</span>"
        player_html = f"<div>{audio_el}{spec_el}</div>"

        load_note = (f"DiT-load {t['dit_load_ms']:.0f} ms ·&nbsp; "
                     if t.get("dit_load_ms", 0) > 100 else "")
        enc_note = f"encode {t['enc_ms']:.0f} ms ·&nbsp; " if t.get("enc_ms") else ""
        cfg_note = f"cfg {cfg} (apg {apg}) ·&nbsp; " if cfg != 1.0 else ""
        timing_html = (
            f"{load_note}{enc_note}{cfg_note}"
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
        return player_html, timing_html, ""

    with gr.Blocks(title="SA3 MLX") as demo:
        gr.Markdown(
            "# SA3 MLX — Apple Silicon\n"
            "All modes wired: text-to-audio, CFG + negative prompt, audio-to-audio "
            "(upload + σmax), inpainting (upload + range). Models cache in unified "
            "memory — first use of a model loads weights, subsequent runs are instant."
        )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    dit_dd = gr.Dropdown(label="DiT model", choices=list(DIT_CHOICES.keys()),
                                         value=initial_dit, scale=1)
                    decoder_dd = gr.Dropdown(label="Decoder (codec)",
                                             choices=list(DECODER_CHOICES.keys()),
                                             value=initial_decoder, scale=1)
                prompt = gr.Textbox(label="Prompt", lines=2,
                                    placeholder="e.g. 'Impending tribal, epic orchestral buildup'")
                with gr.Row():
                    seconds = gr.Slider(label="Seconds", minimum=1,
                                        maximum=MAX_SECONDS.get(initial_dit, 120),
                                        value=default_seconds, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16,
                                      value=default_steps, step=1)
                    cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0,
                                    value=1.0, step=0.1)
                seed = gr.Textbox(label="Seed (optional, blank = random)",
                                  max_lines=1, value="")
                generate_btn = gr.Button("Generate", variant="primary", size="lg",
                                         elem_id="sa3-generate")

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        apg = gr.Slider(label="APG (only applies when CFG > 1)",
                                        minimum=0.0, maximum=1.0, value=1.0, step=0.05)
                        sigma_global = gr.Slider(label="sigma_max",
                                                 minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=1)

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
                        ["Auto-play", "Auto-download", "Infinite Radio"],
                        value=["Auto-play"], label="Options")
                    file_format = gr.Radio(
                        [FORMAT_MP3, FORMAT_WAV] if FFMPEG else [FORMAT_WAV],
                        value=FORMAT_MP3 if FFMPEG else FORMAT_WAV,
                        label="Format", visible=FFMPEG)

            with gr.Column(scale=2):
                gr.Markdown("**Output**")
                output_player = gr.HTML()
                timing = gr.HTML()
                error_box = gr.HTML()

        dit_dd.change(on_dit_change, inputs=[dit_dd, seconds],
                      outputs=[decoder_dd, seconds])

        def on_seconds_change(sec):
            return gr.update(maximum=sec), gr.update(maximum=sec)
        seconds.change(on_seconds_change, inputs=[seconds], outputs=[inp_start, inp_end])

        generate_btn.click(generate,
                           inputs=[dit_dd, decoder_dd, prompt, negative_prompt,
                                   seconds, steps, seed, cfg, apg, sigma_global,
                                   sigma_slider, a2a_audio, inpaint_audio,
                                   inp_start, inp_end, output_opts, file_format],
                           outputs=[output_player, timing, error_box])

        gr.Markdown(
            "<p style='color:#888; font-size:0.85em'>"
            "WAVs saved under <code>output/gradio/</code>. "
            "DiT runs fp16 (MLX canonical), codecs fp32.</p>"
        )

    demo.queue(max_size=16).launch(share=share, server_name="0.0.0.0",
                                   prevent_thread_lock=False, show_error=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", choices=list(DIT_CHOICES.keys()), default="sm-music",
                    help="Initial DiT bundle (switchable at runtime)")
    ap.add_argument("--decoder", choices=list(DECODER_CHOICES.keys()), default=None,
                    help="Initial decoder. Default: pairs with --dit")
    ap.add_argument("--default-seconds", type=float, default=30.0,
                    help="Length to pre-warm the initial DiT at")
    ap.add_argument("--default-steps", type=int, default=8)
    ap.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                    help="Create a public gradio.live URL (default on)")
    args = ap.parse_args()

    if args.decoder is None:
        args.decoder = DEFAULT_DECODERS[args.dit]

    print(f"\n━━━ SA3 MLX — gradio ━━━")
    print(f"  initial dit:     {args.dit}")
    print(f"  initial decoder: {args.decoder}")
    print(f"  models:          {', '.join(DIT_CHOICES.keys())}  (runtime-switchable)")
    build_ui(args.dit, args.decoder, share=args.share,
             default_seconds=args.default_seconds, default_steps=args.default_steps)


if __name__ == "__main__":
    main()
