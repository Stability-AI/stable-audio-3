"""SA3 MLX — gradio web UI (Apple Silicon).

The MLX sibling of optimized/tensorRT/scripts/sa3_gradio.py, with every
generation mode wired (the TRT one only exposes text-to-audio):
  - Model picker: sm-music / sm-sfx / medium (hot-swap; models cache in unified
    memory, LRU-evicted — first switch loads weights, subsequent instant)
  - CFG + negative prompt (batched CFG with APG, same math as sa3_mlx.py)
  - Audio-to-audio: upload init audio + σmax slider
  - Inpainting: init audio + "START,END" seconds range (paste-back guaranteed
    bit-exact outside the range)
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
import sys
import time
import uuid
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
KEEP_RECENT_N = 20
MIN_SIGMA = 0.01
DEFAULT_DECODERS = {name: cfg["default_decoder"] for name, cfg in DIT_CHOICES.items()}


def _prune_old_outputs():
    wavs = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in wavs[KEEP_RECENT_N:]:
        try:
            old.unlink()
        except OSError:
            pass


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
                   init_audio_path: str | None, inpaint_range_sec):
    """Returns (audio_np (2,T) float32, timings dict)."""
    t = {}
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

    # 3a. (a2a / inpaint) encode init audio → latents
    init_latents = None
    if init_audio_path:
        t0 = time.time()
        enc_model, pad_mod = get_encoder(decoder_name)
        enc_T_lat = T_lat
        if (T_lat * 16) % pad_mod != 0:
            enc_T_lat = math.ceil((T_lat * 16) / pad_mod) * pad_mod // 16
        target_samples = enc_T_lat * SAMPLES_PER_LATENT
        audio_np = read_wav(init_audio_path)
        if audio_np.shape[-1] >= target_samples:
            audio_np = audio_np[:, :target_samples]
        else:
            audio_np = np.pad(audio_np, ((0, 0), (0, target_samples - audio_np.shape[-1])))
        patches_np = patch_audio(audio_np[None, ...], patch_size=256)
        init_latents = enc_model(mx.array(patches_np))[..., :T_lat]
        mx.eval(init_latents)
        init_latents = init_latents.astype(dtype)
        t["enc_ms"] = (time.time() - t0) * 1000

    # 3b. DiT + pingpong sample
    dit_model, t["dit_load_ms"] = get_dit(dit_name, T_lat, dtype)
    sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max, use_logsnr_shift=True)

    key = mx.random.key(seed)
    pure_noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=key)
    inpaint_lat = None
    if inpaint_range_sec is not None:
        s0 = max(0, int(round(inpaint_range_sec[0] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(T_lat, int(round(inpaint_range_sec[1] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_lat = (s0, s1)
    if init_latents is not None and inpaint_lat is None:
        noise = init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
    else:
        noise = pure_noise
    mx.eval(noise)

    local_add_cond = None
    paste_back = None
    if inpaint_lat is not None:
        s0, s1 = inpaint_lat
        mask_np = np.ones((1, 1, T_lat), dtype=np.float32)
        mask_np[:, :, s0:s1] = 0.0
        keep = mx.array(mask_np)
        masked_input = init_latents.astype(mx.float32) * keep
        local_add_cond = mx.concatenate([keep, masked_input], axis=1).transpose(0, 2, 1).astype(dtype)
        paste_back = (init_latents, keep)

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
        if apg <= 0.0:
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

    # Pre-warm the initial pipeline so the first click is fast.
    warm_T = max(1, math.ceil(default_seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    print(f"  pre-warming {initial_dit}+{initial_decoder} (T_lat={warm_T})...")
    get_t5(); get_conditioner(initial_dit)
    get_dit(initial_dit, warm_T, mx.float16); get_decoder(initial_decoder)

    def on_dit_change(dit_name):
        return gr.update(value=DEFAULT_DECODERS.get(dit_name, "same-s"))

    def generate(dit_name, decoder_name, prompt, negative_prompt,
                 seconds, steps, seed_text, cfg, apg, sigma_max,
                 init_audio, inpaint_text):
        err = lambda m: ("", "", "", f"<span style='color:#f88'>{m}</span>")
        prompt = (prompt or "").strip()
        try:
            seed = int(seed_text.strip()) if seed_text and seed_text.strip() else _random.randint(0, 2**31 - 1)
        except ValueError:
            return err("error: seed must be an integer")
        if sigma_max < MIN_SIGMA:
            return err(f"error: σmax must be ≥ {MIN_SIGMA} (rf_denoiser is undefined at t≈0)")
        inpaint_range = None
        if inpaint_text and inpaint_text.strip():
            if not init_audio:
                return err("error: inpaint range requires init audio")
            try:
                s_str, e_str = inpaint_text.split(",")
                inpaint_range = (float(s_str), float(e_str))
            except ValueError:
                return err(f"error: inpaint range must be 'START,END' seconds, got {inpaint_text!r}")
            if not (0 <= inpaint_range[0] < inpaint_range[1] <= seconds):
                return err(f"error: need 0 ≤ start < end ≤ {seconds}s")

        mode = ("inpaint" if inpaint_range else
                "audio-to-audio" if init_audio else "text-to-audio")
        try:
            audio_np, t = run_generation(
                dit_name, decoder_name, prompt, negative_prompt or "",
                float(seconds), int(steps), seed, float(cfg), float(apg),
                float(sigma_max), init_audio or None, inpaint_range)
        except Exception as e:
            return err(f"error: {type(e).__name__}: {e}")
        if not np.isfinite(audio_np).all():
            return err("error: model produced non-finite audio (try a higher σmax or different seed)")

        pcm = (np.clip(audio_np, -1, 1) * 32767.0).astype(np.int16).T   # (T, 2)
        out_path = OUTPUT_DIR / f"sa3-{uuid.uuid4().hex[:10]}.wav"
        _save_wav(pcm, out_path)
        _prune_old_outputs()
        b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
        audio_html = (
            f'<audio controls autoplay style="width:100%" '
            f'src="data:audio/wav;base64,{b64}"></audio>'
            f'<div style="font-size:0.85em; margin-top:4px; color:#888">'
            f'{out_path.stat().st_size/1e6:.1f} MB · {mode} · right-click to download</div>'
        )
        try:
            spec_png = render_spectrogram_png(pcm, sample_rate=SAMPLE_RATE,
                                              width=1200, height=240)
            spec_b64 = base64.b64encode(spec_png).decode("ascii")
            spec_html = (
                f'<img src="data:image/png;base64,{spec_b64}" '
                f'style="width:100%; image-rendering:pixelated; border:1px solid #333" '
                f'alt="spectrogram"/>'
                f'<div style="font-size:0.75em; color:#666; margin-top:2px">'
                f'3-band tinted stereo mel · red=bass / green=mid / blue=high · L top, R bottom</div>'
            )
        except Exception as e:
            spec_html = f"<span style='color:#fa3'>spectrogram failed: {type(e).__name__}: {e}</span>"

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
            f"<b>T_lat</b>: {t['T_lat']} ·&nbsp; <b>samples</b>: {t['samples']}"
        )
        return audio_html, spec_html, timing_html, ""

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
                    seconds = gr.Slider(label="Seconds", minimum=1, maximum=120,
                                        value=default_seconds, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16,
                                      value=default_steps, step=1)
                seed = gr.Textbox(label="Seed (optional, blank = random)",
                                  max_lines=1, value="")
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Accordion("CFG / negative prompt", open=False):
                    cfg = gr.Slider(label="CFG scale (1.0 = off)", minimum=1.0,
                                    maximum=10.0, value=1.0, step=0.1)
                    apg = gr.Slider(label="APG (1 = full orthogonal projection, 0 = vanilla CFG)",
                                    minimum=0.0, maximum=1.0, value=1.0, step=0.05)
                    negative_prompt = gr.Textbox(label="Negative prompt (needs CFG > 1)", lines=1)

                with gr.Accordion("Audio-to-audio / inpainting", open=False):
                    init_audio = gr.Audio(label="Init audio (enables a2a; add a range below for inpainting)",
                                          type="filepath")
                    sigma_slider = gr.Slider(label="Init noise level σmax (a2a: 0.4–0.8 typical; 1.0 = ignore init)",
                                             minimum=0.05, maximum=1.2, value=1.0, step=0.05)
                    inpaint_text = gr.Textbox(
                        label="Inpaint range 'START,END' seconds (regenerates just that span; blank = a2a)",
                        max_lines=1, value="")
                    gr.Markdown(
                        "<span style='color:#888; font-size:0.85em'>a2a wants clips ≥ ~20 s "
                        "(shorter inputs give repetitive latents). Inpainting reads best with "
                        "CFG ≥ 5 and a contrasting prompt.</span>")

            with gr.Column(scale=2):
                gr.Markdown("**Audio**")
                output_audio = gr.HTML()
                gr.Markdown("**Spectrogram**")
                output_spec = gr.HTML()
                timing = gr.HTML()
                error_box = gr.HTML()

        dit_dd.change(on_dit_change, inputs=[dit_dd], outputs=[decoder_dd])
        generate_btn.click(generate,
                           inputs=[dit_dd, decoder_dd, prompt, negative_prompt,
                                   seconds, steps, seed, cfg, apg, sigma_slider,
                                   init_audio, inpaint_text],
                           outputs=[output_audio, output_spec, timing, error_box])

        gr.Markdown(
            "<p style='color:#888; font-size:0.85em'>"
            f"WAVs saved under <code>output/gradio/</code> (rotates after {KEEP_RECENT_N} files). "
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
