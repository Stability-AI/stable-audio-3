"""SA3 cpu-amx — gradio web UI (torch-free C++ AMX engines on CPU).

The cpu-amx sibling of optimized/mlx/scripts/sa3_gradio.py, with every generation
mode wired: text-to-audio, CFG + negative prompt + APG, audio-to-audio, and
inpainting. Each clip renders as a 3-band tinted stereo mel spectrogram (numpy
port — no torch) with a click-to-seek playhead; only one clip plays at a time.

Layout mirrors the MLX app (model/decoder/precision row, prompt+seed,
seconds/steps/cfg, Advanced = apg/σmax/negative prompt, Audio-to-audio and
Inpainting accordions, Output options) minus the Apple-only bits (LoRA,
Infinite-Radio/Hotswap) that don't apply here.

Robustness: the int8 C++ DiT core heap-corrupts if it is invoked at more than one
sequence length in a single process (and double-frees at teardown). So each
generation runs the tested CLI (scripts/sa3_cpu_amx.py) in a FRESH subprocess —
one T_lat per process, os._exit(0) before teardown — which makes arbitrary
seconds/steps changes safe. Models reload per generation (mmap, a few seconds).

Launch:
    ./sa3-gradio                 # same-l default, public share link
    ./sa3-gradio --decoder same-s
    ./sa3-gradio --no-share      # local only
"""
from __future__ import annotations

import argparse
import base64
import html as html_lib
import math
import subprocess
import sys
import tempfile
import time
import urllib.parse
import wave
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from spec import render_spectrogram_png  # noqa: E402

CLI = str(SCRIPTS / "sa3_cpu_amx.py")
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
OUTPUT_DIR = ROOT / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECODER_CHOICES = ["same-l", "same-s"]
PRECISION_CHOICES = ["bf16", "int8"]
MAX_SECONDS = 380   # medium's trained max
MIN_SIGMA = 0.01


# ── run one generation via the CLI subprocess (DiT-crash isolation) ─────────
def run_generation(decoder, precision, threads, prompt, negative_prompt, seconds,
                   steps, seed, cfg, apg, sigma_max, a2a_path, inpaint_path,
                   inp_start, inp_end):
    """Returns (audio_np (2,T) float32, info dict). Raises on failure."""
    out = Path(tempfile.gettempdir()) / f"sa3cpuamx_gr_{time.time_ns()}.wav"
    cmd = [sys.executable, CLI, "--dit", "medium",
           "--decoder", str(decoder), "--decoder-precision", str(precision),
           "--threads", str(int(threads)), "--prompt", prompt or "",
           "--seconds", str(float(seconds)), "--steps", str(int(steps)),
           "--seed", str(int(seed)), "--cfg", str(float(cfg)), "--apg", str(float(apg)),
           "--out", str(out)]
    # init audio: inpaint reference takes priority for the encode; a2a guide otherwise
    init_audio = inpaint_path or a2a_path
    if a2a_path and not inpaint_path:
        cmd += ["--init-audio", a2a_path, "--init-noise-level", str(float(sigma_max))]
    elif init_audio:
        cmd += ["--init-audio", init_audio]
    if cfg != 1.0 and negative_prompt and negative_prompt.strip():
        cmd += ["--negative-prompt", negative_prompt.strip()]
    if inpaint_path and inp_end > inp_start:
        cmd += ["--inpaint-range", f"{float(inp_start)},{float(min(inp_end, seconds))}"]
        if a2a_path:
            cmd += ["--init-noise-level", str(float(sigma_max))]
    elif not a2a_path and sigma_max != 1.0:
        cmd += ["--init-noise-level", str(float(sigma_max))]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(ROOT))
    wall = time.time() - t0
    if proc.returncode != 0 or not out.exists():
        last = (proc.stderr.strip() or proc.stdout.strip() or "(no output)").splitlines()[-1]
        raise RuntimeError(last[:200])

    with wave.open(str(out), "rb") as w:
        nch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), np.int16).reshape(-1, nch).T.astype(np.float32) / 32767.0
    out.unlink(missing_ok=True)
    T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    info = {"wall": wall, "T_lat": T_lat, "samples": pcm.shape[-1],
            "realtime": (pcm.shape[-1] / SAMPLE_RATE) / max(wall, 1e-9), "seed": int(seed)}
    return pcm, info


# ── HTML player (spectrogram + seekable audio, one-at-a-time playback) ───────
# On play, pause every other <audio> so only one clip sounds at a time.
_JS_PAUSE = ("var t=this;document.querySelectorAll('audio').forEach(function(o){if(o!==t)o.pause();});")
_JS_PLAYHEAD = ("var p=this.closest('.blk').querySelector('.ph');"
                "if(p&&this.duration)p.style.left=(this.currentTime/this.duration*100)+'%';")
_JS_SEEK = ("var a=this.closest('.blk').querySelector('audio');"
            "var r=this.getBoundingClientRect();"
            "if(a&&a.duration){a.currentTime=(event.clientX-r.left)/r.width*a.duration;a.play();}")


def _save_wav(pcm_int16, path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_int16.tobytes())


def render_player(entry, *, small=False, autoplay=False):
    auto = "autoplay " if autoplay else ""
    src = "gradio_api/file=" + urllib.parse.quote(entry["path"], safe="/")
    audio_el = (f'<audio controls {auto}style="width:100%" '
                f'onplay="{_JS_PAUSE}" ontimeupdate="{_JS_PLAYHEAD}" src="{src}"></audio>')
    spec_core = ""
    if entry.get("spec_b64"):
        h = "height:56px;" if small else ""
        tip = "3-band tinted stereo mel · red=bass green=mid blue=high · L top R bottom · click to seek"
        spec_core = (f'<div style="position:relative;cursor:pointer" title="{tip}" onclick="{_JS_SEEK}">'
                     f'<img src="data:image/png;base64,{entry["spec_b64"]}" '
                     f'style="width:100%;{h}display:block;image-rendering:pixelated;border:1px solid #333"/>'
                     f'<div class="ph" style="position:absolute;top:0;bottom:0;left:0%;width:2px;'
                     f'background:#fff;pointer-events:none;box-shadow:0 0 4px rgba(0,0,0,.8)"></div></div>')
    prompt_disp = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"
    if small:
        row = (f'<div style="display:flex;gap:8px;align-items:center">'
               f'<div style="flex:1;min-width:0">{audio_el}</div>'
               f'<div style="flex:1;min-width:0">{spec_core}</div></div>')
        cap = (f'<div style="font-size:0.8em;margin:2px 0;color:#888">{prompt_disp} · '
               f'{entry["meta"]} · seed {entry["seed"]}</div>')
        bg = entry.get("bg", "")
        return (f'<div class="blk" style="padding:6px 8px;{bg}">{row}{cap}</div>')
    return f'<div class="blk">{audio_el}{spec_core}</div>'


def render_history(hist):
    if not hist:
        return ""
    items = "".join(render_player(dict(e, bg=("background:rgba(127,127,127,0.20);" if i % 2 else "")),
                                  small=True) for i, e in enumerate(hist))
    return (f'<div style="font-weight:600;margin-top:14px">Previous generations ({len(hist)})</div>'
            f'<div style="max-height:480px;overflow-y:auto;margin-top:6px;padding-right:6px">{items}</div>')


# ── UI ──────────────────────────────────────────────────────────────────────
def build_ui(initial_decoder, initial_precision, *, share, default_seconds,
             default_steps, default_threads, server_port):
    import gradio as gr

    def _meta(decoder, precision, cfg, sigma_max, mode):
        parts = [decoder, precision]
        if cfg != 1.0:
            parts.append(f"cfg {cfg:g}")
        if sigma_max != 1.0:
            parts.append(f"σ {sigma_max:g}")
        if mode != "text-to-audio":
            parts.append(mode)
        return " · ".join(parts)

    def generate(decoder, precision, threads, prompt, negative_prompt, seconds, steps,
                 seed_text, cfg, apg, sigma_max, init_noise, a2a_audio, inpaint_audio,
                 inp_start, inp_end, autoplay, state):
        import random as _random
        prompt = (prompt or "").strip()
        try:
            seed = int(seed_text) if seed_text and str(seed_text).strip() else _random.randint(1, 9999)
        except ValueError:
            seed = _random.randint(1, 9999)
        seconds = float(min(seconds, MAX_SECONDS))
        # a2a's init_noise overrides the global σmax (parent-repo semantics)
        smx = float(init_noise) if a2a_audio else float(sigma_max)
        smx = max(smx, MIN_SIGMA)
        mode = ("inpaint" if (inpaint_audio and inp_end > inp_start) else
                "audio-to-audio" if a2a_audio else "text-to-audio")
        try:
            pcm, info = run_generation(decoder, precision, threads, prompt, negative_prompt,
                                       seconds, steps, seed, cfg, apg, smx,
                                       a2a_audio or None, inpaint_audio or None, inp_start, inp_end)
        except Exception as e:
            return (gr.update(), f"<span style='color:#f88'>error: {e}</span>", gr.update(), state)

        pcm_i16 = (np.clip(pcm, -1, 1) * 32767.0).astype(np.int16).T   # (T,2)
        name = f"{(prompt[:40] or 'out').replace('/', '-')}.{seed}.wav"
        out_path = OUTPUT_DIR / name
        _save_wav(pcm_i16, out_path)
        spec_b64 = None
        try:
            spec_b64 = base64.b64encode(
                render_spectrogram_png(pcm_i16, SAMPLE_RATE, 1200, 240)).decode("ascii")
        except Exception:
            pass
        entry = {"prompt": prompt, "seed": seed, "path": str(out_path), "spec_b64": spec_b64,
                 "meta": _meta(decoder, precision, cfg, smx, mode)}
        timing = (f"{html_lib.escape(prompt) or '<i>(no prompt)</i>'} · "
                  f"<b>{info['wall']:.1f}s wall</b> · {info['realtime']:.2f}× realtime · "
                  f"seed <code>{seed}</code> · T_lat {info['T_lat']} · {entry['meta']}")
        if state["current"]:
            state["history"].insert(0, state["current"])
        state["current"] = entry
        main = render_player(entry, autoplay=bool(autoplay))
        return main, timing, render_history(state["history"]), state

    css = "#out .html-container{padding:0 !important;margin:0 !important}#out{gap:4px !important}"
    with gr.Blocks(title="SA3 cpu-amx", css=css) as demo:
        gr.Markdown(
            "# SA3 — CPU AMX (torch-free C++ engines)\n"
            "Text→audio, CFG + negative prompt, audio-to-audio, inpainting — all on CPU "
            "(Xeon AMX). DiT = **medium int8**; decoders = SAME-S / SAME-L (**bf16** default, "
            "int8 optional). audio-to-audio / inpainting init-encode uses the torch-free C++ AMX "
            "SAME-S/SAME-L encoder (matched to the decoder); the whole stack is now 100% C++/numpy.")
        st = gr.State({"current": None, "history": []})
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    gr.Dropdown(label="DiT model", choices=["medium"], value="medium",
                                interactive=False, scale=1)
                    decoder_dd = gr.Dropdown(label="Decoder (codec)", choices=DECODER_CHOICES,
                                             value=initial_decoder, scale=1)
                    precision_dd = gr.Dropdown(label="Decoder precision", choices=PRECISION_CHOICES,
                                               value=initial_precision, scale=1)
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=2, scale=6,
                                        placeholder="e.g. 'warm analog synthwave, driving bassline, 120 bpm'")
                    seed = gr.Textbox(label="Seed (optional)", max_lines=1, value="", scale=1, min_width=80)
                with gr.Row():
                    seconds = gr.Slider(label="Seconds", minimum=1, maximum=MAX_SECONDS,
                                        value=default_seconds, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16, value=default_steps, step=1)
                    cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        apg = gr.Slider(label="APG (only when CFG > 1)", minimum=0.0, maximum=1.0,
                                        value=1.0, step=0.05)
                        sigma_global = gr.Slider(label="σmax", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                        threads = gr.Slider(label="Threads (T5 + decoder)", minimum=1, maximum=64,
                                            value=default_threads, step=1)
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=1)
                with gr.Accordion("Audio-to-audio (guide the whole clip)", open=False):
                    a2a_audio = gr.Audio(label="Guide audio — generation starts from its latents",
                                         type="filepath")
                    sigma_slider = gr.Slider(label="init_noise_level (1.0 = prompt, ~0.6 = variation, 0 = input)",
                                             minimum=0.0, maximum=1.0, value=0.7, step=0.01)
                with gr.Accordion("Inpainting (regenerate a span of reference audio)", open=False):
                    inpaint_audio = gr.Audio(label="Reference audio — kept bit-exact outside the range",
                                             type="filepath")
                    with gr.Row():
                        inp_start = gr.Slider(label="Start (s)", minimum=0, maximum=MAX_SECONDS, value=0, step=0.5)
                        inp_end = gr.Slider(label="End (s)", minimum=0, maximum=MAX_SECONDS, value=0, step=0.5)
                with gr.Accordion("Output", open=False):
                    autoplay = gr.Checkbox(label="Auto-play", value=True)
            with gr.Column(scale=2, elem_id="out"):
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                gr.Markdown("**Output**")
                output_player = gr.HTML()
                timing = gr.HTML()
                history_html = gr.HTML()

        def on_seconds_change(sec):
            return gr.update(maximum=sec), gr.update(maximum=sec)
        seconds.change(on_seconds_change, [seconds], [inp_start, inp_end])

        generate_btn.click(
            generate,
            [decoder_dd, precision_dd, threads, prompt, negative_prompt, seconds, steps,
             seed, cfg, apg, sigma_global, sigma_slider, a2a_audio, inpaint_audio,
             inp_start, inp_end, autoplay, st],
            [output_player, timing, history_html, st])

        gr.Markdown("<p style='color:#888;font-size:0.85em'>WAVs saved under "
                    "<code>output/gradio/</code>. Each generation runs in a fresh process "
                    "(the C++ DiT is single-length per process).</p>")

    demo.queue(max_size=8).launch(share=share, server_name="0.0.0.0", server_port=server_port,
                                  allowed_paths=[str(OUTPUT_DIR)], show_error=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", choices=DECODER_CHOICES, default="same-l",
                    help="Initial decoder (switchable in the UI). Default same-l.")
    ap.add_argument("--decoder-precision", choices=PRECISION_CHOICES, default="bf16")
    ap.add_argument("--default-seconds", type=float, default=10.0)
    ap.add_argument("--default-steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                    help="Create a public gradio.live URL (default on)")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    print("\n━━━ SA3 cpu-amx — gradio ━━━")
    print(f"  DiT:      medium (int8 C++ AMX)")
    print(f"  decoder:  {args.decoder} ({args.decoder_precision})  (switchable in UI)")
    build_ui(args.decoder, args.decoder_precision, share=args.share,
             default_seconds=args.default_seconds, default_steps=args.default_steps,
             default_threads=args.threads, server_port=args.port)


if __name__ == "__main__":
    main()
