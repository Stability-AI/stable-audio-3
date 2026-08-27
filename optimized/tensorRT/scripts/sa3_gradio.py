"""SA3 TRT — gradio web UI (debug build).

Adds (vs the MVP):
  - Model picker: sm-music / sm-sfx / medium (hot-swap with cached SA3Inference
    instances; first switch ~5-7s, subsequent instant)
  - Engine-variant picker per model: auto-detects all `dit_*.trt` files in the
    model dir (canonical fixed, archived buggy, fp32, etc.) so you can A/B
    different precisions / quantizations without restarting
  - Spectrogram display: Underfit-style 3-band tinted stereo mel spectrogram
    rendered inline alongside the audio

Launch:
    ./sa3-gradio                  # share=True by default, sm-music + same-s
    ./sa3-gradio --dit medium
    ./sa3-gradio --no-share       # local-only

The previously-required `--dit ...` is now just the *initial* model — the
runtime dropdown lets you switch between variants/models without restart.
"""
from __future__ import annotations
import argparse
import base64
import math
import sys
import html as html_lib
import subprocess
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sa3_trt import SA3Inference, SAMPLE_RATE, SAMPLES_PER_LATENT  # noqa: E402
import sa3_trt_core as canon  # noqa: E402
from spec import render_spectrogram_png  # noqa: E402


OUTPUT_DIR = SCRIPTS_DIR.parent / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEEP_RECENT_N = 20

from gradio_ui import (  # backend-agnostic UI layer, forked from optimized/mlx
    # The other _JS_* constants (promote / pause-others / scroll / playhead) are wired
    # inside render_player and render_history themselves, so they stay private to that
    # module; only the two this file attaches to its own events are imported.
    _JS_FIX_SLIDERS, _save_wav, render_history, render_player, render_queue_status, verbose_basename,
)

MODELS_ROOT = SCRIPTS_DIR.parent / "models" / canon.ARCH


# Default decoder for each DiT bundle.
DEFAULT_DECODERS = {"sm-music": "same-s", "sm-sfx": "same-s", "medium": "same-l"}

# DiT-name → on-disk subdir mapping (matches sa3_trt_core's DIT_CHOICES)
DIT_SUBDIRS = {"sm-music": "sa3-sm-music", "sm-sfx": "sa3-sm-sfx", "medium": "sa3-m"}


# Magic marker (matched against str(variant_path)) telling generate() to
# dispatch to the PyTorch-eager backend instead of TRT. Only offered for
# the medium+SAME-L+fp32 combo per the user's scope.
PT_EAGER_VARIANT = "<PT_EAGER_FP32_GT>"


def discover_variants(dit_name: str) -> list[tuple[str, Path]]:
    """Return [(label, path)] of available DiT engine files for this model.

    Scans models/<arch>/<dit_subdir>/dit_*.trt. The canonical engine
    (dit_fp16.trt) is always first if present; other variants follow
    alphabetically. For the medium DiT also appends a pseudo-variant
    "pytorch fp32 (GT)" that dispatches to the PyTorch-eager backend.
    """
    subdir = DIT_SUBDIRS.get(dit_name)
    if subdir is None:
        return []
    d = MODELS_ROOT / subdir
    if not d.exists():
        return []
    files = [f for f in sorted(d.glob("dit_*.trt")) if f.name not in RETIRED_DITS]
    canonical_name = "dit_fp16.trt"
    canonical = [f for f in files if f.name == canonical_name]
    others = [f for f in files if f.name != canonical_name]
    out = []
    for f in canonical + others:
        label = f.name[len("dit_"):-len(".trt")] if f.name.startswith("dit_") and f.name.endswith(".trt") else f.name
        if f.name == canonical_name:
            label = "fp16 (canonical)"
        elif "buggy" in f.name:
            label = label + " ← old, broken"
        out.append((label, f))
    # Pseudo-variant: PT FP32 GT (currently medium only).
    if dit_name == "medium":
        out.append(("pytorch fp32 GT (slow, vanilla eager)", Path(PT_EAGER_VARIANT)))
    return out


# Pre-chunkable engines. Removed from the model repo's download path and hidden from the picker.
RETIRED_DECODERS = {"dec_dynamic_bf16.trt", "dec_dynamic_triton_swa.trt", "dec_dynamic_fp32.trt",
                    "dec_fp8.trt", "dec_fp8_fast.trt", "dec_dynamic_fp16mixed.trt"}
# Experimental DiT engines that are not part of the shipped set. "fp16mixed" is the retired name
# for plain fp16 -- every tier is mixed-precision, so the qualifier only ever meant "old".
RETIRED_DITS = {"dit_fp16mixed.trt", "dit_fp16_bands4096_32768.trt", "dit_fp16_maxL32768.trt"}


def _precision_of(filename: str) -> str:
    """Pull the precision token out of an engine filename: dec_fp16_chunkable_limiter -> fp16."""
    for tok in ("fp32", "fp16", "bf16", "fp8"):
        if f"_{tok}" in filename:
            return tok
    return ""


def discover_decoder_variants(decoder_name: str) -> list[tuple[str, Path]]:
    """Return [(label, path)] of decoder quantization tiers for this decoder —
    canonical first. Uses the known tier set (canonical / fp8; see quantize/README.md); engines
    auto-download from HF on selection if missing. Any extra local dec_*.trt not in the known set
    are appended, so a self-built variant shows up — except the RETIRED pre-chunkable engines,
    which are filtered out. A leftover copy of one of those in models/ would otherwise appear in
    the picker and silently serve pre-limiter audio with a 5.7-8.1 GB scratch reservation.
    """
    out, known = [], set(RETIRED_DECODERS)
    for tier, fname in canon.DECODER_TIER_FILENAME.get(decoder_name, {}).items():
        # Name the tier by the precision it actually is -- SAME-L canonical is fp16, SAME-S is
        # bf16 -- rather than rendering "canonical (canonical)".
        prec = _precision_of(fname)
        label = f"{prec} (canonical)" if tier == "canonical" else (prec or tier)
        out.append((label, canon.ARCH_DIR / decoder_name / fname))
        known.add(fname)
    d = canon.ARCH_DIR / decoder_name
    if d.exists():
        for f in sorted(d.glob("dec_*.trt")):
            if f.name in known:
                continue
            stem = f.name
            label = (stem[len("dec_dynamic_"):-len(".trt")] if stem.startswith("dec_dynamic_")
                     else stem[len("dec_"):-len(".trt")])
            out.append((label, f))
    return out


def _prune_old_outputs():
    wavs = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in wavs[KEEP_RECENT_N:]:
        try:
            old.unlink()
        except OSError:
            pass


# ── SA3Inference cache (model+variant → instance) ──────────────────────────
# Each cached entry holds 5-15 GB of VRAM (engines + persistent graph buffers).
# H100 80GB fits ~5 sm-music or ~3 medium combos before OOM. We LRU-evict to
# stay under that limit. Switching back to an evicted variant re-loads (~5-7s).
_inference_cache: dict[tuple[str, str, str, str, bool], SA3Inference] = {}
_inference_lru: list[tuple[str, str, str, str]] = []  # MRU at end
_INFERENCE_CACHE_MAX = 2


def _evict_inference(key):
    """Free the engines + caches held by a cached SA3Inference before dropping it.

    The teardown ORDER is load-bearing and lives in SA3Inference.close(): captured CUDA
    graphs and the eager contexts must go before the runners whose contexts and shared
    scratch they point into. This used to free the runners first and clear the graphs
    after, which poisoned the CUDA context — the next render in a *different* instance
    died with `an illegal memory access was encountered`, so switching models past the
    cache limit bricked the server until restart.
    """
    inf = _inference_cache.pop(key, None)
    if inf is None:
        return
    try:
        inf.close()
    except Exception as e:
        print(f"    (eviction cleanup raised {type(e).__name__}: {e})")
    del inf
    import gc
    gc.collect()


# Set once from --chunking in main(). A module-level default keeps it out of every call site
# and out of the UI, which is right: it is a deployment choice (memory vs speed), not a per-render
# one -- switching it means selecting a different optimization profile, so the engines reload.
_CHUNKING = True


def get_inference(dit: str, decoder: str, dit_variant_path: str,
                   dec_variant_path: str,
                   default_T_lat: int, default_steps: int,
                   default_seconds: float, quiet: bool,
                   chunking: bool | None = None) -> SA3Inference:
    """Return a warm SA3Inference, building (and caching) if not yet loaded.

    The cache key is (dit, decoder, dit_variant_path, dec_variant_path) so
    swapping either the DiT *or* the decoder variant triggers a fresh load.
    LRU eviction caps the cache at _INFERENCE_CACHE_MAX entries.
    """
    chunking = _CHUNKING if chunking is None else chunking
    # chunking is part of the key: it selects the profile, so the two modes are different
    # contexts with different scratch and cannot share a cached SA3Inference.
    key = (dit, decoder, dit_variant_path, dec_variant_path, chunking)
    if key in _inference_cache:
        if key in _inference_lru:
            _inference_lru.remove(key)
        _inference_lru.append(key)
        return _inference_cache[key]

    # LRU evict BEFORE loading new — frees VRAM so the new load doesn't OOM.
    while len(_inference_cache) >= _INFERENCE_CACHE_MAX:
        oldest = _inference_lru.pop(0)
        print(f"\n  ← LRU-evicting SA3Inference{oldest[:2]} "
              f"dit={Path(oldest[2]).name} dec={Path(oldest[3]).name}")
        _evict_inference(oldest)

    # Override the canonical engine path lookups before constructing.
    canon.DIT_CHOICES[dit]["engine"] = Path(dit_variant_path)
    canon.DECODER_PATHS[decoder] = Path(dec_variant_path)
    dec_tier = canon.decoder_tier_from_filename(decoder, Path(dec_variant_path).name)
    print(f"\n  → loading SA3Inference({dit!r}, {decoder!r}, "
          f"dit={Path(dit_variant_path).name}, "
          f"dec={Path(dec_variant_path).name}, tier={dec_tier})")
    inf = SA3Inference(dit, decoder,
                        dec_precision=dec_tier,
                        chunking=chunking,
                        # pass the picked files explicitly -- mutating canon.DIT_CHOICES /
                        # DECODER_PATHS above does not reach the resolver, so the dropdowns
                        # were silently ignored for the DiT.
                        dit_engine=(None if str(dit_variant_path).startswith("<")
                                    else dit_variant_path),
                        dec_engine=dec_variant_path,
                        default_T_lat=default_T_lat,
                        default_steps=default_steps,
                        default_seconds=default_seconds,
                        quiet=quiet)
    _inference_cache[key] = inf
    _inference_lru.append(key)
    return inf


# ── Gradio UI ──────────────────────────────────────────────────────────────
_css = ("#sa3-promote{display:none !important}"
            "#sa3-out{gap:4px !important}"
            "#sa3-out .html-container{padding:0 !important; margin:0 !important}")


def build_ui(initial_dit: str, initial_decoder: str, *,
             share: bool, quiet: bool,
             default_T_lat: int, default_steps: int, default_seconds: float):
    """Same UI as the MLX gradio -- history, a pre-rendered next take, radio mode, the rich
    player -- over the TensorRT backend, plus the controls only TRT has (engine variant per
    model, decoder quantisation tier, chunked vs single-shot).

    No LoRA panel: the TRT engines carry no LoRA branches, so there is nothing to hot-swap
    without first building refit-capable or branch-capable engines.
    """
    import gradio as gr

    initial_variants = discover_variants(initial_dit)
    if not initial_variants:
        raise RuntimeError(f"no DiT engines found for {initial_dit} under {MODELS_ROOT}")
    initial_variant_path = str(initial_variants[0][1])
    initial_dec_variants = discover_decoder_variants(initial_decoder)
    if not initial_dec_variants:
        raise RuntimeError(f"no decoder engines found for {initial_decoder} under {MODELS_ROOT}")
    initial_dec_variant_path = str(initial_dec_variants[0][1])
    get_inference(initial_dit, initial_decoder, initial_variant_path,
                  initial_dec_variant_path, default_T_lat, default_steps,
                  default_seconds, quiet)

    DIT_OPTIONS = list(DEFAULT_DECODERS.keys())
    # The DiT profile runs L=1..4096; the binding limits are the chunkable decoders'
    # profile floor (32 latents) and that same 4096 ceiling. Clamp the slider to both so
    # the UI cannot ask for a length the decoder will refuse.
    MIN_SECONDS = math.ceil(canon.DECODER_MIN_L * SAMPLES_PER_LATENT / SAMPLE_RATE)   # 3 s
    MAX_SECONDS = SA3Inference.DIT_MAX_L * SAMPLES_PER_LATENT / SAMPLE_RATE // 1      # 380 s

    def on_dit_change(dit_name):
        variants = discover_variants(dit_name)
        suggested = DEFAULT_DECODERS.get(dit_name, "same-s")
        dec_variants = discover_decoder_variants(suggested)
        var_u = (gr.update(choices=[(l, str(p)) for l, p in variants], value=str(variants[0][1]))
                 if variants else gr.update(choices=[], value=None))
        dec_u = (gr.update(choices=[(l, str(p)) for l, p in dec_variants],
                           value=str(dec_variants[0][1]))
                 if dec_variants else gr.update(choices=[], value=None))
        return var_u, gr.update(value=suggested), dec_u

    def on_decoder_change(decoder_name):
        dv = discover_decoder_variants(decoder_name)
        if not dv:
            return gr.update(choices=[], value=None)
        return gr.update(choices=[(l, str(p)) for l, p in dv], value=str(dv[0][1]))

    def on_seconds_change(sec):
        """Keep the inpaint range inside the clip length."""
        return gr.update(maximum=float(sec)), gr.update(maximum=float(sec))

    # ── one generation, packaged as a history entry ──────────────────────────────────────────
    def _entry(dit_name, decoder_name, variant_path, dec_variant_path, prompt, negative_prompt,
               seconds, steps, seed_text, cfg, apg, sigma_max, chunking, fmt,
               a2a_path, a2a_sigma, inpaint_path, inp_start, inp_end):
        if not prompt or not prompt.strip():
            return None, "empty prompt"
        try:
            seed = int(seed_text.strip()) if seed_text and seed_text.strip() else None
        except ValueError:
            return None, "seed must be an integer"
        if str(variant_path) == PT_EAGER_VARIANT:
            return None, "PT FP32 GT is not wired through this backend yet"
        try:
            inf = get_inference(dit_name, decoder_name, str(variant_path), str(dec_variant_path),
                                default_T_lat, default_steps, default_seconds, quiet,
                                chunking=bool(chunking))
        except Exception as e:
            return None, f"load failed: {type(e).__name__}: {e}"

        # Which reference wins, and -- just as important -- SAY SO when something the user
        # supplied is being dropped. Silently falling through to a different mode is the
        # worst outcome here: upload an inpaint reference, forget the range sliders, and a
        # bare if/elif quietly runs audio-to-audio instead.
        #
        # The engines take ONE reference (generate_eager reads a single init_audio_path and
        # derives the inpaint keep-mask from its latents), so inpaint and audio-to-audio
        # cannot use two different files the way the MLX backend can.
        kw, mode, notes = {}, "text-to-audio", []
        sec = float(seconds)
        rng = None
        if inpaint_path and float(inp_end) > float(inp_start) and float(inp_start) < sec:
            if float(inp_end) > sec:
                notes.append(f"inpaint end clamped to the clip length ({sec:g}s)")
            rng = (float(inp_start), float(min(float(inp_end), sec)))
        elif inpaint_path and float(inp_end) <= float(inp_start):
            notes.append("inpainting ignored — set the start/end range sliders")
        elif inpaint_path:
            notes.append(f"inpainting ignored — start ({float(inp_start):g}s) is past "
                         f"the end of the clip ({sec:g}s)")
        elif float(inp_end) > float(inp_start):
            notes.append("inpaint range ignored — no reference audio uploaded")

        if rng is not None:
            kw = dict(init_audio_path=inpaint_path, inpaint_range=rng,
                      init_noise_level=float(sigma_max))
            mode = "inpaint"
            if a2a_path:
                notes.append("using the inpaint reference — one reference per render, and "
                             "inpainting needs it to keep the untouched part")
        elif a2a_path:
            # a2a's own slider IS the schedule start (parent-repo semantics), so it
            # overrides sigma_max rather than stacking with it.
            kw = dict(init_audio_path=a2a_path, init_noise_level=float(a2a_sigma))
            mode = "audio-to-audio"
            if float(sigma_max) != 1.0:
                notes.append(f"sigma_max {float(sigma_max):g} ignored — audio-to-audio uses "
                             f"its own noise level ({float(a2a_sigma):g})")
        elif float(sigma_max) != 1.0:
            kw = dict(init_noise_level=float(sigma_max))
        neg = (negative_prompt or "").strip()
        if float(cfg) != 1.0:
            kw.update(cfg=float(cfg), apg=float(apg), negative_prompt=neg or None)
        elif neg:
            notes.append("negative prompt ignored — it only applies when CFG > 1")
        try:
            pcm, t = inf.generate(prompt.strip(), seconds=float(seconds), steps=int(steps),
                                  seed=seed, **kw)
        except NotImplementedError as e:
            return None, f"not supported by this engine: {e}"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

        base = verbose_basename(prompt, neg, float(cfg), float(sigma_max), t["seed"])
        out_path = OUTPUT_DIR / f"{base}.wav"
        _save_wav(pcm, out_path)
        if fmt in ("flac", "mp3"):
            conv = out_path.with_suffix("." + fmt)
            rc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                 "-i", str(out_path), str(conv)],
                                capture_output=True).returncode
            if rc == 0 and conv.exists():
                out_path = conv
        mime = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}[out_path.suffix[1:]]
        try:
            t0 = time.time()
            png = render_spectrogram_png(pcm, sample_rate=SAMPLE_RATE, width=1400, height=280)
            spec_b64 = base64.b64encode(png).decode("ascii")
            spec_ms = (time.time() - t0) * 1000
        except Exception:
            spec_b64, spec_ms = "", 0.0
        timing = (f"engine-load {t.get('graph_build_ms', 0):.0f} ms &nbsp;·&nbsp; "
                  f"Inference : {t['inference_ms']:.1f} ms &nbsp;·&nbsp; "
                  f"{t.get('realtime', 0):.0f}× realtime &nbsp;·&nbsp; "
                  f"seed : {t['seed']} &nbsp;·&nbsp; T_lat : {t['T_lat']} &nbsp;·&nbsp; "
                  f"samples : {t['samples']} &nbsp;·&nbsp; spec {spec_ms:.0f} ms")
        return {
            "notes": notes,
            "key": f"k{time.time_ns()}", "ts": time.time(), "dit": dit_name,
            "neg": neg, "cfg": float(cfg), "smx": float(sigma_max),
            "path": str(out_path), "mime": mime, "name": out_path.name,
            "size_mb": out_path.stat().st_size / 1e6, "mode": mode,
            "prompt": prompt.strip(), "seed": t["seed"],
            "lora": "",                       # kept for the shared renderer's contract
            "spec_b64": spec_b64, "timing": timing,
        }, None

    def _present(state, entry, opts, queued_panel, *, force_autoplay=False):
        """(player, timing, error, history, queued, state) for the shared renderers."""
        auto = force_autoplay or ("Auto-play" in (opts or []))
        radio = "Radio" in (opts or [])
        loop = "Loop" in (opts or [])
        hist = state.get("history", [])
        player = render_player(entry, autoplay=auto, radio=radio, advance=radio, loop=loop,
                               autodl=("Auto-download" in (opts or []))) if entry else ""
        note_html = "".join(
            f"<div style='color:#e8a33d;font-size:0.85em'>note: {html_lib.escape(n)}</div>"
            for n in (entry.get("notes") or [])) if entry else ""
        return (player, (entry.get("timing", "") + note_html) if entry else "", "",
                render_history(hist, advance=radio, loop=loop),
                queued_panel, state)

    CTRL = None  # bound after the widgets exist

    def generate(*args):
        *ctrl, opts, state = args
        state = dict(state or {"current": None, "queued": None, "history": []})
        q = state.get("queued")
        if q is not None:                      # a pre-rendered take is waiting: show it instantly
            state["queued"] = None
            state["current"] = q
            state["history"] = ([q] + state.get("history", []))[:24]
            return _present(state, q, opts, render_queue_status(generating=True),
                            force_autoplay=True)
        entry, err = _entry(*ctrl)
        if err:
            return ("", "", f"<span style='color:#f88'>error: {html_lib.escape(err)}</span>",
                    render_history(state.get("history", [])), render_queue_status(), state)
        state["current"] = entry
        state["history"] = ([entry] + state.get("history", []))[:24]
        return _present(state, entry, opts, render_queue_status(), force_autoplay=True)

    def pregen(*args):
        """Render the NEXT take in the background so the following click is instant."""
        *ctrl, opts, state = args
        state = dict(state or {"current": None, "queued": None, "history": []})
        if state.get("queued") is not None:
            return render_queue_status(state["queued"]), state
        entry, err = _entry(*ctrl)
        state["queued"] = entry if not err else None
        return render_queue_status(state.get("queued")), state

    with gr.Blocks(title="SA3 · TensorRT", css=_css) as demo:
        st = gr.State({"current": None, "queued": None, "history": []})
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    dit_dd = gr.Dropdown(label="DiT model", choices=DIT_OPTIONS,
                                         value=initial_dit, scale=1)
                    dec_dd = gr.Dropdown(label="Decoder", choices=["same-s", "same-l"],
                                         value=initial_decoder, scale=1)
                with gr.Row():
                    var_dd = gr.Dropdown(label="DiT variant", allow_custom_value=True,
                                         choices=[(l, str(p)) for l, p in initial_variants],
                                         value=initial_variant_path, scale=1)
                    dec_var_dd = gr.Dropdown(label="Decoder variant", allow_custom_value=True,
                                             choices=[(l, str(p)) for l, p in initial_dec_variants],
                                             value=initial_dec_variant_path, scale=1)
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=2, scale=6,
                                        placeholder="warm analog house groove")
                    seed = gr.Textbox(label="Seed (optional)", max_lines=1, value="", scale=1)
                with gr.Row():
                    seconds = gr.Slider(label="Seconds", minimum=MIN_SECONDS,
                                        maximum=MAX_SECONDS, step=1,
                                        value=default_seconds, scale=2)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16, step=1,
                                      value=default_steps, scale=1)
                    cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0, step=0.1,
                                    value=1.0, scale=1)
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        apg = gr.Slider(label="APG (only applies when CFG > 1)",
                                        minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                        sigma_global = gr.Slider(label="sigma_max", minimum=0.05, maximum=1.0,
                                                 step=0.05, value=1.0)
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=1)
                    chunking = gr.Checkbox(
                        label="Chunked decode (low VRAM)", value=True,
                        info="On: 256-latent windows, ~509 MB of decoder scratch. Off: single-shot "
                             "on the wide profile — 10-20% faster above L=256, ~5.4 GB more resident.")
                with gr.Accordion("Audio-to-audio (guide the whole clip)", open=False):
                    a2a_audio = gr.Audio(label="Guide audio — generation starts from its latents",
                                         type="filepath")
                    a2a_sigma = gr.Slider(label="sigma_max (lower = closer to the guide)",
                                          minimum=0.05, maximum=1.0, step=0.05, value=0.6)
                with gr.Accordion("Inpainting (regenerate a span)", open=False):
                    inpaint_audio = gr.Audio(label="Reference audio", type="filepath")
                    with gr.Row():
                        inp_start = gr.Slider(label="Start (s)", minimum=0,
                                              maximum=default_seconds, step=0.5, value=0)
                        inp_end = gr.Slider(label="End (s)", minimum=0,
                                            maximum=default_seconds, step=0.5, value=0)
                with gr.Accordion("Output", open=False):
                    output_opts = gr.CheckboxGroup(
                        label="Playback", choices=["Auto-play", "Radio", "Loop", "Auto-download"],
                        value=["Auto-play"])
                    file_format = gr.Radio(label="File format", choices=["wav", "flac", "mp3"],
                                           value="wav")
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                promote_btn = gr.Button("", elem_id="sa3-promote")   # CSS-hidden, DOM-present
            with gr.Column(scale=4):
                output_player = gr.HTML()
                timing = gr.HTML()
                error_box = gr.HTML()
                queued_html = gr.HTML()
                history_html = gr.HTML()

        dit_dd.change(on_dit_change, inputs=[dit_dd], outputs=[var_dd, dec_dd, dec_var_dd])
        dec_dd.change(on_decoder_change, inputs=[dec_dd], outputs=[dec_var_dd])
        seconds.change(on_seconds_change, inputs=[seconds], outputs=[inp_start, inp_end]
                       ).then(None, js=_JS_FIX_SLIDERS)

        CTRL = [dit_dd, dec_dd, var_dd, dec_var_dd, prompt, negative_prompt, seconds, steps,
                seed, cfg, apg, sigma_global, chunking, file_format,
                a2a_audio, a2a_sigma, inpaint_audio, inp_start, inp_end]
        main_out = [output_player, timing, error_box, history_html, queued_html, st]
        # ⚠ js= takes a JS *function expression* ("() => ...") — gradio evaluates it as one.
        # _JS_TRY_PLAY is an inline-attribute body (bare statements, `this` = the <audio>),
        # so passing it here is a syntax error that kills the whole frontend: the config
        # fails to evaluate and the page sits on "Loading..." forever with zero controls.
        # The autoplay rescue belongs where it already is — render_player's oncanplay.
        for btn in (generate_btn, promote_btn):
            btn.click(generate, inputs=CTRL + [output_opts, st], outputs=main_out
                      ).then(pregen, inputs=CTRL + [output_opts, st],
                             outputs=[queued_html, st]
                      ).then(None, js=_JS_FIX_SLIDERS)
        gr.Markdown("Renders are written to `output/`. **Radio** auto-advances to the next take as "
                    "each finishes; the next one is pre-rendered while you listen.")

    demo.queue(max_size=16).launch(share=share, server_name="0.0.0.0", show_error=True,
                                   allowed_paths=[str(OUTPUT_DIR)])
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", choices=list(DEFAULT_DECODERS.keys()),
                    default="sm-music",
                    help="Initial DiT bundle (switchable at runtime)")
    ap.add_argument("--decoder", choices=["same-s", "same-l"], default=None,
                    help="Initial decoder. Default: pairs with --dit")
    ap.add_argument("--default-seconds", type=float, default=120.0,
                    help="Length to pre-warm the initial graph at")
    ap.add_argument("--default-steps", type=int, default=8)
    ap.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                    help="Create a public gradio.live URL (default on)")
    ap.add_argument("--chunking", action=argparse.BooleanOptionalAction, default=True,
                    help="SAME-L: decode/encode in windows on the engines' low profile "
                         "(509 MB of decoder scratch instead of 8143). --no-chunking uses the "
                         "wide profile: single-shot, faster above L=256, ~5.4 GB more resident.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    global _CHUNKING
    _CHUNKING = args.chunking

    if args.decoder is None:
        args.decoder = DEFAULT_DECODERS[args.dit]

    T_lat = max(1, math.ceil(args.default_seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if args.decoder == "same-s" and T_lat % 2 != 0:
        T_lat += 1

    print(f"\n━━━ SA3 TRT — gradio (debug build) ━━━")
    print(f"  initial dit:      {args.dit}")
    print(f"  initial decoder:  {args.decoder}")
    print(f"  warmup:           T_lat={T_lat}  steps={args.default_steps}  "
          f"(~{args.default_seconds}s)")
    # Pretty-print what variants are visible
    for ditname in DEFAULT_DECODERS:
        vs = discover_variants(ditname)
        print(f"  dit-variants[{ditname}]: " + (
            ", ".join(lbl for lbl, _ in vs) if vs else "(none found)"))
    for dec in ("same-s", "same-l"):
        dvs = discover_decoder_variants(dec)
        print(f"  dec-variants[{dec}]:    " + (
            ", ".join(lbl for lbl, _ in dvs) if dvs else "(none found)"))
    print()

    build_ui(args.dit, args.decoder, share=args.share, quiet=args.quiet,
              default_T_lat=T_lat, default_steps=args.default_steps,
              default_seconds=args.default_seconds)


if __name__ == "__main__":
    main()
