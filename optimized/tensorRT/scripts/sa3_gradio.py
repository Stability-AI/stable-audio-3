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
import time
import uuid
import wave
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sa3_trt import SA3Inference, SAMPLE_RATE, SAMPLES_PER_LATENT  # noqa: E402
import sa3_trt_core as canon  # noqa: E402
from spec import render_spectrogram_png  # noqa: E402


OUTPUT_DIR = SCRIPTS_DIR.parent / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEEP_RECENT_N = 20

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
    files = sorted(d.glob("dit_*.trt"))
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
                    "dec_fp8.trt", "dec_fp8_fast.trt"}


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
        label = f"{tier} (canonical)" if tier == "canonical" else tier
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


def _save_wav(pcm, out_path):
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


# ── SA3Inference cache (model+variant → instance) ──────────────────────────
# Each cached entry holds 5-15 GB of VRAM (engines + persistent graph buffers).
# H100 80GB fits ~5 sm-music or ~3 medium combos before OOM. We LRU-evict to
# stay under that limit. Switching back to an evicted variant re-loads (~5-7s).
_inference_cache: dict[tuple[str, str, str, str, bool], SA3Inference] = {}
_inference_lru: list[tuple[str, str, str, str]] = []  # MRU at end
_INFERENCE_CACHE_MAX = 2


def _evict_inference(key):
    """Free the engines + caches held by a cached SA3Inference before dropping it."""
    inf = _inference_cache.pop(key, None)
    if inf is None:
        return
    # Free TRT engines (each runner has a .free() that releases the device buffer).
    for name, runner in list(inf.runners.items()):
        try:
            runner.free()
        except Exception:
            pass
    # Drop cached graphs (their persistent buffers will be freed when GC'd).
    inf._graphs.clear()
    inf._graph_lru.clear()
    del inf
    import gc
    gc.collect()
    try:
        canon.torch.cuda.empty_cache()
        canon.torch.cuda.synchronize()
    except Exception:
        pass


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
def build_ui(initial_dit: str, initial_decoder: str, *,
             share: bool, quiet: bool,
             default_T_lat: int, default_steps: int, default_seconds: float):
    import gradio as gr

    # Pre-warm the initial bundle so the first user click is fast.
    initial_variants = discover_variants(initial_dit)
    if not initial_variants:
        raise RuntimeError(f"no DiT engines found for {initial_dit} under {MODELS_ROOT}")
    initial_variant_path = str(initial_variants[0][1])
    initial_dec_variants = discover_decoder_variants(initial_decoder)
    if not initial_dec_variants:
        raise RuntimeError(f"no decoder engines found for {initial_decoder} under {MODELS_ROOT}")
    initial_dec_variant_path = str(initial_dec_variants[0][1])
    get_inference(initial_dit, initial_decoder,
                   initial_variant_path, initial_dec_variant_path,
                   default_T_lat, default_steps, default_seconds, quiet)

    DIT_OPTIONS = list(DEFAULT_DECODERS.keys())

    def on_dit_change(dit_name):
        """When user picks a new DiT, refresh the DiT-variant dropdown + suggest decoder."""
        variants = discover_variants(dit_name)
        suggested_decoder = DEFAULT_DECODERS.get(dit_name, "same-s")
        dec_variants = discover_decoder_variants(suggested_decoder)
        var_update = (gr.update(choices=[(lbl, str(p)) for lbl, p in variants],
                                 value=str(variants[0][1]))
                       if variants else gr.update(choices=[], value=None))
        dec_var_update = (gr.update(choices=[(lbl, str(p)) for lbl, p in dec_variants],
                                     value=str(dec_variants[0][1]))
                          if dec_variants else gr.update(choices=[], value=None))
        return var_update, gr.update(value=suggested_decoder), dec_var_update

    def on_decoder_change(decoder_name):
        """When user picks a new decoder, refresh decoder-variant choices."""
        dec_variants = discover_decoder_variants(decoder_name)
        if not dec_variants:
            return gr.update(choices=[], value=None)
        return gr.update(choices=[(lbl, str(p)) for lbl, p in dec_variants],
                          value=str(dec_variants[0][1]))

    def generate(dit_name, decoder_name, variant_path, dec_variant_path,
                 prompt, seconds, steps, seed_text):
        if not prompt or not prompt.strip():
            return "", "", "", "<span style='color:#f88'>error: empty prompt</span>"
        try:
            seed = int(seed_text.strip()) if seed_text and seed_text.strip() else None
        except ValueError:
            return "", "", "", "<span style='color:#f88'>error: seed must be an integer</span>"

        # PT-eager dispatch (if the user picked the GT pseudo-variant).
        is_pt_eager = str(variant_path) == PT_EAGER_VARIANT
        load_ms = 0.0
        t0 = time.time()
        try:
            if is_pt_eager:
                if dit_name != "medium":
                    return "", "", "", ("<span style='color:#f88'>PT FP32 GT is currently "
                                          "only wired for medium DiT.</span>")
                from pt_inference import get_pt_inference
                inf = get_pt_inference()
                load_ms = (time.time() - t0) * 1000
            else:
                inf = get_inference(dit_name, decoder_name, variant_path, dec_variant_path,
                                     default_T_lat, default_steps, default_seconds, quiet)
                load_ms = (time.time() - t0) * 1000
        except Exception as e:
            return "", "", "", f"<span style='color:#f88'>load failed: {type(e).__name__}: {e}</span>"

        # Run.
        try:
            pcm, t = inf.generate(prompt.strip(), seconds=float(seconds),
                                   steps=int(steps), seed=seed)
        except NotImplementedError as e:
            return "", "", "", f"<span style='color:#f88'>not yet implemented: {e}</span>"
        except Exception as e:
            return "", "", "", f"<span style='color:#f88'>error: {type(e).__name__}: {e}</span>"

        # WAV (persist + base64 inline)
        out_path = OUTPUT_DIR / f"sa3-{uuid.uuid4().hex[:10]}.wav"
        _save_wav(pcm, out_path)
        _prune_old_outputs()
        wav_bytes = out_path.read_bytes()
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        audio_html = (
            f'<audio controls autoplay style="width:100%" '
            f'src="data:audio/wav;base64,{b64}"></audio>'
            f'<div style="font-size:0.85em; margin-top:4px; color:#888">'
            f'{len(wav_bytes)/1e6:.1f} MB · right-click or use player menu to download'
            f'</div>'
        )

        # Spectrogram (Underfit algorithm)
        try:
            t0 = time.time()
            spec_png = render_spectrogram_png(pcm, sample_rate=SAMPLE_RATE,
                                                width=1200, height=240)
            spec_ms = (time.time() - t0) * 1000
            spec_b64 = base64.b64encode(spec_png).decode("ascii")
            spec_html = (
                f'<img src="data:image/png;base64,{spec_b64}" '
                f'style="width:100%; image-rendering:pixelated; border:1px solid #333" '
                f'alt="spectrogram"/>'
                f'<div style="font-size:0.75em; color:#666; margin-top:2px">'
                f'underfit-style 3-band tinted stereo mel spec · '
                f'red=bass / green=mid / blue=high · L on top, R on bottom · '
                f'rendered in {spec_ms:.0f} ms</div>'
            )
        except Exception as e:
            spec_html = (f"<span style='color:#fa3'>spectrogram failed: "
                          f"{type(e).__name__}: {e}</span>")

        # Timing
        load_note = (f"engine-load {load_ms:.0f} ms ·&nbsp; "
                     if load_ms > 100 else "")
        build_note = (f"graph-build {t['graph_build_ms']:.0f} ms ·&nbsp; "
                      if t.get("graph_build_ms", 0) > 1 else "")
        backend_tag = (
            "<span style='background:#fae;color:#603;padding:1px 6px;border-radius:3px'>"
            "PT FP32 GT (vanilla eager)</span> &nbsp; "
            if is_pt_eager else ""
        )
        pt_breakdown = (
            f" &nbsp;<span style='color:#888'>(t5={t.get('t5_ms', 0):.0f} ms · "
            f"sample={t.get('sampling_ms', 0):.0f} ms · "
            f"decode={t.get('decode_ms', 0):.0f} ms)</span>"
            if is_pt_eager else ""
        )
        timing_html = (
            f"{backend_tag}{load_note}{build_note}"
            f"<b>Inference</b>: {t['inference_ms']:.1f} ms{pt_breakdown} ·&nbsp; "
            f"<b>{t['realtime']:.0f}× realtime</b> ·&nbsp; "
            f"<b>seed</b>: <code>{t['seed']}</code> ·&nbsp; "
            f"<b>T_lat</b>: {t['T_lat']} ·&nbsp; "
            f"<b>samples</b>: {t['samples']}"
        )
        if t["T_lat"] < 256:
            timing_html += (" ·&nbsp; <span style='color:#fa3'>warning: T_lat &lt; 256 "
                            "is below the DiT's trained range; output quality "
                            "is undefined.</span>")
        return audio_html, spec_html, timing_html, ""

    with gr.Blocks(title=f"SA3 TRT — debug") as demo:
        gr.Markdown(
            "# SA3 TRT — debug build\n"
            "Loaded engines auto-cache by (model, variant). First switch to a "
            "new combo is slow (~5-7s); subsequent uses replay in ~30-100 ms."
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    dit_dd = gr.Dropdown(label="DiT model", choices=DIT_OPTIONS,
                                          value=initial_dit, scale=1)
                    decoder_dd = gr.Dropdown(label="Decoder",
                                              choices=["same-s", "same-l"],
                                              value=initial_decoder, scale=1)
                with gr.Row():
                    variant_dd = gr.Dropdown(label="DiT variant",
                                              choices=[(lbl, str(p)) for lbl, p in initial_variants],
                                              value=initial_variant_path, scale=1)
                    dec_variant_dd = gr.Dropdown(label="Decoder variant",
                                                  choices=[(lbl, str(p)) for lbl, p in initial_dec_variants],
                                                  value=initial_dec_variant_path, scale=1)

                prompt = gr.Textbox(label="Prompt", lines=2,
                                     placeholder="e.g. 'Death Metal'")
                with gr.Row():
                    seconds = gr.Slider(label="Seconds", minimum=1, maximum=120,
                                         value=120, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16,
                                       value=8, step=1)
                seed = gr.Textbox(label="Seed (optional, blank = random)",
                                   max_lines=1, value="1")
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Accordion("Advanced (coming soon)", open=False):
                    gr.Markdown(
                        "These don't work yet — wiring through SA3Inference.generate "
                        "for CFG / negative prompt / init audio / inpaint coming next."
                    )
                    gr.Slider(label="CFG", minimum=1.0, maximum=10.0, value=1.0,
                              step=0.1, interactive=False)
                    gr.Textbox(label="Negative prompt", interactive=False)
                    gr.Slider(label="Init noise level (σmax)", minimum=0.1,
                              maximum=1.2, value=1.0, step=0.05, interactive=False)
                    gr.Audio(label="Init audio", type="filepath", interactive=False)
                    gr.Textbox(label="Inpaint range", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("**Audio**")
                output_audio = gr.HTML()
                gr.Markdown("**Spectrogram** (Underfit-style)")
                output_spec = gr.HTML()
                timing = gr.HTML()
                error_box = gr.HTML()

        # Wire pickers to refresh dependent dropdowns
        dit_dd.change(on_dit_change, inputs=[dit_dd],
                       outputs=[variant_dd, decoder_dd, dec_variant_dd])
        decoder_dd.change(on_decoder_change, inputs=[decoder_dd],
                           outputs=[dec_variant_dd])

        generate_btn.click(generate,
                            inputs=[dit_dd, decoder_dd, variant_dd, dec_variant_dd,
                                     prompt, seconds, steps, seed],
                            outputs=[output_audio, output_spec, timing, error_box])

        gr.Markdown(
            "<p style='color:#888; font-size:0.85em'>"
            "WAVs saved under <code>output/gradio/</code> "
            f"(rotates after {KEEP_RECENT_N} files). "
            "Variant dropdown auto-scans <code>models/&lt;arch&gt;/&lt;dit&gt;/dit_*.trt</code>."
            "</p>"
        )

    demo.queue(max_size=16).launch(share=share, server_name="0.0.0.0",
                                     theme=gr.themes.Soft(),
                                     prevent_thread_lock=False,
                                     show_error=True)


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
