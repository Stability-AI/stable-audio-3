"""SA3 text-to-audio inference on CPU via torch-free C++ AMX engines.

The cpu-amx sibling of optimized/mlx/scripts/sa3_mlx.py and optimized/tensorRT.
Same CLI, same modes; the compute runs on Xeon AMX C++ engines instead of MLX:

    prompt -> T5Gemma (C++ AMX) -> numpy conditioner
           -> DiT pingpong (C++ AMX int8 | bf16, MEDIUM) -> SAME-S/SAME-L decoder (C++ AMX)
           -> WAV

Modes (identical flags to the MLX / TensorRT releases):
    text-to-audio    --prompt P
    audio-to-audio   --prompt P --init-audio IN.wav [--init-noise-level sigma]
    inpainting       --prompt P --init-audio IN.wav --inpaint-range START,END
    negative CFG     --prompt P --cfg N [--negative-prompt P_NEG] [--apg S]

cpu-amx specifics vs MLX:
    * DiT is MEDIUM only, in int8 (default) or bf16 (--dit-precision). --dit sm-music /
      sm-sfx are rejected with a pointer to optimized/tflite or optimized/mlx.
    * MLX's per-model --dit-dtype splits into --dit-precision {int8,bf16} and
      --decoder-precision {bf16,int8}. The int8 DiT is pinned to 1 thread (its .so
      heap-races higher); the bf16 DiT (near-lossless, fp32 RoPE/RMSNorm islands) runs
      at --threads. T5Gemma is bf16 regardless.
    * audio-to-audio / inpainting init-encode is the torch-free C++ AMX SAME encoder;
      the whole stack is 100% C++/numpy.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from shutil import which

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import pipeline as P              # noqa: E402  (vendored numpy pipeline)
import backends as B             # noqa: E402  (C++ AMX engine loaders)

SAMPLE_RATE = P.SAMPLE_RATE
SAMPLES_PER_LATENT = P.SAMPLES_PER_LATENT

DIT_CHOICES = ["medium"]                 # cpu-amx: MEDIUM int8 core only
DIT_UNAVAILABLE = ("sm-music", "sm-sfx")  # accepted for a clear rejection message
DECODER_CHOICES = ["same-s", "same-l"]
DEFAULT_DECODER = "same-l"               # medium's native codec (mirrors MLX)


# ─── display helpers (ANSI colour when stdout is a TTY) — copied from sa3_mlx ─
_USE_COLOR = sys.stdout.isatty()
_RULE_W = 64

def _c(code, s):   return f"\x1b[{code}m{s}\x1b[0m" if _USE_COLOR else s
def bold(s):       return _c("1", s)
def dim(s):        return _c("2", s)
def cyan(s):       return _c("36", s)
def yellow(s):     return _c("33", s)
def green(s):      return _c("32", s)
def magenta(s):    return _c("35", s)

def rule(char="━", color=cyan):
    print(color(char * _RULE_W))

def banner(title):
    rule(); print(f"  {bold(title)}"); rule()

def stage(idx_total, label, ms=None):
    head = f"  {cyan(idx_total)} {bold(label)}"
    if ms is None:
        print(head); return
    visible = len(f"  {idx_total} {label}")
    fill = max(2, _RULE_W - visible - 9)
    print(f"{head} {dim('·' * fill)} {yellow(f'{ms:>5.0f} ms')}")

def sub(text):
    print(f"        {dim(text)}")

def _peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB on Linux
    except Exception:
        return 0.0


# ─── arrow-key picker (posix termios; numeric fallback off-TTY) — from sa3_mlx ─
def _arrow_pick(prompt, options, default=None):
    if not sys.stdin.isatty():
        print(prompt)
        for i, o in enumerate(options):
            print(f"  {'*' if o == default else ' '} [{i}] {o}")
        s = input(f"Choose [0-{len(options)-1}] (Enter for default): ").strip()
        if s == "":
            return default or options[0]
        if s.isdigit() and 0 <= int(s) < len(options):
            return options[int(s)]
        return s if s in options else (default or options[0])
    import termios, tty
    idx = options.index(default) if default in options else 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt)
    for _ in options:
        print()
    try:
        tty.setcbreak(fd)
        while True:
            sys.stdout.write(f"\x1b[{len(options)}A")
            for i, o in enumerate(options):
                sys.stdout.write(f"\x1b[2K\x1b[36m▶ {o}\x1b[0m\n" if i == idx
                                 else f"\x1b[2K  {o}\n")
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":   idx = (idx - 1) % len(options)
                elif seq == "[B": idx = (idx + 1) % len(options)
            elif ch in ("\n", "\r"):
                return options[idx]
            elif ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class _HelpfulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"\nerror: {message}\n\n")
        self.print_help(sys.stderr)
        sys.exit(2)
    def print_help(self, file=None):
        super().print_help(file)
        try:
            from examples import print_example_commands
            print_example_commands()
        except Exception:
            pass


def _play(path):
    for player in ("ffplay", "aplay", "paplay", "play", "afplay"):
        if which(player):
            args = [player, "-autoexit", "-nodisp", path] if player == "ffplay" else [player, path]
            try:
                print(f"  {bold('▶ playing')}   {path}   {dim('(Ctrl-C to stop)')}")
                subprocess.run(args, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except KeyboardInterrupt:
                print()
            return
    print(f"  {dim('(--play: no audio player found — install ffmpeg/alsa-utils to hear it)')}")


def main():
    """Top-level wrapper: run, then ALWAYS exit via os._exit so the DiT .so's
    teardown double-free can never fire (it would mask the real error as a
    confusing SIGABRT). Real errors are printed here and exit non-zero."""
    code = 0
    try:
        _run()
    except SystemExit as e:
        c = e.code
        if isinstance(c, str):      # sys.exit("message")
            print(c, file=sys.stderr); code = 1
        else:
            code = int(c) if c is not None else 0
    except KeyboardInterrupt:
        code = 130
    except Exception:
        import traceback
        traceback.print_exc(); code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


def _run():
    ap = _HelpfulParser(
        description="SA3 text-to-audio (+ audio-to-audio + inpainting) on CPU via C++ AMX engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes\n"
            "  text-to-audio    --prompt P\n"
            "  audio-to-audio   --prompt P --init-audio IN.wav [--init-noise-level σ]\n"
            "  inpainting       --prompt P --init-audio IN.wav --inpaint-range START,END\n"
            "  negative CFG     --prompt P --cfg N --negative-prompt P_NEG\n"
        ),
    )
    # Inputs
    ap.add_argument("--prompt", default=None,
                    help="Text prompt. Empty string is valid (unconditional). "
                         "If omitted, asked interactively via stdin.")
    ap.add_argument("--negative-prompt", default=None,
                    help="Negative prompt for CFG's uncond branch. No effect at --cfg=1.0. "
                         "When unset and --cfg≠1.0, the uncond branch uses the learned "
                         "padding embedding (all-zero prompt).")
    ap.add_argument("--init-audio", default=None,
                    help="WAV (44.1 kHz stereo/mono) starting point. With --init-noise-level "
                         "= audio-to-audio; with --inpaint-range = inpainting. Init-encode "
                         "uses the torch-free C++ AMX SAME-{S,L} encoder (matched to --decoder). "
                         "Trimmed/padded to --seconds.")
    ap.add_argument("--inpaint-range", default=None,
                    help="Inpaint span 'START,END' in seconds (needs --init-audio). The span is "
                         "regenerated; the rest is kept bit-exact (per-step paste-back).")
    # Models
    ap.add_argument("--dit", choices=DIT_CHOICES + list(DIT_UNAVAILABLE), default=None,
                    help="DiT model. cpu-amx ships MEDIUM only (the int8 C++ core). "
                         "sm-music / sm-sfx are not available here — use optimized/tflite or mlx.")
    ap.add_argument("--decoder", choices=DECODER_CHOICES, default=None,
                    help="Audio decoder. 'same-l' = native 426M medium codec (default). "
                         "'same-s' = distilled 50M (faster, shares the medium latent space).")
    ap.add_argument("--dit-precision", choices=["int8", "bf16"], default="int8",
                    help="DiT C++ engine precision. int8 (default, ~40 dB) or bf16 (near-lossless "
                         "fp32 RoPE/RMSNorm islands, ~59/54 dB @L1292/L4096, ~1.24× the int8 latency). "
                         "Both run at --threads. The cpu-amx analogue of MLX's per-model dtype.")
    ap.add_argument("--decoder-precision", choices=["bf16", "int8"], default="bf16",
                    help="Decoder C++ engine precision. bf16 (default, best fidelity) or int8 "
                         "(SmoothQuant+GPTQ fused w8a8, smaller/faster). T5Gemma is bf16 regardless.")
    ap.add_argument("--threads", type=int, default=16,
                    help="Thread count for T5Gemma + the decoder + the DiT (default 16). Both DiT "
                         "precisions run multi-threaded — at 1 thread the DiT is ~7.6× slower.")
    ap.add_argument("--lora", action="append", nargs="+", default=None, metavar="ADAPTER",
                    help="(not supported in cpu-amx — accepted and ignored with a note; the int8 "
                         "C++ DiT core has no runtime LoRA merge. Use optimized/mlx for LoRA.)")
    ap.add_argument("--lora-strength", type=float, default=1.0, help="(ignored in cpu-amx)")
    # Sampling
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="Output length in seconds. T_lat = ceil(seconds*44100/4096) (decoder-"
                         "independent). Final WAV trimmed to exactly --seconds.")
    ap.add_argument("--steps", type=int, default=8,
                    help="Pingpong sampling steps. Minimum 1 (single forward). The rf_denoiser is "
                         "distilled for 8 (default); >8 gives diminishing returns.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed (any int). If omitted, chosen randomly and printed at the end.")
    ap.add_argument("--init-noise-level", type=float, default=1.0,
                    help="σmax — the schedule's starting noise level. With --init-audio: 0.4–0.8 "
                         "typical for variation, 1.0 = full regeneration. Min 0.01.")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="Classifier-Free Guidance scale. 1.0 = off (single forward). >1 pushes "
                         "toward the prompt; [0,1) toward the uncond/negative branch. Any value "
                         "≠1.0 costs ~2× per step (sequential cond + uncond forward).")
    ap.add_argument("--apg", type=float, default=1.0,
                    help="Adaptive Projected Guidance [0..1], only when --cfg≠1.0. 1.0 = full APG "
                         "(project cond−uncond orthogonal to cond_denoised); 0.0 = vanilla CFG.")
    # Runtime / output
    ap.add_argument("--free-models", action=argparse.BooleanOptionalAction, default=True,
                    help="Free each model after its last use to lower peak RAM (default on).")
    ap.add_argument("--out", "-o", default=None,
                    help="Output WAV path. Relative paths land in output/; absolute as-is. "
                         "16-bit PCM stereo @ 44.1 kHz, trimmed to --seconds. Auto-named if omitted.")
    ap.add_argument("--play", action="store_true",
                    help="After writing, play the WAV (ffplay/aplay/paplay/afplay if present).")
    args = ap.parse_args()

    if args.steps < 1:
        ap.error(f"--steps must be ≥ 1 (got {args.steps})")

    # DiT selection — MEDIUM only.
    if args.dit in DIT_UNAVAILABLE:
        sys.exit(f"error: --dit {args.dit} is not available in cpu-amx (MEDIUM int8 C++ core only).\n"
                 f"       Use optimized/tflite or optimized/mlx for sm-music / sm-sfx.")
    if args.dit is None:
        # Only one DiT, so pick the decoder interactively (matches MLX's picker feel).
        args.dit = "medium"
    if args.decoder is None:
        if sys.stdin.isatty() and sys.stdout.isatty() and args.prompt is not None:
            args.decoder = _arrow_pick("Choose audio decoder:", DECODER_CHOICES, default=DEFAULT_DECODER)
            print(f"  → {args.decoder}")
        else:
            args.decoder = DEFAULT_DECODER
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    if args.prompt is None:
        args.prompt = input("Prompt: ").strip()
    if args.lora:
        print(dim("  note: --lora is not supported in cpu-amx (int8 C++ DiT core has no runtime "
                  "LoRA merge) — ignoring. Use optimized/mlx for LoRA."))

    # Output path
    if args.out is None:
        import re
        slug = re.sub(r'[^a-z0-9]+', '_', args.prompt.lower()).strip('_')[:48]
        args.out = f"{slug}_{args.seed}.wav" if slug else f"out_{args.seed}.wav"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = SCRIPTS.parent / "output" / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out = str(out_path)

    # T_lat: natural ceil (decoder-independent), matches MLX / TRT.
    T_lat = max(1, math.ceil(args.seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    target_dur = T_lat * SAMPLES_PER_LATENT / SAMPLE_RATE

    # Inpaint validation + latent range
    inpaint_range = None
    if args.inpaint_range is not None:
        if args.init_audio is None:
            sys.exit("error: --inpaint-range requires --init-audio")
        try:
            s_str, e_str = args.inpaint_range.split(",")
            inp_start_sec, inp_end_sec = float(s_str), float(e_str)
        except ValueError:
            sys.exit(f"error: --inpaint-range must be 'START,END' seconds; got {args.inpaint_range!r}")
        if not (0 <= inp_start_sec < inp_end_sec <= args.seconds):
            sys.exit(f"error: invalid inpaint range {inp_start_sec}-{inp_end_sec}s "
                     f"(need 0 ≤ start < end ≤ {args.seconds}s)")
        s0 = max(0, int(round(inp_start_sec * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(T_lat, int(round(inp_end_sec * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_range = (s0, s1)

    sigma_max = float(args.init_noise_level)
    mode = ("inpaint" if inpaint_range else
            "audio-to-audio" if args.init_audio else "text-to-audio")
    MIN_SIGMA = 0.01
    if sigma_max < MIN_SIGMA:
        sys.exit(f"error: --init-noise-level={sigma_max} too low (min {MIN_SIGMA}); "
                 f"the model is undefined at t≈0.")

    t_wall = time.time()
    print()
    banner(f"SA3 → CPU-AMX  {mode}")
    k = lambda s: dim(f"{s:>12}")
    v = lambda s, w=10: f"{s:<{w}}"
    print(f"  {k('prompt')}  {bold(repr(args.prompt))}")
    if args.negative_prompt:
        suffix = "" if args.cfg != 1.0 else dim("  (ignored: --cfg=1.0)")
        print(f"  {k('neg prompt')}  {bold(repr(args.negative_prompt))}{suffix}")
    line = f"  {k('dit')}  {magenta(v(args.dit))}   {k('decoder')}  {magenta(v(args.decoder))}"
    if args.init_audio:
        line += f"   {k('encoder')}  {magenta(v(f'C++ {args.decoder}'))}"
    print(line)
    if args.init_audio:
        print(f"  {k('init audio')}  {bold(args.init_audio)}")
        if inpaint_range:
            print(f"  {k('inpaint')}  {bold(f'{inp_start_sec:.2f}s..{inp_end_sec:.2f}s')} "
                  f"{dim(f'(latent {inpaint_range[0]}..{inpaint_range[1]} of {T_lat})')}")
    print(f"  {k('σmax')}  {bold(f'{sigma_max:.2f}')}")
    print(f"  {k('seconds')}  {v(f'{args.seconds}s')}   {k('steps')}  {v(args.steps)}   {k('seed')}  {args.seed}")
    cfg_label = f"{args.cfg}" + (f" (apg={args.apg})" if args.cfg != 1.0 else "")
    print(f"  {k('dit prec')}  {v(args.dit_precision)}   {k('dec prec')}  {v(args.decoder_precision)}   "
          f"{k('threads')}  {v(args.threads)}   {k('cfg')}  {cfg_label}")
    print(f"  {k('T_lat')}  {T_lat} {dim(f'({target_dur:.2f}s → trimmed to {args.seconds}s)')}")
    print()

    # ── 1. T5Gemma encode ──
    t0 = time.time()
    tok = P.Tokenizer()
    ids, mask = tok(args.prompt)
    t5 = B.load_t5gemma(threads=args.threads)
    last_hidden = t5(ids.astype(np.int32), mask.astype(np.int32))          # (1,256,768)
    stage("[1/5]", "T5Gemma encode", (time.time() - t0) * 1000)
    sub(f"last_hidden {last_hidden.shape}  nnz(mask)={int(mask.sum())}")

    # ── 2. Conditioning ──
    t0 = time.time()
    cond = P.Conditioner()
    cross, gcond = cond.build(last_hidden, mask, args.seconds)             # (1,257,768),(1,768)
    null_cross = None
    if args.cfg != 1.0:
        if args.negative_prompt:
            n_ids, n_mask = tok(args.negative_prompt)
            neg_hidden = t5(n_ids.astype(np.int32), n_mask.astype(np.int32))
            null_cross, _ = cond.build(neg_hidden, n_mask, args.seconds)
        else:
            # learned-padding uncond: conditioner on all-zero hidden+mask
            null_cross, _ = cond.build(np.zeros((1, 256, 768), np.float32),
                                       np.zeros((1, 256), np.int32), args.seconds)
    stage("[2/5]", "Conditioning", (time.time() - t0) * 1000)
    sub(f"cross {cross.shape}  global {gcond.shape}"
        + (f"  null_cross ({'neg prompt' if args.negative_prompt else 'learned padding'})"
           if null_cross is not None else ""))
    if args.free_models:
        del t5   # T5Gemma no longer needed

    # ── 3a. (a2a / inpaint) encode init audio → init_latents (C++ AMX SAME-{S,L} encoder) ──
    init_latents = None
    if args.init_audio:
        stage("[3a]", f"Encode init audio → latents (C++ AMX {args.decoder} encoder)")
        t0 = time.time()
        enc = B.load_encoder(args.decoder, precision=args.decoder_precision, threads=args.threads)
        audio_in = P.read_wav(args.init_audio)                            # (2, N)
        init_latents = enc.encode(audio_in, T_lat)                        # (1,256,T_lat)
        sub(f"device={enc.device}  {(time.time()-t0)*1000:.0f} ms  latents {init_latents.shape}")
        if args.free_models:
            del enc

    # ── 3b. DiT load + pingpong sample ──
    stage("[3/5]", f"DiT — load + sample ({args.dit_precision}, {args.steps} steps, σmax={sigma_max:.2f})")
    t0 = time.time()
    dit = B.load_dit(precision=args.dit_precision, threads=args.threads)
    sub(f"load {time.time()-t0:.1f}s  (medium {args.dit_precision} C++ core, {args.threads} thread{'s' if args.threads != 1 else ''})")

    sigmas = P.build_pingpong_schedule(args.steps, sigma_max=sigma_max)
    sub("schedule  " + " · ".join(f"{float(x):.3f}" for x in sigmas))

    x0, step_noise = P.make_noise(T_lat, args.steps, args.seed)
    if init_latents is not None and inpaint_range is None:
        x0 = init_latents * (1.0 - sigma_max) + x0 * sigma_max            # a2a init mix
        sub(f"init: latent * {1-sigma_max:.2f} + noise * {sigma_max:.2f}")

    paste_back = None
    if inpaint_range is not None:
        s0, s1 = inpaint_range
        keep = np.ones((1, 1, T_lat), np.float32); keep[:, :, s0:s1] = 0.0
        paste_back = (init_latents.astype(np.float32), keep)
        sub(f"inpaint mask {s0}..{s1} of {T_lat} ({(s1-s0)/max(T_lat,1)*100:.0f}% regenerated); "
            f"paste-back keeps the rest bit-exact")

    def _on_step(i, n):
        if not _USE_COLOR:
            return
        bar_w = 20; filled = int(round(bar_w * i / n))
        bar = cyan("█" * filled) + dim("·" * (bar_w - filled))
        sys.stdout.write(f"\r\x1b[K        {dim('sampling')} {bar} {bold(f'step {i}/{n}')}")
        sys.stdout.flush()

    t0 = time.time()
    if args.cfg == 1.0:
        latents = P.sample(dit, x0, step_noise, sigmas, cross, gcond,
                           on_step=_on_step, paste_back=paste_back)
    else:
        latents = P.sample_cfg(dit, x0, step_noise, sigmas, cross, gcond, null_cross,
                               cfg_scale=args.cfg, apg=args.apg, batched=False,
                               on_step=_on_step, paste_back=paste_back)
    sample_ms = (time.time() - t0) * 1000
    if _USE_COLOR:
        sys.stdout.write("\r\x1b[K")
    if not np.isfinite(latents).all():
        sys.exit("error: DiT produced non-finite latents (try a different seed or σmax)")
    sub(f"sample {sample_ms:.0f} ms  ({sample_ms/max(args.steps,1):.0f} ms/step)  "
        f"latent {latents.shape}")
    if args.free_models:
        del dit   # the DiT crashes on teardown anyway; os._exit(0) skips it

    # ── 4. Decode → audio ──
    stage("[4/5]", f"Decoder ({args.decoder}, {args.decoder_precision} C++ AMX)")
    t0 = time.time()
    dec = B.load_decoder(args.decoder, args.decoder_precision, threads=args.threads)
    audio_np = dec.decode(latents)                                        # (2, T_lat*4096)
    stage("[4/5]", "decode", (time.time() - t0) * 1000)
    sub(f"audio {audio_np.shape}")

    # ── 5. Trim + write WAV ──
    t0 = time.time()
    requested = int(round(args.seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > requested:
        audio_np = audio_np[..., :requested]
    P.save_wav(args.out, audio_np)
    peak = float(np.abs(audio_np).max()); rms = float(np.sqrt((audio_np ** 2).mean()))
    stage("[5/5]", "write WAV", (time.time() - t0) * 1000)
    sub(f"audio {audio_np.shape}  peak {peak:.3f}  rms {rms:.3f}")

    total = time.time() - t_wall
    audio_dur = audio_np.shape[-1] / SAMPLE_RATE
    print()
    rule()
    print(f"  {bold(green('done'))}   {bold(f'{total:.2f}s')} wall  →  {audio_dur:.1f}s audio  →  "
          f"{bold(yellow(f'{audio_dur/max(total,1e-9):.2f}× realtime'))}   "
          f"{dim(f'peak RSS {_peak_rss_mb()/1024:.2f} GB')}   {dim(f'seed {args.seed}')}")
    abs_out = os.path.abspath(args.out)
    print(f"  {bold(green('▸ saved'))}  {bold(abs_out)}")
    rule()

    if args.play:
        _play(args.out)
    # main() exits via os._exit — the DiT .so double-frees at teardown, so we
    # must never let the interpreter run global destructors.


if __name__ == "__main__":
    main()
