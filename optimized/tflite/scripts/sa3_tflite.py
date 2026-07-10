"""SA3 text-to-audio (+ audio-to-audio + inpainting + CFG) on CPU via the BAKED-I/O
varlen TFLite / LiteRT models — the portable-CPU sibling of the MLX and TensorRT
releases. CLI at feature parity with sa3_mlx.py / sa3_trt.py, sharing their flag names.

Baked I/O (ONNX / TensorRT convention — conditioner + patch/unpatch are IN-GRAPH):
  DiT (6-input): x[1,256,L], t[1], t5_hidden[1,256,768], t5_mask[1,256],
      seconds_total[1], local_add_cond[1,257,L]  ->  velocity[1,256,L]
      (feed RAW T5 outputs; prompt-padding + seconds-embed happen in-graph)
  Encoder (audio-in):  audio[1,2,N]     -> latents[1,256,N/4096]   (patch-encode baked)
  Decoder (audio-out): latents[1,256,L] -> audio[1,2,L*4096]       (unpatch baked)
  T5Gemma: t5gemma/encoder_fp16.tflite (fixed 256 text tokens; tokenizer bundled)

Modes (identical flags to sa3_mlx / sa3_trt):
  text-to-audio    --prompt P
  audio-to-audio   --prompt P --init-audio IN.wav [--init-noise-level σ]
  inpainting       --prompt P --init-audio IN.wav --inpaint-range START,END
  negative CFG     --prompt P --cfg N [--negative-prompt P_NEG] [--apg A]

CFG uses the TRT/ONNX baked-conditioner convention: the uncond branch feeds the negative
prompt's T5 output (or an all-zero hidden+mask, which the in-graph conditioner turns into
learned padding embeddings) — it does NOT zero cross_attn like MLX (our DiT bakes the
conditioner, so there's no raw cross_attn). CFG runs cond+uncond as ONE batch=2 invoke on the
variable-batch DiT (--cfg-batched, default; ~7-29%% faster on Apple-Silicon AMX) or as a
sequential batch=1 dual-pass (--no-cfg-batched). Bit-identical at fp32/w8a32; ~80 dB at
w16a32; w8a8-dyn diverges by design (activation scales shared across the batch).

All model files auto-download from HuggingFace (stabilityai/stable-audio-3-optimized)
on first use and symlink into models/tflite/ from the HF cache. See scripts/weights.py.
"""
from __future__ import annotations
import argparse, math, os, random, re, subprocess, sys, termios, time, tty, wave
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent   # project root (scripts/ is one level down)
sys.path.insert(0, str(REPO))                    # so `from models.defs.* import` resolves
sys.path.insert(0, str(REPO / "scripts"))        # so `from weights import *` resolves

from models.defs import tflite_pipeline as P    # Tokenizer, T5GemmaTFLite, build_pingpong_schedule, make_noise, sample, save_wav
from weights import ensure_local, is_present, PRECISIONS, dit_rel, dec_rel, enc_rel

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
COND_TOKENS = 256
COND_DIM = 768
MIN_SIGMA = 0.01   # rf_denoiser is undefined at t≈0 → NaN below this

# ─── Model manifest (local rel paths; resolved via weights.ensure_local at load) ───
# --precision picks the DiT + decoder variant (same flag as the TensorRT release's
# --precision; values here use the wXaY convention — weights/activations bit-widths,
# "16" = fp16, there is no int16). Encoders + T5Gemma have one precision, like TRT.
#   fp32      reference models (default — on CPU fp16 is SLOWER than fp32, so unlike
#             TRT the fastest-and-accurate choice is fp32)
#   w16a32    fp16 weights / fp32 activations — half size, ≈lossless, 1.5-3× slower
#             on an M4 Pro (legacy name: fp16mixed; ≈ TRT's fp16mixed in spirit,
#             storage-only here)
#   w8a32     GPTQ weight-only int8 — ¼ size at fp32 speed (legacy: woint8)
#   w8a8-dyn  GPTQ dynamic int8 — fastest (~1.2-1.3×), lowest quality (legacy: dynint8)
# PRECISIONS + the path builders live in weights.py (single source of truth with
# the download manifest — a precision the CLI accepts is guaranteed downloadable).
DIT_REL = {d: dit_rel(d) for d in ("sm-music", "sm-sfx", "medium")}   # fp32 (picker)
DEC_REL = {d: dec_rel(d) for d in ("same-s", "same-l")}
ENC_REL = {d: enc_rel(d) for d in ("same-s", "same-l")}
T5_REL = "models/tflite/t5gemma/encoder_fp16.tflite"
DEFAULT_DECODER = {"sm-music": "same-s", "sm-sfx": "same-s", "medium": "same-l"}

# SAME-L dense SWA mask is O(S^2): chunk long decodes. SAME-S has a narrow field: whole.
SAMEL_CHUNK = 64     # latent tokens/window — throughput optimum (6.5x RT) vs the O(S^2) dense SWA mask
SAMEL_OVERLAP = 8    # symmetric latent-token context each interior side (SAME-L needs >=8)


def _free():
    import gc; gc.collect()


# ─── ANSI display (match sa3_mlx.py style) ─────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
_RULE_W = 64
def _c(code, s): return f"\x1b[{code}m{s}\x1b[0m" if _USE_COLOR else s
def bold(s):    return _c("1", s)
def dim(s):     return _c("2", s)
def cyan(s):    return _c("36", s)
def yellow(s):  return _c("33", s)
def green(s):   return _c("32", s)
def magenta(s): return _c("35", s)
def red(s):     return _c("31", s)
def rule(ch="━"): print(cyan(ch * _RULE_W))
def banner(t):
    rule(); print(f"  {bold(t)}"); rule()
def stage(idx, label, ms=None):
    head = f"  {cyan(idx)} {bold(label)}"
    if ms is None:
        print(head, flush=True); return
    visible = len(f"  {idx} {label}")
    dots = dim("·" * max(2, _RULE_W - visible - 9))
    print(f"{head} {dots} {yellow(f'{ms:>5.0f} ms')}", flush=True)
def sub(t): print(f"        {dim(t)}", flush=True)


# ─── Interactive arrow-key picker (from sa3_mlx.py) ────────────────────────
def _arrow_pick(prompt: str, options: list[str], default: str | None = None) -> str:
    """Tiny arrow-key picker — no external deps, posix termios only.

    Up/Down to move, Enter to select, Ctrl-C to abort. Falls back to a
    numeric prompt when stdin isn't a TTY (piped input, CI, etc.).
    """
    if not sys.stdin.isatty():
        print(prompt)
        for i, o in enumerate(options):
            mark = "*" if o == default else " "
            print(f"  {mark} [{i}] {o}")
        s = input(f"Choose [0-{len(options)-1}] (Enter for default): ").strip()
        if s == "":
            return default or options[0]
        if s.isdigit() and 0 <= int(s) < len(options):
            return options[int(s)]
        return s if s in options else (default or options[0])

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
                if i == idx:
                    sys.stdout.write(f"\x1b[2K\x1b[36m▶ {o}\x1b[0m\n")
                else:
                    sys.stdout.write(f"\x1b[2K  {o}\n")
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    idx = (idx - 1) % len(options)
                elif seq == "[B":
                    idx = (idx + 1) % len(options)
            elif ch in ("\n", "\r"):
                return options[idx]
            elif ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_user_if_missing(args):
    """Interactive arrow-key selection when --dit / --decoder aren't supplied."""
    if args.dit is None:
        args.dit = _arrow_pick("Choose DiT model:", list(DIT_REL.keys()), default="sm-music")
        print(f"  → {args.dit}")
    if args.decoder is None:
        suggested = DEFAULT_DECODER[args.dit]
        args.decoder = _arrow_pick("Choose audio decoder:", list(DEC_REL.keys()), default=suggested)
        print(f"  → {args.decoder}")
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    return args


def _preflight_download(args, dec) -> None:
    """Resolve every model file this run needs and download any that are missing —
    BEFORE the banner prints and BEFORE the wall-clock starts. Network time then
    isn't charged against '×realtime' and the user sees download progress as a
    clearly separate setup step."""
    needed = [T5_REL, dit_rel(args.dit, args.dit_precision), dec_rel(dec, args.decoder_precision)]
    if args.init_audio:
        needed.append(enc_rel(dec, args.encoder_precision))
    missing = [p for p in needed if not is_present(p)]
    if not missing:
        return
    print(f"  Fetching {len(missing)} missing model file(s) before starting:")
    for rel in missing:
        ensure_local(rel)
    print()


# ─── WAV read (16-bit PCM fast path + ffmpeg fallback) ─────────────────────
def read_wav(path: str) -> np.ndarray:
    """Return (2, T) float32 in [-1, 1]. 16-bit/44.1k native; else via ffmpeg. Mono→stereo."""
    try:
        with wave.open(path, "rb") as w:
            nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if sr == SAMPLE_RATE and sw == 2:
                raw = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32767.0
                if nch == 1:
                    return np.stack([raw, raw], 0)
                return raw.reshape(-1, nch).T[:2]
    except wave.Error:
        pass
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
                            "-ar", str(SAMPLE_RATE), "-ac", "2", "-"],
                           capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(f"{path}: unsupported WAV format. Install ffmpeg for 24/32-bit or "
                           f"non-44.1kHz input: brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{path}: ffmpeg failed — {e.stderr.decode().strip()}")
    raw = np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32767.0
    return raw.reshape(-1, 2).T


# ─── Baked DiT backend (6-input; batched OR sequential CFG + inpaint local_add_cond) ───
def _apply_cfg(x, t, v_cond, v_uncond, cfg, apg):
    """Combine cond/uncond velocities in denoised space (RF), optional APG. fp32.
    Matches sa3_trt_core.model_fn. Returns cfg_v so `denoised = x - t*v` is the guided one."""
    sigma = np.float32(t)
    xf = x.astype(np.float32)
    cond_d   = xf - v_cond.astype(np.float32)   * sigma
    uncond_d = xf - v_uncond.astype(np.float32) * sigma
    diff = cond_d - uncond_d
    if apg <= 0.0:
        cfg_diff = diff
    else:
        norm = np.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
        unit = cond_d / np.maximum(norm, 1e-8)
        parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
        diff_orth = diff - parallel
        cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
    cfg_d = cond_d + (cfg - 1.0) * cfg_diff
    return ((xf - cfg_d) / sigma).astype(np.float32)


class BakedDiT:
    """model_fn(x,t,cross,gcond)->v compatible with P.sample. cross/gcond are IGNORED
    (conditioning is baked in-graph, driven by the raw T5 outputs held here).

    cfg==1.0  -> one batch=1 forward with the (cond) T5 output.
    cfg!=1.0  -> CFG combining a cond and an uncond (null_hidden/null_mask) velocity in
                 denoised space with optional APG. Two backends (bit-identical output at
                 fp32/w8a32; ~80 dB at w16a32; w8a8-dyn diverges — batch-shared activation
                 scales; auto-falls-back to batch=1 for cfg==1.0):
      batched=True  (default): ONE batch=2 invoke/step — cond=row0, uncond=row1 — on the
                    canonical variable-batch DiT (its batch axis is dynamic). ~7-29%% faster
                    on Apple-Silicon (the AMX matrix unit amortizes weight loads across the 2 rows).
      batched=False: SEQUENTIAL dual-pass — two batch=1 invokes/step, like TensorRT (whose
                    engine is static-batch=1). Use if a backend/delegate dislikes batch=2.
    Returns cfg_v so P.sample's `denoised = x - t*v` yields the guided denoised."""
    def __init__(self, path, L, t5_hidden, t5_mask, seconds, threads=8,
                 cfg=1.0, apg=1.0, null_hidden=None, null_mask=None, local_add_cond=None,
                 batched=True):
        from ai_edge_litert import interpreter as tfl
        self.L = L
        self.cfg = float(cfg); self.apg = float(apg)
        self.n_fwd = 0
        # Batched CFG only applies when there's an uncond branch to co-batch (cfg != 1.0).
        self.batched = bool(batched) and self.cfg != 1.0
        B = 2 if self.batched else 1
        self.B = B

        self.it = tfl.Interpreter(model_path=str(path), num_threads=threads)
        det = self.it.get_input_details()
        def pick(pred): return [d for d in det if pred(d)]
        self.i_x   = pick(lambda d: len(d["shape"]) == 3 and d["shape"][1] == 256 and d["shape"][2] != COND_DIM)[0]
        self.i_lac = pick(lambda d: len(d["shape"]) == 3 and d["shape"][1] == 257)[0]
        self.i_t5h = pick(lambda d: len(d["shape"]) == 3 and d["shape"][2] == COND_DIM)[0]
        self.i_t5m = pick(lambda d: len(d["shape"]) == 2)[0]
        scalars = sorted(pick(lambda d: len(d["shape"]) == 1), key=lambda d: d["name"])
        self.i_t, self.i_sec = scalars[0], scalars[1]   # args_1=t < args_4=seconds by name
        self.out = self.it.get_output_details()[0]["index"]
        # Resize batch axis to B (the canonical DiT is variable-batch) + length axis to L.
        self.it.resize_tensor_input(self.i_x["index"],   [B, 256, L], strict=False)
        self.it.resize_tensor_input(self.i_lac["index"], [B, 257, L], strict=False)
        self.it.resize_tensor_input(self.i_t5h["index"], [B, COND_TOKENS, COND_DIM], strict=False)
        self.it.resize_tensor_input(self.i_t5m["index"], [B, COND_TOKENS], strict=False)
        self.it.resize_tensor_input(self.i_t["index"],   [B], strict=False)
        self.it.resize_tensor_input(self.i_sec["index"], [B], strict=False)
        self.it.allocate_tensors()

        t5h = t5_hidden.astype(np.float32)
        t5m = t5_mask.astype(np.float32)
        sec = np.array([np.float32(seconds)], np.float32)
        lac = (np.zeros((1, 257, L), np.float32) if local_add_cond is None
               else local_add_cond.astype(np.float32))
        self.null_h = None if null_hidden is None else null_hidden.astype(np.float32)
        self.null_m = None if null_mask is None else null_mask.astype(np.float32)
        if self.batched:
            # row0 = cond, row1 = uncond. t5_hidden/mask differ per row; seconds + lac are
            # shared → tiled. All four are constant across steps → set resident once.
            self.it.set_tensor(self.i_t5h["index"], np.concatenate([t5h, self.null_h], axis=0))
            self.it.set_tensor(self.i_t5m["index"], np.concatenate([t5m, self.null_m], axis=0))
            self.it.set_tensor(self.i_sec["index"], np.concatenate([sec, sec], axis=0))
            self.it.set_tensor(self.i_lac["index"], np.concatenate([lac, lac], axis=0))
        else:
            self.t5h, self.t5m = t5h, t5m
            self.it.set_tensor(self.i_sec["index"], sec)   # seconds + lac constant → resident
            self.it.set_tensor(self.i_lac["index"], lac)

    def _fwd(self, x, t, t5h, t5m):
        """One batch=1 invoke (cfg==1.0 or sequential CFG)."""
        self.it.set_tensor(self.i_x["index"], x.astype(np.float32))
        self.it.set_tensor(self.i_t["index"], np.array([np.float32(t)], np.float32))
        self.it.set_tensor(self.i_t5h["index"], t5h)
        self.it.set_tensor(self.i_t5m["index"], t5m)
        self.it.invoke()
        self.n_fwd += 1
        return self.it.get_tensor(self.out).copy()

    def _fwd_batched(self, x, t):
        """One batch=2 invoke -> (v_cond, v_uncond). x is the shared batch=1 state (both CFG
        branches denoise the same latent), tiled to [2,256,L]; t5h/t5m/sec/lac already resident."""
        x2 = np.concatenate([x, x], axis=0).astype(np.float32)
        self.it.set_tensor(self.i_x["index"], x2)
        self.it.set_tensor(self.i_t["index"], np.array([np.float32(t), np.float32(t)], np.float32))
        self.it.invoke()
        self.n_fwd += 2
        v2 = self.it.get_tensor(self.out)
        return v2[0:1].copy(), v2[1:2].copy()

    def __call__(self, x, t, cross=None, gcond=None):
        if self.cfg == 1.0:
            return self._fwd(x, t, self.t5h, self.t5m)
        if self.batched:
            v_cond, v_uncond = self._fwd_batched(x, t)
        else:
            v_cond   = self._fwd(x, t, self.t5h,   self.t5m)
            v_uncond = self._fwd(x, t, self.null_h, self.null_m)
        return _apply_cfg(x, t, v_cond, v_uncond, self.cfg, self.apg)


# ─── Baked audio-in encoder (audio -> latents; SAME-S needs even L) ────────
class BakedEncoder:
    def __init__(self, path, threads=8, needs_even=False):
        from ai_edge_litert import interpreter as tfl
        self.it = tfl.Interpreter(model_path=str(path), num_threads=threads)
        self.i = self.it.get_input_details()[0]["index"]
        self.o = self.it.get_output_details()[0]["index"]
        self.needs_even = needs_even
        self._cur_N = None

    def _resize(self, N):
        if self._cur_N != N:
            self.it.resize_tensor_input(self.i, [1, 2, N], strict=False)
            self.it.allocate_tensors()
            self._cur_N = N

    def encode(self, audio, T_lat):
        """audio: (1,2,M), M a multiple of 4096 (caller pads to even L for SAME-S).
        Returns latents (1,256,T_lat) — trimmed back to the natural (decoder-independent) T_lat."""
        self._resize(audio.shape[-1])
        self.it.set_tensor(self.i, audio.astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.o)[:, :, :T_lat].copy()


# ─── Baked audio-out decoder (whole for SAME-S; chunked for SAME-L) ─────────
class BakedDecoder:
    def __init__(self, path, threads=8, needs_even=False):
        from ai_edge_litert import interpreter as tfl
        self.tfl = tfl
        self.path = str(path)
        self.threads = threads
        self.needs_even = needs_even   # SAME-S: T_aud=L*16 must be %32 -> pad odd L->even
        self.it = tfl.Interpreter(model_path=self.path, num_threads=threads)
        self.i = self.it.get_input_details()[0]["index"]
        self.o = self.it.get_output_details()[0]["index"]
        self.i_det = self.it.get_input_details()[0]
        self._cur_L = None

    def _resize(self, L):
        if self._cur_L != L:
            self.it.resize_tensor_input(self.i, [1, 256, L], strict=False)
            self.it.allocate_tensors()
            self._cur_L = L

    def decode_whole(self, latents):
        L = latents.shape[2]
        if self.needs_even and L % 2 != 0:
            # SAME-S needs even L. Pad one edge-replicated latent token, decode at L+1,
            # trim the extra token's audio. Narrow receptive field => negligible boundary
            # effect on the kept L*4096 samples. Keeps odd-L requests working without
            # changing the DiT/noise path (which stays natural-ceil, MLX/TRT-matched).
            latents = np.concatenate([latents, latents[:, :, -1:]], axis=2)
            self._resize(L + 1)
            self.it.set_tensor(self.i, latents.astype(np.float32))
            self.it.invoke()
            return self.it.get_tensor(self.o)[:, :, :L * SAMPLES_PER_LATENT].copy()
        self._resize(L)
        self.it.set_tensor(self.i, latents.astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.o).copy()          # [1,2,L*4096]

    def decode_chunked(self, latents, chunk, overlap, on_chunk=None):
        """Stitch audio directly (stride = 4096 samples per latent token)."""
        B, C, L = latents.shape
        if L <= chunk:
            return self.decode_whole(latents)
        S = SAMPLES_PER_LATENT
        out = np.zeros((B, 2, L * S), np.float32)
        K = chunk - 2 * overlap
        assert K > 0, (chunk, overlap)
        core = 0
        n_windows = (L + K - 1) // K
        wi = 0
        while core < L:
            core_end = min(core + K, L)
            win_start = core - overlap
            win_end = win_start + chunk
            if win_start < 0:
                win_start, win_end = 0, chunk
            if win_end > L:
                win_end, win_start = L, L - chunk
            win = latents[:, :, win_start:win_end]
            y = self.decode_whole(win)                    # [1,2,chunk*4096]
            ks = (core - win_start) * S
            ke = (core_end - win_start) * S
            out[:, :, core * S: core_end * S] = y[:, :, ks:ke]
            wi += 1
            if on_chunk:
                on_chunk(wi, n_windows)
            core = core_end
        return out


def valid_T_lat(seconds):
    """seconds -> T_lat via natural ceil, DECODER-INDEPENDENT and identical to MLX/TRT
    (sa3_mlx / sa3_trt.resolve_T_lat = max(1, ceil(seconds*44100/4096))). So MLX == TRT ==
    TFLite pick the SAME length -> no length-driven divergence, and true ODD-length
    requests are honored. The DiT handles odd L natively; SAME-L takes any L; SAME-S
    (even T_aud=L*16 % 32) pads odd->even at encode/decode and trims."""
    return max(1, int(np.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT)))


class _HelpfulParser(argparse.ArgumentParser):
    """argparse that prints full help (not just usage) when a flag is unknown / invalid,
    and tacks the shared example-commands block onto the end of -h / --help."""
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
            pass  # never let an examples-block failure mask the actual --help


def main():
    ap = _HelpfulParser(
        description="SA3 text-to-audio (+ audio-to-audio + inpainting + CFG) — baked-I/O varlen TFLite / CPU",
        allow_abbrev=False,   # --pr/--di/--de became ambiguous once *-precision flags landed
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("modes\n"
                "  text-to-audio    --prompt P\n"
                "  audio-to-audio   --prompt P --init-audio IN.wav [--init-noise-level σ]\n"
                "  inpainting       --prompt P --init-audio IN.wav --inpaint-range START,END\n"
                "  negative CFG     --prompt P --cfg N [--negative-prompt P_NEG] [--apg A]\n"))
    # Inputs
    ap.add_argument("--prompt", default=None,
                    help="Text prompt. Empty string = unconditional. If omitted, asked via stdin.")
    ap.add_argument("--negative-prompt", default=None,
                    help="Negative prompt for CFG's uncond branch. Ignored when --cfg=1.0. "
                         "When unset and --cfg≠1.0, the uncond branch uses all-zero T5 hidden+mask "
                         "(→ learned padding embeddings in-graph).")
    ap.add_argument("--init-audio", default=None,
                    help="WAV to start from. With --init-noise-level → audio-to-audio; with "
                         "--inpaint-range → inpainting. Encoder loaded automatically; audio is "
                         "trimmed/zero-padded to --seconds. Any format via ffmpeg fallback.")
    ap.add_argument("--inpaint-range", default=None,
                    help="Inpaint time range 'START,END' in seconds (e.g. '5,10'). Requires "
                         "--init-audio. Regenerates the masked span, preserves the rest (paste-back).")
    # Models
    ap.add_argument("--dit", choices=["sm-music", "sm-sfx", "medium"], default=None,
                    help="DiT model (names match sa3_mlx / sa3_trt). If omitted, prompts "
                         "interactively with an arrow-key picker.")
    ap.add_argument("--decoder", choices=["same-s", "same-l"], default=None,
                    help="Audio decoder. Default: same-s for sm-music/sm-sfx, same-l for medium. "
                         "If omitted, prompts interactively with an arrow-key picker.")
    ap.add_argument("--precision", choices=list(PRECISIONS), default="fp32",
                    help="DiT + decoder precision, wXaY = weight/activation bits (same flag "
                         "as the TensorRT CLI): 'fp32' (default — reference; on CPU also the "
                         "fast choice) | 'w16a32' (fp16 weights, half size, ≈lossless, 1.5-3× "
                         "slower) | 'w8a32' (GPTQ int8 weights, ¼ size at fp32 speed) | "
                         "'w8a8-dyn' (dynamic int8, fastest, lower quality). "
                         "Applies to the DiT, codec decoder, and (for a2a/inpaint) the "
                         "encoder; T5Gemma is single-precision. Per-component overrides below.")
    ap.add_argument("--dit-precision", choices=list(PRECISIONS), default=None,
                    help="Override --precision for the DiT only (e.g. a quantized DiT "
                         "with an fp32 codec).")
    ap.add_argument("--decoder-precision", choices=list(PRECISIONS), default=None,
                    help="Override --precision for the SAME codec only. The codec runs "
                         "once on a fixed latent, so its precision maps directly to "
                         "audio quality (w8a32 is transparent at 40-46 dB).")
    ap.add_argument("--encoder-precision", choices=list(PRECISIONS), default=None,
                    help="Override --precision for the SAME encoder (audio-to-audio / "
                         "inpainting only). Latent PSNR vs fp32 (same-s/same-l): "
                         "w16a32 ≈lossless (66/71 dB), w8a32 (GPTQ) 36/46 dB, "
                         "w8a8-dyn 24/30 dB.")
    # Sampling
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="Output length. T_lat = ceil(seconds*44100/4096) (natural ceil, decoder-"
                         "independent, matches MLX/TRT). Final WAV trimmed to exactly --seconds.")
    ap.add_argument("--steps", type=int, default=8,
                    help="Pingpong sampling steps (≥1). rf_denoiser is distilled for 8 (default).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed. If omitted, a random seed is chosen and printed at the end.")
    ap.add_argument("--init-noise-level", type=float, default=1.0,
                    help="σmax — schedule's starting noise level. With --init-audio: 0.4-0.8 varies, "
                         "1.0 = full regen (init ignored). Min %.2f (model NaNs at t≈0)." % MIN_SIGMA)
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="Classifier-Free Guidance scale. 1.0 = off (single pass). >1 toward prompt, "
                         "<1 toward uncond. Any value ≠1.0 runs cond+uncond each step (see --cfg-batched).")
    ap.add_argument("--apg", type=float, default=1.0,
                    help="Adaptive Projected Guidance [0..1], only when --cfg≠1.0. 1.0 = full APG "
                         "(orthogonal projection), 0.0 = vanilla CFG. rf_denoiser default 1.0.")
    ap.add_argument("--cfg-batched", action=argparse.BooleanOptionalAction, default=True,
                    help="When --cfg≠1.0, run cond+uncond as ONE batch=2 invoke on the variable-batch "
                         "DiT (default; ~7-29%% faster on Apple-Silicon AMX). --no-cfg-batched forces the "
                         "sequential batch=1 dual-pass (like TensorRT). Bit-identical at "
                         "fp32/w8a32; ~80 dB at w16a32; w8a8-dyn diverges by design (batch-"
                         "shared activation scales) — use --no-cfg-batched to reproduce "
                         "sequential baselines with w8a8-dyn.")
    # Runtime / output
    ap.add_argument("--threads", type=int, default=8, help="XNNPACK CPU threads.")
    ap.add_argument("--free-models", action=argparse.BooleanOptionalAction, default=True,
                    help="Free each model after its last use to lower peak RAM (default on).")
    ap.add_argument("--out", "-o", default=None,
                    help="Output WAV path. Relative → output/<file>; absolute → as-is. "
                         "If omitted, auto-named from the prompt + seed.")
    ap.add_argument("--play", action="store_true",
                    help="Play the WAV via macOS `afplay` after writing (blocking).")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error(f"--steps must be ≥ 1 (got {args.steps})")
    # per-component overrides fall back to the shared --precision
    args.dit_precision = args.dit_precision or args.precision
    args.decoder_precision = args.decoder_precision or args.precision
    args.encoder_precision = args.encoder_precision or args.precision

    # Interactive fills (match MLX/TRT).
    args = prompt_user_if_missing(args)
    if args.prompt is None:
        args.prompt = input("Prompt: ").strip()
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    dec = args.decoder or DEFAULT_DECODER[args.dit]
    T_lat = valid_T_lat(args.seconds)
    target_dur = T_lat * SAMPLES_PER_LATENT / SAMPLE_RATE

    # Resolve output path — auto-name from prompt+seed when --out is not given.
    # Relative paths land in output/ (auto-created); absolute paths honored as-is.
    # A relative path that already starts with "output/" is taken as-is (relative to
    # cwd) so `--out output/foo.wav` doesn't become output/output/foo.wav.
    if args.out is None:
        slug = re.sub(r'[^a-z0-9]+', '_', args.prompt.lower()).strip('_')[:48] or "out"
        # non-fp32 runs get a precision suffix so same-prompt/seed A/B runs across
        # --precision values don't overwrite each other
        if args.dit_precision == args.decoder_precision:
            prec_tag = "" if args.precision == "fp32" else f"_{args.precision}"
        else:
            prec_tag = f"_dit-{args.dit_precision}_dec-{args.decoder_precision}"
        args.out = f"{slug}{prec_tag}_{args.seed}.wav"
    out_path = Path(args.out)
    if not out_path.is_absolute() and out_path.parts[:1] != ("output",):
        out_path = REPO / "output" / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out = str(out_path)

    # Inpaint validation → latent range.
    inpaint_range = None
    inp_start_sec = inp_end_sec = None
    if args.inpaint_range is not None:
        if args.init_audio is None:
            sys.exit("error: --inpaint-range requires --init-audio (the audio to inpaint into)")
        try:
            s_str, e_str = args.inpaint_range.split(",")
            inp_start_sec = float(s_str.strip()); inp_end_sec = float(e_str.strip())
        except ValueError:
            sys.exit(f"error: --inpaint-range must be 'START,END' in seconds; got {args.inpaint_range!r}")
        if not (0 <= inp_start_sec < inp_end_sec <= args.seconds):
            sys.exit(f"error: invalid inpaint range {inp_start_sec}-{inp_end_sec}s "
                     f"(must satisfy 0 <= start < end <= {args.seconds}s)")
        inp_start_lat = max(0, int(round(inp_start_sec * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inp_end_lat   = min(T_lat, int(round(inp_end_sec   * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_range = (inp_start_lat, inp_end_lat)

    sigma_max = float(args.init_noise_level)
    if sigma_max < MIN_SIGMA:
        sys.exit(f"error: --init-noise-level={sigma_max} too low (min {MIN_SIGMA}; model NaNs at t≈0)")
    mode = ("inpaint" if inpaint_range else
            "audio-to-audio" if args.init_audio else "text-to-audio")

    # ── Preflight: download any missing model files BEFORE the banner/wall-clock. ──
    _preflight_download(args, dec)

    # ── Banner ──
    print()
    banner(f"SA3 → TFLite/CPU  {mode}")
    k = lambda s: dim(f"{s:>11}")
    print(f"  {k('prompt')}  {bold(repr(args.prompt))}")
    if args.negative_prompt:
        suffix = "" if args.cfg != 1.0 else dim("  (ignored: --cfg=1.0)")
        print(f"  {k('neg prompt')}  {bold(repr(args.negative_prompt))}{suffix}")
    line = f"  {k('dit')}  {magenta(args.dit)}   {k('decoder')}  {magenta(dec)}"
    if args.init_audio:
        enc_note = "" if args.encoder_precision == args.dit_precision else f" ({args.encoder_precision})"
        line += f"   {k('encoder')}  {magenta(dec + enc_note)}"
    prec_disp = (args.dit_precision if args.dit_precision == args.decoder_precision
                 else f"dit {args.dit_precision} · codec {args.decoder_precision}")
    line += f"   {k('precision')}  {magenta(prec_disp)}   {k('threads')}  {args.threads}"
    print(line)
    if args.init_audio:
        print(f"  {k('init audio')}  {bold(args.init_audio)}")
        if inpaint_range:
            s0, s1 = inpaint_range
            print(f"  {k('inpaint')}  {bold(f'{inp_start_sec:.2f}s..{inp_end_sec:.2f}s')} "
                  f"{dim(f'(latent {s0}..{s1} of {T_lat})')}")
    cfg_line = f"  {k('σmax')}  {bold(f'{sigma_max:.2f}')}   {k('cfg')}  {args.cfg}"
    if args.cfg != 1.0:
        cfg_line += dim(f"  (apg={args.apg}, {'batched' if args.cfg_batched else 'sequential'} CFG)")
    print(cfg_line)
    print(f"  {k('seconds')}  {args.seconds}s   {k('steps')}  {args.steps}   {k('seed')}  {args.seed}")
    print(f"  {k('T_lat')}  {T_lat} {dim(f'({target_dur:.2f}s → trimmed to {args.seconds:.2f}s)')}")
    print()

    # Stage numbering (extra stage when encoding init audio).
    N = 3 + (1 if args.init_audio else 0)
    TAG = {"t5": f"[1/{N}]"}
    if args.init_audio:
        TAG.update(enc=f"[2/{N}]", dit=f"[3/{N}]", dec=f"[4/{N}]")
    else:
        TAG.update(dit=f"[2/{N}]", dec=f"[3/{N}]")

    t_wall = time.perf_counter()

    # ── T5Gemma (tokenize + encode; + negative prompt for CFG) ──
    stage(TAG["t5"], "T5Gemma (tokenize + encode)")
    t0 = time.perf_counter()
    tok = P.Tokenizer()
    ids, mask = tok(args.prompt)                              # (1,256) int32 each
    t5 = P.T5GemmaTFLite(ensure_local(T5_REL), args.threads)
    t5_hidden = t5(ids, mask)                                 # (1,256,768) fp32
    null_h = null_m = None
    if args.cfg != 1.0:
        if args.negative_prompt:
            n_ids, n_mask = tok(args.negative_prompt)
            null_h = t5(n_ids, n_mask)
            null_m = n_mask.astype(np.float32)
        else:
            # All-zero hidden+mask → in-graph conditioner emits learned padding embeds
            # for every position (the standard unconditional branch). No extra T5 pass.
            null_h = np.zeros((1, COND_TOKENS, COND_DIM), np.float32)
            null_m = np.zeros((1, COND_TOKENS), np.float32)
    t5_ms = (time.perf_counter() - t0) * 1000
    stage(TAG["t5"], "T5Gemma (tokenize + encode)", t5_ms)
    sub(f"t5_hidden {t5_hidden.shape}   mask sum={int(mask.sum())}"
        + (f"   neg ({'prompt' if args.negative_prompt else 'zeros'})" if null_h is not None else ""))
    if args.free_models:
        del t5; _free()

    # ── (audio-to-audio / inpaint) Encode init audio → init_latents ──
    init_latents = None
    if args.init_audio:
        stage(TAG["enc"], f"Encode init audio → latents ({dec})")
        t0 = time.perf_counter()
        # SAME-S encoder needs even L; round the encode grid up, trim latents to T_lat.
        enc_L = T_lat + 1 if (dec == "same-s" and T_lat % 2 != 0) else T_lat
        target_samples = enc_L * SAMPLES_PER_LATENT
        audio_in = read_wav(args.init_audio)                 # (2, T_in)
        if audio_in.shape[-1] >= target_samples:
            audio_in = audio_in[:, :target_samples]
            init_action = f"trimmed to {target_samples} samples"
        else:
            pad = target_samples - audio_in.shape[-1]
            audio_in = np.pad(audio_in, ((0, 0), (0, pad)))
            init_action = f"zero-padded by {pad} samples"
        audio_in = audio_in[None]                            # (1,2,target_samples)
        enc = BakedEncoder(ensure_local(enc_rel(dec, args.encoder_precision)), args.threads, needs_even=(dec == "same-s"))
        init_latents = enc.encode(audio_in, T_lat)           # (1,256,T_lat)
        enc_ms = (time.perf_counter() - t0) * 1000
        stage(TAG["enc"], f"Encode init audio → latents ({dec})", enc_ms)
        sub(f"{init_action}   latents {init_latents.shape}")
        if args.free_models:
            del enc; _free()

    # ── Build inpaint local_add_cond + paste-back, and initial noise ──
    local_add_cond = None
    paste_back = None
    if inpaint_range is not None:
        s0, s1 = inpaint_range
        keep = np.ones((1, 1, T_lat), np.float32); keep[:, :, s0:s1] = 0.0   # 1=keep, 0=regen
        masked = init_latents.astype(np.float32) * keep
        local_add_cond = np.concatenate([keep, masked], axis=1)              # (1,257,T_lat), TRT layout
        paste_back = (init_latents.astype(np.float32), keep)

    x0, step_noise = P.make_noise(T_lat, args.steps, args.seed)              # x0 = pure noise
    if init_latents is not None and inpaint_range is None:
        # rf_denoiser init mix (linear interp): noise = init*(1-σmax) + pure*σmax
        x0 = init_latents.astype(np.float32) * (1.0 - sigma_max) + x0 * sigma_max

    # ── DiT load + pingpong sample ──
    stage(TAG["dit"], f"DiT — load + sample ({args.steps} steps, σmax={sigma_max:.2f})")
    t0 = time.perf_counter()
    cfg_note = (("CFG batched (1× batch=2 invoke/step)" if args.cfg_batched
                 else "CFG sequential (2× batch=1 invokes/step)") if args.cfg != 1.0 else "")
    print(f"        {dim('loading baked DiT ' + args.dit + ' ...')}", flush=True)
    backend = BakedDiT(ensure_local(dit_rel(args.dit, args.dit_precision)), T_lat, t5_hidden, mask.astype(np.float32),
                       args.seconds, args.threads, cfg=args.cfg, apg=args.apg,
                       null_hidden=null_h, null_mask=null_m, local_add_cond=local_add_cond,
                       batched=args.cfg_batched)
    load_ms = (time.perf_counter() - t0) * 1000
    sub(f"load {load_ms/1000:.1f}s" + (f"   {cfg_note}" if cfg_note else ""))

    sig = P.build_pingpong_schedule(args.steps, sigma_max)
    sched_str = " · ".join(f"{float(x):.3f}" for x in sig)
    sub(f"schedule  {sched_str}")

    step_prev = [time.perf_counter()]
    def on_step(i, total):
        now = time.perf_counter(); el = (now - step_prev[0]) * 1000; step_prev[0] = now
        if _USE_COLOR:
            bar_w = 20; filled = int(round(bar_w * i / total))
            bar = cyan("█" * filled) + dim("·" * (bar_w - filled))
            sys.stdout.write(f"\r\x1b[K        {dim('sampling')} {bar} "
                             f"{bold(f'step {i}/{total}')}  {yellow(f'{el:.0f} ms')}")
            sys.stdout.flush()
        else:
            print(f"        sampling step {i}/{total}  {el:.0f} ms", flush=True)

    t0 = time.perf_counter()
    latents = P.sample(backend, x0, step_noise, sig, None, None,
                       on_step=on_step, paste_back=paste_back)
    samp_ms = (time.perf_counter() - t0) * 1000
    if _USE_COLOR:
        sys.stdout.write("\r\x1b[K")
    stage(TAG["dit"], f"DiT — load + sample ({args.steps} steps, σmax={sigma_max:.2f})",
          load_ms + samp_ms)
    sub(f"sample {samp_ms:.0f} ms  ({samp_ms/max(args.steps,1):.0f} ms/step, "
        f"{backend.n_fwd} forwards)   latents {latents.shape}")
    if args.free_models:
        del backend; _free()

    # ── Decode (audio-out) + WAV ──
    stage(TAG["dec"], f"Decoder ({dec}, audio-out) + WAV")
    t0 = time.perf_counter()
    print(f"        {dim('loading baked decoder ' + dec + ' ...')}", flush=True)
    decoder = BakedDecoder(ensure_local(dec_rel(dec, args.decoder_precision)), args.threads, needs_even=(dec == "same-s"))
    load2_ms = (time.perf_counter() - t0) * 1000
    sub(f"load {load2_ms:.0f} ms")

    t0 = time.perf_counter()
    if dec == "same-l" and T_lat > SAMEL_CHUNK:
        print(f"        {dim(f'SAME-L chunked decode (chunk={SAMEL_CHUNK}, ovl={SAMEL_OVERLAP}) ...')}", flush=True)
        def on_chunk(i, n):
            print(f"        {dim(f'decode chunk {i}/{n}')}", flush=True)
        audio = decoder.decode_chunked(latents, SAMEL_CHUNK, SAMEL_OVERLAP, on_chunk=on_chunk)
        dmode = f"chunked (chunk={SAMEL_CHUNK}, ovl={SAMEL_OVERLAP})"
    else:
        print(f"        {dim('whole decode ...')}", flush=True)
        audio = decoder.decode_whole(latents)
        dmode = "whole"
    dec_ms = (time.perf_counter() - t0) * 1000

    audio_np = audio[0]                                        # (2, L*4096)
    req = int(round(args.seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > req:
        audio_np = audio_np[:, :req]
    P.save_wav(args.out, audio_np)
    stage(TAG["dec"], f"Decoder ({dec}, audio-out) + WAV", load2_ms + dec_ms)
    peak = float(np.abs(audio_np).max()); rms = float(np.sqrt((audio_np**2).mean()))
    sub(f"decode {dmode}  {dec_ms:.0f} ms   audio {audio_np.shape}   peak {peak:.3f} rms {rms:.3f}")
    if args.free_models:
        del decoder; _free()

    total = time.perf_counter() - t_wall
    dur = audio_np.shape[-1] / SAMPLE_RATE
    print()
    rule()
    print(f"  {bold(green('done'))}   {bold(f'{total:.2f}s')} wall  →  {dur:.1f}s audio  →  "
          f"{bold(yellow(f'{dur/total:.2f}× realtime'))}   {dim(f'seed {args.seed}')}")
    abs_out = os.path.abspath(args.out)
    try:
        rel_out = os.path.relpath(abs_out)
    except ValueError:
        rel_out = abs_out
    shown = rel_out if len(rel_out) <= len(abs_out) and not rel_out.startswith("..") else abs_out
    print(f"  {bold(green('▸ saved'))}  {bold(shown)}   {dim(f'({abs_out})' if shown != abs_out else '')}".rstrip())
    rule()

    if args.play:
        try:
            print(f"  {bold('▶ playing')}   {args.out}   {dim('(Ctrl-C to stop)')}")
            subprocess.run(["afplay", args.out], check=False)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
