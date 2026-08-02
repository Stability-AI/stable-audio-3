"""SA3 text-to-audio via TensorRT — full-pipeline CUDA-graph variant.

Same as sa3_trt.py but captures the entire fast-path inference pipeline
(T5 encode + DiT 8-step loop + decoder + narrow + DtoH) in ONE CUDA graph.
A single g.replay() runs everything from input_ids to a pinned-host int16
PCM tensor, eliminating per-stage Python / CUDA submission overhead.

Mega-graph path requirements:
  --cfg 1.0 (the default)
  --init-noise-level 1.0 (the default)
  no --inpaint-range
  no --init-audio
Anything else delegates to sa3_trt_core.main()'s eager pipeline. That handoff happens
BEFORE this module loads any engine — canon.main() loads its own set, so loading here
first left two full sets resident and doubled peak VRAM (see _delegate_to_eager).

Usage: same CLI as sa3_trt.py. Optional --no-mega-graph forces the eager
path even for the fast case (useful for ablation).

Bit-exact parity with sa3_trt.py: the build() warmup deliberately replays
canonical's exact pre-capture call sequence (3 warmups with zero T5/DiT/
decoder inputs, then 3 more 8-step DiT loops to mimic canonical's
graph_sampler.build) so that the decoder context's internal state at the
captured-call boundary matches canonical's "real" call. With this match,
each fresh process produces a WAV byte-identical to sa3_trt.py's.

Caveat — multi-inference drift: BOTH this script and sa3_trt.py drift
after the first inference within a single process, because the SAME-S/L
decoder engine carries internal state across calls (its TRT workspace
contents evolve). This is a property of the engine, not the wrapper.
The canonical CLI invokes one inference per process, so users never see
the drift in practice.
"""
from __future__ import annotations
import argparse, math, os, random, sys, threading, time, wave
from pathlib import Path
import numpy as np

# Reuse everything from the canonical script — including its lazy torch/trt
# imports, engine path resolution, and all the helper classes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sa3_trt_core as canon
from sa3_trt_core import (
    # Module-level state we touch:
    SAMPLE_RATE, SAMPLES_PER_LATENT, IO_CHANNELS, T5_MAX_LEN, COND_DIM,
    DIT_CHOICES, DECODER_PATHS, ENCODER_PATHS,
    DIT_ENGINE_FILES, DECODER_FILES, ENCODER_FILES, SHARED_FILES,
    HF_REPO_ID, ARCH, DECODER_MIN_L,
    # Helpers:
    _import_heavy, _ensure_files, _init_nvml, _stage_vram, _my_vram_bytes,
    _STAGE_VRAMS, _silence_fd, _fmt_mem,
    stage, sub, banner, bold, dim, cyan, yellow, green, magenta, red, rule,
    prompt_user_if_missing, _arrow_pick, _HelpfulParser,
    TRTRunner, DiTRunner, GraphPingpongSampler,
    t5gemma_encode, encoder_encode, decoder_decode,
    build_pingpong_schedule, sample_flow_pingpong,
    save_wav, read_wav,
    # Checked enqueue for capture regions — shared so both graphs behave alike.
    _enqueue_captured as _enqueue,
)


_WRAPPER_ONLY_FLAGS = ("--mega-graph", "--no-mega-graph")


def _delegate_to_eager(args) -> None:
    """Hand off to canon.main() for any configuration the mega-graph cannot serve.

    MUST be called before this module loads a single engine. canon.main() loads its
    own T5 + DiT + decoder, so loading them here first left two full sets resident —
    this frame keeps the first set referenced for as long as canon.main() runs. That
    doubled peak VRAM (measured 26.7-26.8 GB against 14.7 GB for medium + same-l) and
    on a 32 GB card it OOM'd *every* eager configuration: --cfg != 1,
    audio-to-audio, inpaint, --init-noise-level != 1, and --no-mega-graph. It is also
    the "malloc_consolidate(): invalid chunk size" abort that had been recorded
    against medium + same-l + cfg>1 — the trigger was never CFG itself, only that
    asking for CFG routes you here.

    canon.main() re-parses sys.argv from scratch, so anything this wrapper already
    resolved has to be pinned into argv or it gets re-derived differently. --seed is
    the one that silently corrupts results: prompt_user_if_missing() picks a RANDOM
    seed when the user omits it and the banner has already printed that value, so
    letting canon draw its own meant the render did not match the seed we reported
    and could not be reproduced from it.
    """
    print()
    print(dim("  Mega-graph path unavailable for this configuration; "
              "delegating to sa3_trt_core.main()..."))
    argv = [a for a in sys.argv if a not in _WRAPPER_ONLY_FLAGS]
    for flag, val in (("--dit", args.dit), ("--decoder", args.decoder),
                      ("--precision", args.precision), ("--seed", args.seed)):
        if val is not None and flag not in argv:
            argv += [flag, str(val)]
    sys.argv = argv
    canon.main()


# ─── Full-pipeline CUDA-graph runner ─────────────────────────────────────
class FullPipelineGraph:
    """Captures T5 + DiT-loop + decoder + narrow + DtoH in ONE CUDA graph.

    Layout (all on capture_stream):
      1. T5 engine reads input_ids_buf / attn_mask_buf, writes hidden_buf
      2. t5_mask_buf <- attn_mask_buf.float()
      3. dit._x_buf <- initial_noise_buf
      4. 8-step DiT pingpong loop (same body as GraphPingpongSampler.build):
            for i in range(steps):
              dit._t_buf[0] = sigma_curr_bufs[i]
              dit_engine.execute_async_v3
              ... pingpong math reads dit._vel_buf / dit._x_buf / noise_bufs[i] ...
      5. decoder_in_buf <- final_latents_buf
      6. decoder engine reads decoder_in_buf, writes pcm_int32_buf
         (1, T_lat*4096, 2) int32
      7. pcm_int16_buf <- pcm_int32_buf.to(int16)  (narrow cast)
      8. pinned_host_pcm.copy_(pcm_int16_buf[:requested_samples], non_blocking)

    Per-inference (replay path):
      1. Copy host int64 input_ids/attn_mask into pinned staging buffers
         (Python tokenizer output) — those staging buffers were registered
         as persistent device-readable srcs of HtoD copies into
         input_ids_buf / attn_mask_buf inside the captured graph.
      2. Refresh the 7 in-loop noise buffers + initial_noise_buf with
         fresh seeded randn, on the capture stream.
      3. Write seconds_total scalar into dit._sec_buf via copy_().
      4. self._graph.replay()
      5. sync — host pcm is ready in pinned_host_pcm[:requested_samples].
    """

    def __init__(self, t5_runner: TRTRunner, dit: DiTRunner, dec_runner: TRTRunner,
                 L: int, steps: int, requested_samples: int):
        self.t5_runner = t5_runner
        self.dit = dit
        self.dec_runner = dec_runner
        self.L = L
        self.steps = steps
        self.requested_samples = requested_samples
        # Will be allocated in build().
        self.input_ids_buf = None       # (1, 256) int64, device
        self.attn_mask_buf = None       # (1, 256) int64, device
        self.t5_hidden_internal = None  # (1, 256, 768) fp32, device — T5's output
        self.initial_noise_buf = None   # (1, 256, L) fp32, device
        self._noise_bufs = None         # list of (1, 256, L) fp32, device
        self._sigma_curr_bufs = None
        self._sigma_next_bufs = None
        self.latents_out_buf = None     # (1, 256, L) fp32, device — final latent
        self.decoder_in_buf = None      # (1, 256, L) fp32, device — what dec reads
        self.pcm_int32_buf = None       # (1, T_lat*4096, 2) int32, device
        self.pcm_int16_buf = None       # (T_lat*4096, 2) int16, device
        self.pinned_host_pcm = None     # (T_lat*4096, 2) int16, pinned host
        self.local_add_cond_buf = None  # (1, 257, L) fp32, device (kept zero)
        self._graph = None
        self._built = False

    def build(self, sigmas, seconds: float, sigma_max: float):
        """Allocate persistent buffers, prime + warm all engines, capture the graph.

        sigmas: 1-D tensor of length steps+1 (pre-shifted schedule).
        seconds: float — duration condition for DiT.
        sigma_max: float — used to initialize seeding semantics (informational).
        """
        torch = canon.torch
        L = self.L
        ns = self.steps
        T_full = L * SAMPLES_PER_LATENT

        # We force all three engines to execute on the same stream and capture
        # the whole pipeline on that one stream. Easiest: take the DiT runner's
        # stream as the canonical "capture stream" and just hand its
        # cuda_stream to every execute_async_v3 call.
        capture_stream = self.dit.runner.stream

        # ── T5 engine: set static shapes + bind addresses ──
        t5_ctx = self.t5_runner.context
        t5_ctx.set_input_shape("input_ids", (1, T5_MAX_LEN))
        t5_ctx.set_input_shape("attention_mask", (1, T5_MAX_LEN))
        # Allocate I/O buffers.
        ids_dt = self.t5_runner.in_dtype["input_ids"]        # int64
        mask_dt = self.t5_runner.in_dtype["attention_mask"]   # int64
        self.input_ids_buf = canon.torch.zeros(1, T5_MAX_LEN, dtype=ids_dt, device="cuda")
        self.attn_mask_buf = canon.torch.zeros(1, T5_MAX_LEN, dtype=mask_dt, device="cuda")
        hid_dt = self.t5_runner.out_dtype["hidden_states"]    # fp32
        self.t5_hidden_internal = canon.torch.zeros(1, T5_MAX_LEN, COND_DIM, dtype=hid_dt, device="cuda")
        t5_ctx.set_tensor_address("input_ids", self.input_ids_buf.data_ptr())
        t5_ctx.set_tensor_address("attention_mask", self.attn_mask_buf.data_ptr())
        t5_ctx.set_tensor_address("hidden_states", self.t5_hidden_internal.data_ptr())

        # ── DiT: bind persistent buffers (this also configures shapes/output) ──
        self.dit.bind_persistent(L)
        # The DiT consumes (1, 257, L) local_add_cond. For the text-to-audio
        # fast path it's all zeros — pre-fill once and never touch again.
        self.local_add_cond_buf = self.dit._local_add_cond_buf
        self.local_add_cond_buf.zero_()
        # Seconds is per-inference (small) but constant during a graph replay.
        # We write to dit._sec_buf BEFORE replay (outside the graph) so the
        # captured kernels read the new value.
        self.dit._sec_buf[0] = float(seconds)

        # ── Initial noise buffer (the graph copies it into dit._x_buf) ──
        self.initial_noise_buf = canon.torch.empty(1, IO_CHANNELS, L,
                                                    dtype=canon.torch.float32, device="cuda")

        # ── Per-step sigma buffers (graph reads from these by pointer) ──
        self._sigma_curr_bufs = [
            canon.torch.tensor(float(sigmas[i]), dtype=canon.torch.float32, device="cuda")
            for i in range(ns)
        ]
        self._sigma_next_bufs = [
            canon.torch.tensor(float(sigmas[i + 1]), dtype=canon.torch.float32, device="cuda")
            for i in range(ns)
        ]
        self._noise_bufs = [
            canon.torch.empty(1, IO_CHANNELS, L, dtype=canon.torch.float32, device="cuda")
            for _ in range(ns - 1)
        ]

        # ── Final-latent buffer (graph writes here after last step) ──
        self.latents_out_buf = canon.torch.empty(1, IO_CHANNELS, L,
                                                  dtype=canon.torch.float32, device="cuda")

        # ── Decoder: bind static-L addresses ──
        dec_ctx = self.dec_runner.context
        dec_ctx.set_input_shape("latent", (1, IO_CHANNELS, L))
        dec_in_dt = self.dec_runner.in_dtype["latent"]
        self.decoder_in_buf = canon.torch.empty(1, IO_CHANNELS, L, dtype=dec_in_dt, device="cuda")
        # Auto-detect output flavor like decoder_decode does.
        if "pcm" in self.dec_runner.out_dtype:
            self._dec_out_name = "pcm"
            pcm_dt = self.dec_runner.out_dtype["pcm"]
            self.pcm_int32_buf = canon.torch.empty(1, T_full, 2, dtype=pcm_dt, device="cuda")
            dec_ctx.set_tensor_address("latent", self.decoder_in_buf.data_ptr())
            dec_ctx.set_tensor_address("pcm", self.pcm_int32_buf.data_ptr())
        else:
            # Legacy fp output — graph-capture supported but the postprocess
            # path is different. Not the production target.
            self._dec_out_name = "audio"
            au_dt = self.dec_runner.out_dtype["audio"]
            # (1, 2, T_full) audio in [-1, 1]
            self._audio_legacy_buf = canon.torch.empty(1, 2, T_full, dtype=au_dt, device="cuda")
            dec_ctx.set_tensor_address("latent", self.decoder_in_buf.data_ptr())
            dec_ctx.set_tensor_address("audio", self._audio_legacy_buf.data_ptr())

        # ── int16 narrow + pinned-host output buffers ──
        # Full (T_full, 2) int16 GPU buffer — kept resident; we narrow with a
        # tensor view at replay time. The DtoH copy goes only to the requested
        # window (first requested_samples).
        self.pcm_int16_buf = canon.torch.empty(T_full, 2, dtype=canon.torch.int16, device="cuda")
        # Pinned host — pre-pinned, persistent. Exactly the right size for
        # this T_lat. The captured DMA copies from device to here.
        self.pinned_host_pcm = canon.torch.empty(T_full, 2, dtype=canon.torch.int16,
                                                  pin_memory=True)

        # ── Warmup: replicate canonical's warmup sequence exactly so that the
        #    decoder context's internal state at the time of the captured call
        #    matches canonical's state at its "real" call.
        #
        #    The decoder engine carries internal state across calls (TRT 10
        #    workspace + perhaps cached cuBLAS handles). The state evolves
        #    with each call and depends on the inputs. To get bit-exact
        #    parity with canonical we MUST follow the same call sequence:
        #      3 warmups with zero T5/DiT/decoder inputs,
        #      then the real captured call.
        with canon.torch.cuda.stream(capture_stream):
            # Pre-fill the input buffers with zeros to mimic canonical warmup.
            self.input_ids_buf.zero_()
            self.attn_mask_buf.zero_()
            # decoder_in_buf will be loaded with zero latents below.
            zero_latents = canon.torch.zeros_like(self.decoder_in_buf)
            zero_t = canon.torch.tensor([0.5], device="cuda")
            zero_h = canon.torch.zeros_like(self.dit._t5_hidden_buf)
            zero_m = canon.torch.zeros_like(self.dit._t5_mask_buf)
            zero_l = canon.torch.zeros_like(self.dit._local_add_cond_buf)
            zero_x = canon.torch.zeros_like(self.dit._x_buf)
            for _w in range(3):
                # T5 with zero ids (matches t5gemma_encode(" ") closely enough
                # that the engine state evolution is the same — what matters
                # for downstream determinism is that EACH engine's call count
                # and rough input pattern matches canonical).
                _enqueue(self.t5_runner.context, capture_stream, "t5 (mega-graph)")
                # DiT single step with zero inputs (matches canon's warmup
                # `dit.step(_w_x, _w_t, _w_h, _w_m, 30.0, _w_l)`).
                self.dit._t_buf[0] = 0.5
                self.dit._x_buf.copy_(zero_x)
                self.dit._t5_hidden_buf.copy_(zero_h)
                self.dit._t5_mask_buf.copy_(zero_m)
                self.dit._local_add_cond_buf.copy_(zero_l)
                self.dit._sec_buf[0] = 30.0
                _enqueue(self.dit.runner.context, capture_stream, "dit (mega-graph)")
                # Decoder with zero latents (matches canon's _w_lat = zeros).
                self.decoder_in_buf.copy_(zero_latents)
                _enqueue(self.dec_runner.context, capture_stream, "decoder (mega-graph)")
            # Then canon runs ONE full sampling pass (8 DiT calls, no decoder).
            # Mirror that here so the DiT context's state also matches.
            self.dit._x_buf.copy_(zero_x)
            for i in range(ns):
                self.dit._t_buf[0] = float(sigmas[i])
                _enqueue(self.dit.runner.context, capture_stream, "dit (mega-graph)")
            # Then canon's GraphPingpongSampler.build() does 2 more 8-step
            # passes of the DiT (still no decoder). Mirror that too.
            for _w in range(2):
                self.dit._x_buf.copy_(zero_x)
                for i in range(ns):
                    self.dit._t_buf[0] = float(sigmas[i])
                    _enqueue(self.dit.runner.context, capture_stream, "dit (mega-graph)")
                    v = self.dit._vel_buf.float()
                    denoised = self.dit._x_buf - self._sigma_curr_bufs[i] * v
                    if i < ns - 1:
                        nb = self._noise_bufs[i]
                        nb.normal_()
                        new_x = (1.0 - self._sigma_next_bufs[i]) * denoised + self._sigma_next_bufs[i] * nb
                    else:
                        new_x = denoised
                    if i < ns - 1:
                        self.dit._x_buf.copy_(new_x)
                    else:
                        self.latents_out_buf.copy_(new_x)
            # Now restore the DiT's persistent input buffers to zeros (they
            # got mutated above) — the captured graph will refill them anyway,
            # but be clean.
            self.dit._t5_hidden_buf.zero_()
            self.dit._t5_mask_buf.zero_()
            self.dit._local_add_cond_buf.zero_()
            self.dit._x_buf.zero_()
        capture_stream.synchronize()

        # ── Now capture ──
        self._graph = canon.torch.cuda.CUDAGraph()
        with canon.torch.cuda.stream(capture_stream):
            with canon.torch.cuda.graph(self._graph, stream=capture_stream):
                # Stage 1: T5
                _enqueue(self.t5_runner.context, capture_stream, "t5 (mega-graph)")
                # Cast attn_mask → fp32 → dit's t5_mask_buf
                self.dit._t5_mask_buf.copy_(self.attn_mask_buf.float())
                # T5 hidden (fp32) → dit's t5_hidden_buf
                self.dit._t5_hidden_buf.copy_(self.t5_hidden_internal)
                # Initial noise → dit's x_buf
                self.dit._x_buf.copy_(self.initial_noise_buf)
                # Stage 3: DiT loop
                for i in range(ns):
                    self.dit._t_buf[0] = self._sigma_curr_bufs[i]
                    _enqueue(self.dit.runner.context, capture_stream, "dit (mega-graph)")
                    v = self.dit._vel_buf.float()
                    denoised = self.dit._x_buf - self._sigma_curr_bufs[i] * v
                    if i < ns - 1:
                        nb = self._noise_bufs[i]
                        new_x = (1.0 - self._sigma_next_bufs[i]) * denoised + self._sigma_next_bufs[i] * nb
                    else:
                        new_x = denoised
                    if i < ns - 1:
                        self.dit._x_buf.copy_(new_x)
                    else:
                        self.latents_out_buf.copy_(new_x)
                # Stage 4: Decoder
                self.decoder_in_buf.copy_(self.latents_out_buf)
                _enqueue(self.dec_runner.context, capture_stream, "decoder (mega-graph)")
                # Stage 5a: narrow + cast int32 → int16 (or legacy fp32 → int16)
                if self._dec_out_name == "pcm":
                    # Belt-and-suspenders int16 clamp. New engines (from the
                    # FP32 clip+scale fix in the ONNX producer) already bound
                    # the int32 output to ±32767, so this is a no-op for them.
                    # Kept for backwards-compat with any older engine still in
                    # use, which has BF16 trunk rounding 32767 → 32768 and
                    # wrapping on int16 downcast (audible clicks).
                    self.pcm_int16_buf.copy_(self.pcm_int32_buf[0].clamp(-32767, 32767))
                else:
                    a = self._audio_legacy_buf[0].clamp(-1.0, 1.0) * 32767.0
                    self.pcm_int16_buf.copy_(a.to(canon.torch.int16).T)
                # Stage 5b: DtoH (captured non_blocking copy into pinned host)
                self.pinned_host_pcm[:self.requested_samples].copy_(
                    self.pcm_int16_buf[:self.requested_samples], non_blocking=True)

        self._built = True

    def run(self, input_ids_cpu, attn_mask_cpu, seed: int, seconds: float):
        """Execute the captured pipeline. Returns a numpy view of the pinned PCM.

        input_ids_cpu / attn_mask_cpu: (1, 256) int64 CPU tensors (tokenizer output).
        seed: int — used to seed the in-loop noise + the initial latent randn.
        seconds: float — duration condition. Written into dit._sec_buf before replay.

        Returns: numpy.ndarray (requested_samples, 2) int16 — a view into the
        pinned host buffer (zero-copy; valid until next run() overwrites it).
        """
        assert self._built, "call build() first"
        torch = canon.torch
        stream = self.dit.runner.stream

        # All host writes must happen on the SAME stream the graph replays on,
        # otherwise the replay races the input copies.
        with torch.cuda.stream(stream):
            # Update T5 input buffers via HtoD copy (captured by the graph
            # as raw device buffers — but the HtoD copy itself runs here,
            # OUTSIDE the graph, before replay).
            self.input_ids_buf.copy_(input_ids_cpu.to(self.input_ids_buf.dtype),
                                      non_blocking=True)
            self.attn_mask_buf.copy_(attn_mask_cpu.to(self.attn_mask_buf.dtype),
                                      non_blocking=True)
            # Update scalar inputs.
            self.dit._sec_buf[0] = float(seconds)
            # Refresh noise + initial latent (matches sa3_trt.py:
            #   noise = randn(seed=args.seed); replay-seed = args.seed + 1)
            g_init = torch.Generator(device="cuda")
            g_init.manual_seed(int(seed))
            self.initial_noise_buf.normal_(generator=g_init)
            g_loop = torch.Generator(device="cuda")
            g_loop.manual_seed(int(seed) + 1)
            for nb in self._noise_bufs:
                nb.normal_(generator=g_loop)
            # Replay the full pipeline.
            self._graph.replay()
        stream.synchronize()
        # pinned_host_pcm has been written by the DtoH; return a view.
        return self.pinned_host_pcm[:self.requested_samples].numpy()


# ─── Reusable inference class (CLI + gradio share this) ─────────────────
class SA3Inference:
    """SA3 TRT inference, set up once and reused.

    Loads engines + warms up + builds a CUDA graph at __init__. Subsequent
    generate() calls are graph replays (~30 ms on H100 at L=324). Different
    (T_lat, steps) combos build their own graphs on demand and are cached
    (LRU evict at MAX_GRAPHS).

    MVP scope: cfg=1.0, no init_audio, no inpaint, sigma_max=1.0 — i.e. the
    mega-graph fast path. Other configs raise NotImplementedError; the
    eager-fallback wiring is the next step.

    Thread-safe: serialize_generate() acquires an internal Lock so two
    callers can't trample the shared CUDA stream / persistent buffers.
    """

    MAX_GRAPHS = 4
    DEFAULT_SIGMA_MAX = 1.0  # mega-graph path requires this
    # Hard latent-length bounds. Every DiT engine's TRT profile is L=1..4096, the
    # SAME-L decoder caps at 4096, and for --precision fp8 the baked RoPE cos/sin
    # tables are sized to exactly 4096 (a longer render needs a re-bake). Enforced
    # in get_graph() so every library caller (gradio / API / generate) is covered,
    # not just the CLI.
    DIT_MIN_L = 1
    DIT_MAX_L = 4096
    # The decoders' own floor is DECODER_MIN_L (32) from sa3_trt_core — the DiT's
    # profile starts at 1, so nothing used to enforce it and a short render passed
    # validation, ran the DiT happily, then died or fell silent inside the decoder
    # instead of being rejected up front.

    def __init__(self, dit: str, decoder: str, *,
                 precision: str | None = None,
                 dec_precision: str = "canonical",
                 default_T_lat: int = 324, default_steps: int = 8,
                 default_seconds: float = 30.0,
                 models_dir: Path | None = None,
                 with_encoder: bool = False,
                 quiet: bool = False):
        """Load engines + build a warmup graph.

        Args:
            dit:            one of DIT_CHOICES — "sm-music" / "sm-sfx" / "medium"
            decoder:        one of DECODER_PATHS — "same-s" / "same-l"
            precision:      None (default → "fp16" for every model), or an
                            explicit "fp16" (canonical: FP16 trunk, FP32
                            RMSNorm/RoPE islands, FMHA-fused FP16 attention core),
                            "fp8" (medium only; max-speed clean tier — fp8 linears
                            + bf16 fused FMHA + baked fp32 RoPE, ~1.3× faster than
                            fp16 and clean at long sequence, capped L≤4096),
                            or "fp32" (bit-equiv
                            PyTorch eager, ~2× slower). ("fp16mixed" and the
                            retired medium-only "bf16" alias to "fp16".)
                            Engines auto-download
                            from HF if the requested file is missing.
            default_T_lat:  latent length to build the initial graph at
            default_steps:  pingpong steps for the initial graph
            default_seconds: duration condition for the initial graph (used for
                            the warmup pass; per-call generate() overrides)
            models_dir:     override the canonical models/ root (optional)
            with_encoder:   also load the audio encoder TRT engine (needed for
                            future audio-to-audio / inpaint modes)
            quiet:          suppress per-stage print() output from canon helpers
        """
        if dit not in DIT_CHOICES:
            raise ValueError(f"unknown dit={dit!r}; valid: {list(DIT_CHOICES)}")
        if decoder not in DECODER_PATHS:
            raise ValueError(f"unknown decoder={decoder!r}; valid: {list(DECODER_PATHS)}")
        # Resolve precision default per model: fp16 everywhere. Retired tokens
        # (fp16mixed, bf16) alias to fp16.
        if precision is None:
            precision = canon.default_precision(dit)
        precision = canon.normalize_precision(precision)
        if precision not in canon.PRECISIONS:
            raise ValueError(f"unknown precision={precision!r}; valid: {canon.PRECISIONS}")
        if precision == "fp8" and dit != "medium":
            raise ValueError("precision='fp8' is only available for dit='medium' "
                             "(fp8 linears + bf16 FMHA + baked fp32 RoPE; sm-music/sm-sfx ship fp16 only)")

        # Quiet: patch canon's stage/sub/_stage_vram to no-ops so loading
        # doesn't spam stdout (gradio in particular wants a clean log).
        if quiet:
            canon._USE_COLOR = False
            canon.stage = lambda *a, **kw: None
            canon.sub = lambda *a, **kw: None
            canon._stage_vram = lambda *a, **kw: 0

        # models-dir override (mirrors main()'s logic).
        if models_dir is not None and str(models_dir) != str(canon.MODELS_DIR):
            new_root = Path(models_dir).resolve()
            new_arch_dir = new_root / ARCH
            canon.T5GEMMA_PATH = new_arch_dir / "t5gemma" / "t5gemma_fp16.trt"
            for kk in DIT_CHOICES:
                DIT_CHOICES[kk]["engine"] = new_arch_dir / DIT_CHOICES[kk]["engine"].relative_to(canon.ARCH_DIR)
            for kk in DECODER_PATHS:
                DECODER_PATHS[kk] = new_arch_dir / DECODER_PATHS[kk].relative_to(canon.ARCH_DIR)
            for kk in ENCODER_PATHS:
                ENCODER_PATHS[kk] = new_arch_dir / ENCODER_PATHS[kk].relative_to(canon.ARCH_DIR)
            canon.MODELS_DIR = new_root
            canon.ARCH_DIR = new_arch_dir

        self.dit_name = dit
        self.decoder_name = decoder
        self.precision = precision
        self.with_encoder = with_encoder
        self.quiet = quiet

        # 1. Lazy-download any missing engines (precision-aware).
        needed = canon.get_engine_files(dit, decoder, precision, with_encoder=with_encoder, dec_tier=dec_precision)
        _ensure_files(needed)

        # 2. Heavy imports (torch + tensorrt + plugin).
        sub(dim("Loading..."))
        t0 = time.time()
        _import_heavy()
        sub(f"{dim('heavy imports')} {(time.time()-t0)*1000:.0f} ms")
        _init_nvml()
        torch = canon.torch

        # 3. Tokenizer + dist_shift (from bundled runtime.py).
        t0 = time.time()
        import runtime as rt
        rt.MODELS_DIR = str(canon.MODELS_DIR)
        rt.ARCH_DIR = str(canon.ARCH_DIR)
        state = rt.load()
        self.tokenizer = state["tokenizer"]
        self.dist_shift = state["dist_shift"]
        sub(f"{dim('tokenizer + dist-shift')} {(time.time()-t0)*1000:.0f} ms")

        # 4. Engine load (parallel).
        import concurrent.futures
        engine_specs = {
            "t5":  canon.T5GEMMA_PATH,
            "dit": canon.get_dit_engine_path(dit, precision),
            "dec": canon.resolve_decoder_engine(decoder, precision, dec_precision),
        }
        if with_encoder:
            engine_specs["enc"] = canon.resolve_encoder_engine(decoder, dec_precision)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(engine_specs)) as ex:
            futs = {name: ex.submit(TRTRunner, path) for name, path in engine_specs.items()}
            self.runners = {name: fut.result() for name, fut in futs.items()}
        sub(f"{dim(f'engine load (parallel, {len(self.runners)} engines)')} {(time.time()-t0)*1000:.0f} ms")

        self.dit = DiTRunner(self.runners["dit"])

        # 5. Graph cache + lock.
        self._graphs: dict[tuple[int, int], FullPipelineGraph] = {}
        self._graph_lru: list[tuple[int, int]] = []
        self._lock = threading.Lock()

        # 6. Build the initial graph at the default config.
        t0 = time.time()
        self.get_graph(default_T_lat, default_steps, default_seconds)
        torch.cuda.synchronize()
        sub(f"{dim(f'warmup + capture (T_lat={default_T_lat}, steps={default_steps})')} "
            f"{(time.time()-t0)*1000:.0f} ms")

    @staticmethod
    def resolve_T_lat(seconds: float, decoder: str) -> int:
        """seconds → T_lat (natural ceil; no even-bump).

        Earlier code bumped T_lat to even for same-s under the assumption that
        the BF16 trunk's internal chunking required it. Empirically, on real
        DiT-output latents the engine produces correct audio (cos ≥ 0.99 vs
        PT eager) at any L in [32, 4096], odd or even. Removing the bump keeps
        the DiT sequence length at exactly what the user requested.
        """
        return max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))

    def get_graph(self, T_lat: int, steps: int, seconds: float) -> FullPipelineGraph:
        """Return the cached graph for (T_lat, steps), building on miss.

        LRU-evicts when the cache exceeds MAX_GRAPHS to keep VRAM bounded.
        Called under the inference lock when reached from generate().
        """
        # Hard length cap (see DIT_MAX_L). Reject cleanly here so the library path
        # fails with a clear message rather than a cryptic TRT profile error deep
        # in graph build. For fp8 this is a real correctness limit (baked RoPE table).
        if T_lat < DECODER_MIN_L:
            min_s = DECODER_MIN_L * SAMPLES_PER_LATENT / SAMPLE_RATE
            raise ValueError(
                f"T_lat={T_lat} is below the {self.decoder} decoder's profile minimum of "
                f"{DECODER_MIN_L} (~{min_s:.2f}s). The DiT accepts it but the decoder "
                f"refuses the shape, so it is rejected here instead of failing obscurely "
                f"further in.")
        if not (self.DIT_MIN_L <= T_lat <= self.DIT_MAX_L):
            max_s = self.DIT_MAX_L * SAMPLES_PER_LATENT / SAMPLE_RATE
            reason = ("the fp8 baked RoPE table (sized to L=4096; a longer render needs a "
                      "re-bake — build/make_rope_baked_onnx.py --max-t)"
                      if self.precision == "fp8" else
                      "the DiT engine's optimization profile (built for L=1..4096)")
            raise ValueError(
                f"T_lat={T_lat} is out of range [{self.DIT_MIN_L}, {self.DIT_MAX_L}] "
                f"(~{max_s:.0f}s max), bounded by {reason} and the SAME-L decoder's 4096 cap.")
        key = (T_lat, steps)
        if key in self._graphs:
            self._graph_lru.remove(key)
            self._graph_lru.append(key)
            return self._graphs[key]

        sigma_max = self.DEFAULT_SIGMA_MAX
        sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max,
                                          dist_shift=self.dist_shift, latent_len=T_lat)
        # Size output buffer to MAX possible for this T_lat. generate() slices
        # to the actual requested samples, so one graph per (T_lat, steps)
        # serves any seconds within that T_lat's range.
        max_samples = T_lat * SAMPLES_PER_LATENT
        graph = FullPipelineGraph(self.runners["t5"], self.dit, self.runners["dec"],
                                   T_lat, steps, max_samples)
        graph.build(sigmas, seconds, sigma_max)
        canon.torch.cuda.synchronize()

        self._graphs[key] = graph
        self._graph_lru.append(key)
        # LRU eviction. Python GC + torch's caching allocator reclaim GPU
        # memory once the FullPipelineGraph reference is dropped.
        while len(self._graphs) > self.MAX_GRAPHS:
            evict = self._graph_lru.pop(0)
            del self._graphs[evict]
        return graph

    def generate(self, prompt: str, *,
                 seconds: float = 30.0, steps: int = 8,
                 seed: int | None = None,
                 # Below: MVP raises NotImplementedError; wiring planned.
                 init_noise_level: float = 1.0,
                 negative_prompt: str | None = None,
                 cfg: float = 1.0,
                 init_audio_path: str | None = None,
                 inpaint_range: tuple[float, float] | None = None,
                 ) -> tuple[np.ndarray, dict]:
        """Generate one audio clip. Returns (pcm_int16, timing_dict).

        Returns:
            pcm:    (T_samples, 2) int16 numpy array, T_samples = round(seconds*44100)
            timing: dict with 'inference_ms', 'graph_build_ms' (0 if cache hit),
                    'realtime', 'seed', 'T_lat', 'samples'
        """
        # MVP scope gate. These all raise; SA3Inference is wired for them on
        # the API surface but the implementations route through the eager
        # path in sa3_trt_core (TBD).
        if cfg != 1.0:
            raise NotImplementedError("CFG support not yet wired through SA3Inference")
        if init_audio_path is not None:
            raise NotImplementedError("audio-to-audio not yet wired through SA3Inference")
        if inpaint_range is not None:
            raise NotImplementedError("inpaint not yet wired through SA3Inference")
        if init_noise_level != 1.0:
            raise NotImplementedError("non-unity init_noise_level not yet wired")

        T_lat = self.resolve_T_lat(seconds, self.decoder_name)
        if not (1 <= T_lat <= 4096):
            raise ValueError(f"T_lat={T_lat} out of engine range [1, 4096]")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        with self._lock:
            t0 = time.time()
            graph = self.get_graph(T_lat, steps, seconds)
            graph_build_ms = (time.time() - t0) * 1000

            # Tokenize on CPU (outside graph).
            tok = self.tokenizer(prompt, return_tensors="pt", max_length=T5_MAX_LEN,
                                  padding="max_length", truncation=True)
            ids_cpu = tok["input_ids"]
            mask_cpu = tok["attention_mask"]

            t0 = time.time()
            pcm_full = graph.run(ids_cpu, mask_cpu, seed=int(seed), seconds=seconds)
            inference_ms = (time.time() - t0) * 1000

        # Slice and detach from the pinned-host buffer (the next run() would
        # overwrite it — we want a stable copy to hand to the caller).
        actual_samples = int(round(seconds * SAMPLE_RATE))
        pcm = pcm_full[:actual_samples].copy()

        return pcm, {
            "inference_ms":   inference_ms,
            "graph_build_ms": graph_build_ms,
            "realtime":       (seconds * 1000.0) / inference_ms if inference_ms > 0 else 0.0,
            "seed":           int(seed),
            "T_lat":          T_lat,
            "samples":        actual_samples,
        }


# ─── Main (mirrors sa3_trt.main() but routes through FullPipelineGraph) ─
def main():
    # Update module-level engine paths if --models-dir is overridden. We keep
    # canon's globals authoritative so we go through its path-fixup logic.
    ap = _HelpfulParser(
        description="SA3 text-to-audio via TensorRT — full-pipeline CUDA-graph variant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes\n"
            "  text-to-audio    --prompt P                          (mega-graph fast path)\n"
            "  audio-to-audio   --prompt P --init-audio IN.wav      (falls back to eager)\n"
            "  inpainting       --prompt P --init-audio IN.wav --inpaint-range A,B   (eager)\n"
            "  negative CFG     --prompt P --cfg N --negative-prompt P_NEG           (eager)\n"
        ),
    )
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--init-audio", default=None)
    ap.add_argument("--inpaint-range", default=None)
    ap.add_argument("--dit", choices=list(DIT_CHOICES.keys()), default=None)
    ap.add_argument("--decoder", choices=list(DECODER_PATHS.keys()), default=None)
    ap.add_argument("--dec-precision", choices=list(canon.DECODER_TIERS), default="canonical",
                    help="Decoder/encoder quantization tier (orthogonal to --precision, which is the "
                         "DiT). 'canonical' (bf16, default); 'fp8' (~1.14x, near-transparent); "
                         "'fp8_fast' (SAME-S only; fp8 attention projections too, ~1.22x, lossier). "
                         "Engine auto-downloads from HF. (For max-fidelity decode use --precision fp32.)")
    ap.add_argument("--precision", choices=list(canon.PRECISIONS) + list(canon._PRECISION_ALIAS), default=None,
                    help="DiT engine precision. Default is 'fp16' for every model: "
                         "canonical FP16 trunk + FP32 RMSNorm/RoPE islands + FMHA-fused FP16 "
                         "attention core, fp32-accurate at every length. 'fp8' (medium only) "
                         "is the max-speed clean tier — fp8 linears + bf16 fused FMHA + baked "
                         "fp32 RoPE, ~1.3x faster than fp16 at every length and clean at "
                         "long sequence (L<=4096). 'fp32' = bit-equiv PyTorch eager, slower. "
                         "('fp16mixed' and the retired medium-only 'bf16' are accepted as "
                         "deprecated aliases for 'fp16'.) Auto-downloads from HF.")
    ap.add_argument("--models-dir", default=str(canon.MODELS_DIR))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-noise-level", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--apg", type=float, default=1.0)
    ap.add_argument("--free-models", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pinned-copy", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--mega-graph", action=argparse.BooleanOptionalAction, default=True,
                    help="Capture the entire pipeline in one CUDA graph (T5+DiT+decoder+narrow+DtoH). "
                         "On by default. Falls back to eager path for cfg≠1.0, inpaint, or audio-to-audio.")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error(f"--steps must be ≥ 1 (got {args.steps})")
    if args.precision in canon._PRECISION_ALIAS:   # retired token → canonical, with a note
        _canon_p = canon._PRECISION_ALIAS[args.precision]
        print(f"  note: --precision {args.precision} is retired → using {_canon_p}", file=sys.stderr)
        args.precision = _canon_p

    # Mute display in quiet mode — match sa3_trt's behavior.
    if args.quiet:
        canon._USE_COLOR = False
        # The imports above are by name; we have to re-bind the names that
        # other code calls.
        def _noop_stage(idx_total, label, ms=None): pass
        def _noop_sub(text): pass
        def _noop_vram(label): return 0
        canon.stage = _noop_stage
        canon.sub = _noop_sub
        canon._stage_vram = _noop_vram

    # Engine dir override — fix up canon's globals if user passed --models-dir.
    if args.models_dir != str(canon.MODELS_DIR):
        new_root = Path(args.models_dir).resolve()
        new_arch_dir = new_root / ARCH
        canon.T5GEMMA_PATH = new_arch_dir / "t5gemma" / "t5gemma_fp16.trt"
        for kk in DIT_CHOICES:
            DIT_CHOICES[kk]["engine"] = new_arch_dir / DIT_CHOICES[kk]["engine"].relative_to(canon.ARCH_DIR)
        for kk in DECODER_PATHS:
            DECODER_PATHS[kk] = new_arch_dir / DECODER_PATHS[kk].relative_to(canon.ARCH_DIR)
        for kk in ENCODER_PATHS:
            ENCODER_PATHS[kk] = new_arch_dir / ENCODER_PATHS[kk].relative_to(canon.ARCH_DIR)
        canon.MODELS_DIR = new_root
        canon.ARCH_DIR = new_arch_dir

    args = prompt_user_if_missing(args)
    if args.prompt is None:
        args.prompt = input("Prompt: ").strip()

    # T_lat
    T_lat = max(1, math.ceil(args.seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    target_dur = T_lat * SAMPLES_PER_LATENT / SAMPLE_RATE

    DIT_MIN_L, DIT_MAX_L = 1, 4096
    if T_lat < DECODER_MIN_L:
        min_s = DECODER_MIN_L * SAMPLES_PER_LATENT / SAMPLE_RATE
        sys.exit(f"error: --seconds {args.seconds} gives T_lat={T_lat}, below the "
                 f"{args.decoder} decoder's minimum of {DECODER_MIN_L} latents "
                 f"(~{min_s:.2f}s). The DiT would accept it, but the decoder's "
                 f"optimization profile starts at {DECODER_MIN_L}; it would refuse the "
                 f"shape and the render would come back empty rather than erroring. "
                 f"Use --seconds {min_s:.2f} or more.")
    if T_lat < DIT_MIN_L or T_lat > DIT_MAX_L:
        max_s = DIT_MAX_L * SAMPLES_PER_LATENT / SAMPLE_RATE
        why = ("the fp8 baked RoPE table (sized to L=4096; a longer render needs a re-bake — "
               "build/make_rope_baked_onnx.py --max-t)"
               if args.precision == "fp8" else
               "the DiT engine's L=1..4096 profile") + " and the SAME-L decoder's 4096 cap"
        sys.exit(f"error: T_lat={T_lat} (= {args.seconds}s) out of [{DIT_MIN_L}, {DIT_MAX_L}] "
                 f"(~{max_s:.0f}s max), bounded by {why}.")

    # Inpaint validation (unchanged from sa3_trt)
    inpaint_range = None
    inp_start_sec = inp_end_sec = None
    if args.inpaint_range is not None:
        if args.init_audio is None:
            sys.exit("error: --inpaint-range requires --init-audio")
        s_str, e_str = args.inpaint_range.split(",")
        inp_start_sec = float(s_str.strip()); inp_end_sec = float(e_str.strip())
        if not (0 <= inp_start_sec < inp_end_sec <= args.seconds):
            sys.exit(f"error: invalid inpaint range")
        inp_start_lat = max(0, int(round(inp_start_sec * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inp_end_lat   = min(T_lat, int(round(inp_end_sec   * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_range = (inp_start_lat, inp_end_lat)

    sigma_max = float(args.init_noise_level)
    mode = ("inpaint" if inpaint_range else
            "audio-to-audio" if args.init_audio else "text-to-audio")

    # Can we use the mega-graph fast path?
    use_mega = (args.mega_graph and args.cfg == 1.0 and
                inpaint_range is None and args.init_audio is None and
                sigma_max == 1.0)
    # sigma_max==1 → noise = pure_noise (no init_latents mixing); plus the
    # negative cases above. Anything else → eager.

    # Banner
    _STAGE_VRAMS.clear()
    print()
    banner(f"SA3 → TRT  {mode}  {'[mega-graph]' if use_mega else '[eager]'}")
    k = lambda s: dim(f"{s:>10}")
    v = lambda s, w=10: f"{s:<{w}}"
    print(f"  {k('prompt')}  {bold(repr(args.prompt))}")
    if args.negative_prompt:
        suffix = "" if args.cfg != 1.0 else dim("  (ignored: --cfg=1.0)")
        print(f"  {k('neg prompt')}  {bold(repr(args.negative_prompt))}{suffix}")
    _dtier = f"   {k('dec-tier')}  {magenta(v(args.dec_precision))}" if args.dec_precision != "canonical" else ""
    print(f"  {k('dit')}  {magenta(v(args.dit))}   {k('decoder')}  {magenta(v(args.decoder))}{_dtier}   {k('precision')}  {v(args.precision)}")
    print(f"  {k('σmax')}  {bold(f'{sigma_max:.2f}')}")
    print(f"  {k('seconds')}  {v(f'{args.seconds}s')}   {k('steps')}  {v(args.steps)}   {k('seed')}  {args.seed}")
    print(f"  {k('cfg')}  {v(args.cfg)}   {k('mega-graph')}  {v('on' if use_mega else 'off (fallback)')}")
    print(f"  {k('T_lat')}  {T_lat} {dim(f'({target_dur:.2f}s → trimmed to {args.seconds}s)')}")
    print()

    # Everything below this point allocates. If the mega-graph cannot serve this
    # configuration, hand off NOW — before the lazy-download, the heavy imports and
    # above all the engine load — so only one set of engines is ever resident.
    if not use_mega:
        _delegate_to_eager(args)
        return

    # Lazy-download
    needed = canon.get_engine_files(args.dit, args.decoder, args.precision,
                                      with_encoder=bool(args.init_audio), dec_tier=args.dec_precision)
    _ensure_files(needed)

    # Heavy imports
    sub(dim("Loading..."))
    t0 = time.time()
    _import_heavy()
    sub(f"{dim('heavy imports')} {(time.time()-t0)*1000:.0f} ms")

    _init_nvml()

    # Tokenizer + dist_shift
    t0 = time.time()
    import runtime as rt
    rt.MODELS_DIR = str(canon.MODELS_DIR)
    rt.ARCH_DIR = str(canon.ARCH_DIR)
    state = rt.load()
    tokenizer = state["tokenizer"]
    dist_shift = state["dist_shift"]
    sub(f"{dim('tokenizer + dist-shift')} {(time.time()-t0)*1000:.0f} ms")

    # Engine load
    import concurrent.futures
    engine_specs = {
        "t5":  canon.T5GEMMA_PATH,
        "dit": canon.get_dit_engine_path(args.dit, args.precision),
        "dec": canon.resolve_decoder_engine(args.decoder, args.precision, args.dec_precision),
    }
    if args.init_audio:
        engine_specs["enc"] = canon.resolve_encoder_engine(args.decoder, args.dec_precision)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engine_specs)) as ex:
        futs = {name: ex.submit(TRTRunner, path) for name, path in engine_specs.items()}
        runners = {name: fut.result() for name, fut in futs.items()}
    sub(f"{dim(f'engine load (parallel, {len(runners)} engines)')} {(time.time()-t0)*1000:.0f} ms")

    # ── Warmup + graph build ──
    torch = canon.torch
    WARMUP_PASSES = 3
    t0 = time.time()
    dit = DiTRunner(runners["dit"])

    if use_mega:
        # ── Mega-graph fast path ──
        sigmas = build_pingpong_schedule(args.steps, sigma_max=sigma_max,
                                          dist_shift=dist_shift, latent_len=T_lat)
        requested_samples = int(round(args.seconds * SAMPLE_RATE))
        # The graph copies the full pinned-host buffer up to requested_samples;
        # the WAV save reads the same buffer.
        mega = FullPipelineGraph(runners["t5"], dit, runners["dec"],
                                  T_lat, args.steps, requested_samples)
        mega.build(sigmas, args.seconds, sigma_max)
        torch.cuda.synchronize()
        sub(f"{dim(f'warmup + capture')} {(time.time()-t0)*1000:.0f} ms")
        print()

        # ── Inference ──
        t_wall_start = time.time()
        # Tokenize (CPU only — outside the graph).
        tok = tokenizer(args.prompt, return_tensors="pt", max_length=T5_MAX_LEN,
                         padding="max_length", truncation=True)
        ids_cpu = tok["input_ids"]               # (1, 256) int64
        mask_cpu = tok["attention_mask"]
        # One replay does everything (T5 + DiT loop + dec + narrow + DtoH).
        pcm = mega.run(ids_cpu, mask_cpu, seed=int(args.seed), seconds=args.seconds)
        t_inference = time.time() - t_wall_start
        stage("[1/1]", f"Full pipeline (mega-graph: T5+DiT+dec+DtoH)", t_inference * 1000)
        _stage_vram("Mega-graph inference")

        # WAV save (excluded from inference timing — disk I/O).
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = canon.OUTPUT_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t0w = time.time()
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        t_disk = (time.time() - t0w) * 1000
        try:
            out_display = out_path.relative_to(canon.REPO).as_posix()
        except (ValueError, AttributeError):
            out_display = str(out_path)
        stage("[2/2]", f"WAV → {out_display}", t_disk)

        t_full = time.time() - t_wall_start
        audio_dur = pcm.shape[0] / SAMPLE_RATE
        print()
        rule()
        def _fmt_t(s):
            return f"{s*1000:.0f} ms" if s < 1 else f"{s:.2f} s"
        col1 = f"Inference {bold(_fmt_t(t_inference))}"
        col2 = f"Inference + Saving the WAV {bold(_fmt_t(t_full))}"
        col1_rt = dim(f"{audio_dur/t_inference:.0f}× realtime")
        col2_rt = dim(f"{audio_dur/t_full:.0f}× realtime")
        import re
        _ANSI = re.compile(r'\x1b\[[0-9;]*m')
        def _vlen(s): return len(_ANSI.sub('', s))
        gap = max(_vlen(col1), _vlen(col1_rt)) + 4
        indent = "  " + " " * len("done   ")
        print(f"  {bold(green('done'))}   {col1}{' ' * (gap - _vlen(col1))}{col2}")
        print(f"{indent}{col1_rt}{' ' * (gap - _vlen(col1_rt))}{col2_rt}")
        rule()
        print(f"  {bold(green('▸ saved'))}  {bold(cyan(out_display))}   {dim(f'({out_path.resolve()})')}")
        return


if __name__ == "__main__":
    main()
