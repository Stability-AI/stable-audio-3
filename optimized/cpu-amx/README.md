# sa3 cpu-amx — Stable Audio 3 on CPU (torch-free C++ AMX)

CPU-native inference for **Stable Audio 3 medium**, running the whole pipeline on
**torch-free C++ AMX engines** (Intel AMX / AVX-512). No PyTorch, MLX, TFLite, or
stable-audio-tools at runtime for text-to-audio — just the C++ `.so`s + numpy.

```
prompt ─▶ T5Gemma (C++ AMX) ─▶ numpy conditioner ─▶ DiT pingpong (C++ AMX int8)
       ─▶ SAME-S / SAME-L decoder (C++ AMX) ─▶ WAV
                     ▲
   audio-to-audio / inpaint: SAME-S / SAME-L encoder (C++ AMX) init-encode
```

The DiT 24-block int8 forward runs in `dit_cpu_amx.so`, T5Gemma in
`t5gemma_cpu_amx.so`, the decoder in `same_{s,l}_cpu_amx.so` (bf16) or
`same_{s,l}_int8fused_cpu_amx.so` (int8), and the encoder (audio→latent, for
`--init-audio`) in `same_{s,l}_encoder_cpu_amx.so` — all static oneDNN + AOT
kernels. The fp32 pre/post and the pingpong sampler are numpy. **Every mode is
100% torch-free** — text-to-audio, CFG, and (now) audio-to-audio / inpainting,
which encode the input through the C++ AMX SAME encoders (the old fp32 torch
autoencoder is kept only as a fallback).

## Documentation

- **[BUILD.md](BUILD.md)** — how to (re)build every `.so` + dump its weight blob (oneDNN, the AOT-Triton DiT, HF weight sourcing).
- **[TESTING.md](TESTING.md)** — the test suite: per-engine validations (`tests/`) + the CLI integration matrix; run `bash tests/run_all.sh`.
- **[LESSONS.md](LESSONS.md)** — hard-won findings (AMX-only speedups, fp32 attention islands, the bf16-RoPE bug, int8 quant, the global-static-weights trap, …). Read before touching the engines.

## One model, four modes

cpu-amx ships the **medium** DiT only (the int8 C++ core). For `sm-music` /
`sm-sfx`, use [`optimized/tflite`](../tflite) or [`optimized/mlx`](../mlx).

| `--decoder` | codec | notes |
|-------------|-------|-------|
| `same-l` (default) | native 426M medium codec | best fidelity |
| `same-s` | distilled 50M | faster, shares the medium latent space |

| mode | flags | example |
|------|-------|---------|
| text-to-audio  | `--prompt P` | new clip from a description |
| audio-to-audio | `--prompt P --init-audio IN.wav --init-noise-level σ` | variation of an existing clip |
| inpainting     | `--prompt P --init-audio IN.wav --inpaint-range "S,E"` | regenerate one span, keep the rest |
| CFG + negative | `--cfg 3.0 --negative-prompt P_NEG` | steer toward / away from prompts |

## Run

`./sa3` is a thin wrapper that picks a python interpreter (`$SA3_PYTHON`, else
`.venv/bin/python`, else `python3`) and runs `scripts/sa3_cpu_amx.py`.

```bash
# Text-to-audio (native codec)
./sa3 --prompt "A beautiful piano arpeggio grows into a cinematic climax" \
      --dit medium --decoder same-l --seconds 30 --out piano.wav

# Faster distilled decoder
./sa3 --prompt "lofi house loop, 120 BPM" --dit medium --decoder same-s --seconds 15 --out lofi.wav

# int8 fused decoder (smaller/faster), more threads
./sa3 --prompt "techno beat" --dit medium --decoder same-s \
      --decoder-precision int8 --threads 32 --out techno.wav

# Audio-to-audio variation (uses the fp32 torch AE to encode the input)
./sa3 --prompt "jazz fusion with electric piano" --dit medium --decoder same-l \
      --init-audio funk.wav --init-noise-level 0.7 --out funk_jazz.wav

# Inpaint seconds 4-7 (kept region stays bit-exact in latent space)
./sa3 --prompt "explosive drum break" --dit medium --decoder same-l \
      --init-audio funk.wav --inpaint-range "4,7" --out funk_drums.wav

# CFG + negative prompt
./sa3 --prompt "ambient drone" --cfg 3.0 --negative-prompt "drums, vocals" \
      --dit medium --decoder same-l --out drone.wav

# Play after writing (ffplay/aplay/paplay/afplay), and all examples
./sa3 --prompt "rainforest" --dit medium --decoder same-l --play
./sa3 --help
```

Omit `--decoder` for an interactive picker. Omit `--prompt` for a stdin prompt.
Relative `--out` paths land in `output/`; absolute paths are used as-is. The
output path is printed as a `▸ saved` line at the end.

### Without the wrapper

```bash
python scripts/sa3_cpu_amx.py --prompt "..." --dit medium --decoder same-l
```

## Web UI (gradio)

```bash
./sa3-gradio                  # public gradio.live share link (same-l default)
./sa3-gradio --no-share       # local-only (http://127.0.0.1:7860)
./sa3-gradio --decoder same-s
```

Every mode is wired: text-to-audio, CFG 0–10 + negative prompt + APG,
audio-to-audio (guide audio + init_noise_level), and inpainting (reference audio
+ start/end sliders). Each clip renders a 3-band tinted stereo mel spectrogram
(numpy port — no torch) with a click-to-seek playhead. Decoder and precision
switch from the dropdowns; WAVs land in `output/gradio/`. Extra UI packages
(`gradio`, `pillow`, `soundfile`) — `pip install -r requirements-gradio.txt`.

> Each generation runs in a **fresh subprocess** (of the tested CLI). The int8
> C++ DiT core is single-length per process, so this keeps changing seconds /
> steps safe at the cost of reloading the (mmap'd) weights each time.

## Precision & threads

MLX's `--dit-dtype` maps to two cpu-amx dials:

| flag | default | notes |
|------|---------|-------|
| `--decoder-precision` | `bf16` | `bf16` = best fidelity; `int8` = SmoothQuant+GPTQ fused w8a8 (smaller/faster) |
| `--threads` | 16 | threads for T5Gemma + the decoder |

The **DiT is int8-fixed** (the only shipped C++ core) and pinned to **1 thread**
— its `.so` heap-races at higher thread counts; at short clip lengths one thread
is already fast. T5Gemma is bf16 regardless.

## Flag reference

| Flag | Default | Notes |
|------|---------|-------|
| `--prompt` | (asks) | Text prompt; empty string = unconditional |
| `--negative-prompt` | — | CFG uncond branch; only used when `--cfg ≠ 1.0` |
| `--dit` | medium | **medium only**; `sm-music`/`sm-sfx` rejected (use tflite/mlx) |
| `--decoder` | same-l | `same-l` (native) or `same-s` (distilled) |
| `--decoder-precision` | bf16 | `bf16` or `int8` |
| `--threads` | 16 | T5Gemma + decoder threads (DiT is pinned to 1) |
| `--seconds` | 30 | Output length; `T_lat = ceil(seconds·44100/4096)` |
| `--steps` | 8 | Pingpong steps; 1 = single forward, 8 = sweet spot |
| `--seed` | random | Set for reproducibility; printed at the end |
| `--cfg` | 1.0 | Guidance scale; 1.0 = off, >1 toward prompt, <1 toward uncond |
| `--apg` | 1.0 | Adaptive Projected Guidance; only when `--cfg ≠ 1` |
| `--init-audio` | — | WAV input for audio-to-audio / inpaint (fp32 torch AE encode) |
| `--init-noise-level` | 1.0 | σmax; 0.4–0.8 typical for variation, 1.0 = full regen |
| `--inpaint-range` | — | `START,END` seconds; regenerate that span, keep the rest |
| `--free-models` | on | Free each model after last use; `--no-free-models` keeps them |
| `--out` / `-o` | auto | Relative → `output/`; absolute → as-is |
| `--play` | off | Play after writing (ffplay/aplay/paplay/afplay) |
| `--lora` | — | **not supported** in cpu-amx (accepted + ignored with a note) |

## Files

```
cpu-amx/
├── sa3                       ← CLI wrapper (use this)
├── sa3-gradio                ← web UI wrapper
├── README.md
├── requirements.txt          ← numpy, sentencepiece, soundfile (+ torch for a2a/inpaint)
├── requirements-gradio.txt   ← gradio, pillow, soundfile
├── assets/
│   ├── t5gemma_f16.npz        ← SentencePiece tokenizer (4.2 MB)
│   └── cond_medium.npz        ← conditioner weights (learned padding + seconds embedder)
├── output/                   ← default landing zone for WAVs
└── scripts/
    ├── sa3_cpu_amx.py        ← orchestrator CLI (invoked by ./sa3)
    ├── sa3_gradio.py         ← web UI (invoked by ./sa3-gradio)
    ├── pipeline.py           ← vendored numpy pipeline (tokenizer/conditioner/sampler/CFG/WAV)
    ├── backends.py           ← C++ AMX engine loaders + fp32 torch AE encoder
    ├── spec.py               ← mel-spectrogram renderer (numpy, torch-free)
    ├── examples.py           ← shared examples block (--help)
    └── test_all_configs.py   ← full-stack self-test (assets + every CLI mode)
```

The heavy weights are **not** in this directory — the C++ engines mmap them from
their build directories, imported by `sys.path` (see `scripts/backends.py`):

| component | engine `.so` + weights |
|-----------|------------------------|
| T5Gemma | `/weka2/cj/clod/t5gemma_cpu_amx/` |
| DiT (medium int8) | `/weka2/cj/clod/tritoncpu_sa3/aot_speedprove/` + `.../aot_stage2/` |
| SAME-S / SAME-L (bf16) | `/weka2/cj/clod/same_{s,l}_cpu_amx/` |
| SAME-S / SAME-L (int8) | `/weka2/cj/clod/same_{s,l}_int8fused_cpu_amx/` |
| fp32 AE encoder (a2a/inpaint) | `/weka2/cj/clod/sa3s/fast_load/` (torch) |

## Notes on the design

- **One process = one T_lat.** The int8 C++ DiT core allocates length-specific
  scratch on first call and heap-corrupts if invoked at a *second* sequence
  length in the same process; it also double-frees at teardown. The CLI runs one
  generation per process and calls `os._exit(0)` before teardown, so this is
  invisible. The gradio isolates every generation in a fresh CLI subprocess.
- **Encoder is the only torch step.** No C++/TFLite encoder exists for this
  platform, so `--init-audio` (audio-to-audio / inpainting) loads the fp32 torch
  SAME-L autoencoder to encode the input to latents (auto-picks a free CUDA GPU;
  CPU fallback). It is imported lazily — pure text-to-audio never touches torch.
- **Inpainting is paste-back only.** The C++ DiT core does not expose the
  `local_add_cond` (to-local-embed) conditioning channel, so the regenerated
  span is conditioned on the prompt (+ noise) rather than the surrounding
  context. Per-step paste-back keeps the latents **outside** the mask bit-exact.
- **CFG uncond convention.** With no negative prompt, the uncond branch runs the
  DiT on the conditioner's **learned padding embedding** (conditioner applied to
  an all-zero T5 hidden + mask) — the standard unconditional case, matching the
  TensorRT / baked-TFLite releases. A negative prompt replaces it with that
  prompt's conditioning. CFG is a sequential dual-pass (no batch-2 C++ DiT).
- **Verified fidelity.** The C++ T5Gemma matches TFLite at cosine 0.99997; the
  full C++ pipeline matches the TFLite reference at ~43 dB / 0.997 correlation.

## License & attribution

Model weights derived from Stability AI's Stable Audio 3 checkpoints. T5Gemma
text encoder from Google. Use of the Stable Audio 3 weights is governed by the
**Stability AI Community License** — see <https://stability.ai/license>.
