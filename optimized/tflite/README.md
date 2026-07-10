# sa3_tflite — Stable Audio 3 on CPU via LiteRT / TFLite

Portable CPU inference for **Stable Audio 3** — the LiteRT/TFLite sibling of the
[MLX](../mlx) (Apple Silicon) and [TensorRT](../tensorRT) (NVIDIA GPU) releases.
No PyTorch, transformers, or stable-audio-tools at runtime — just `ai_edge_litert`
(LiteRT) driving fully self-contained `.tflite` graphs through the XNNPACK CPU
delegate. Runs anywhere LiteRT runs: **macOS / Linux, x86 / ARM**.

## Quick Install

One line on a fresh machine — installs everything and plays back ~30 seconds of
"Impending tribal, epic orchestral buildup":

```bash
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash
```

Already cloned the repo? Run from inside `optimized/tflite/`:

```bash
./install.sh                                              # one-time setup
./sa3 --prompt "Impending tribal, epic orchestral buildup" --play           # generates + plays
```

## Three models, four modes

| `--dit`    | model              | best for                       |
|------------|--------------------|--------------------------------|
| `sm-music` | sa3-sm-music (50 M block)  | fast music generation  |
| `sm-sfx`   | sa3-sm-sfx   (50 M block)  | sound effects          |
| `medium`   | sa3-medium-ARC (1.4 B)     | higher-quality music, slower |

| mode             | flags                                         | example                          |
|------------------|-----------------------------------------------|----------------------------------|
| text-to-audio    | `--prompt P`                                  | new clip from a description      |
| audio-to-audio   | `--prompt P --init-audio IN.wav --init-noise-level σ` | variation of an existing clip |
| inpainting       | `--prompt P --init-audio IN.wav --inpaint-range "S,E"` | regenerate one section, keep rest |
| CFG + negative   | `--cfg 3.0 --negative-prompt P_NEG`           | steer toward / away from prompts |

```
prompt ─▶ T5Gemma encoder ─▶ DiT pingpong sampler ─▶ SAME-S/L decoder ─▶ WAV
                                       ▲
                  optional: encoder + init audio (audio-to-audio / inpaint)
```

## Precision variants (`--precision`)

Same flag as the TensorRT CLI's `--precision`; the values use the wXaY naming
(weights/activations bit-widths). One flag switches the DiT, the codec decoder,
and (for audio-to-audio / inpainting) the encoder together; T5Gemma stays
single-precision. All
variants keep the full feature set: variable length (any `--seconds`, odd or even
latent counts) and variable batch (batched or sequential CFG). Missing files
lazy-download from HuggingFace on first use.

| `--precision` | legacy name | size (sm DiT / medium DiT / codecs) | quality | CPU speed |
|---------------|-------------|-------------------------------------|---------|-----------|
| `fp32` (default) | — | 1.8 GB / 5.8 GB / 0.2–1.8 GB | reference | 1× — on CPU this is *also* the fast choice |
| `w16a32` | fp16mixed | 0.9 / 2.9 / 0.1–0.9 GB | ≈lossless (fp16 weights, fp32 activations; 62–75 dB per-forward) | 1.5–3× slower, model-dependent (XNNPACK dequantizes per-matmul) |
| `w8a32` | woint8 | 0.45 / 1.5 / 0.05–0.5 GB | GPTQ int8 weights — codecs transparent (40–46 dB); DiT gives a *different but plausible* sample | ≈fp32 |
| `w8a8-dyn` | dynint8 | 0.45 / 1.5 / 0.05–0.5 GB | lowest (int8 weights + activations, per-invoke dynamic scales) | fastest (~1.2–1.3×) |

*(Names follow the wXaY convention — weight/activation bit-widths, as in LLM releases:
quantization here touches the FULLY_CONNECTED weights; activations stay fp32 except
`w8a8-dyn`, whose int8×int8 matmuls are what make it the only faster-than-fp32 variant.
All bit-widths are per-channel weight grids; "16" is fp16 — there is no int16 variant.
Speed factors measured on an Apple M4 Pro MacBook Pro, XNNPACK, 8 threads.)*

Two things worth knowing, both counter-intuitive:

- **Quantization buys *size*, not speed, on CPU.** Weight-only int8 dequantizes
  to fp32 before each matmul, so they run at fp32 speed; fp16 is *slower* than fp32.
  Only `dynint8` (quantized activations → true int8 matmuls) is faster.
- **DiT quantization error compounds.** The 8-step sampler is chaotically sensitive:
  per-step weight error turns into a *different* (still plausible) sample rather than
  a noisy version of the fp32 one. Judge DiT precisions by ear; decoder precisions
  by PSNR (they run once on a fixed latent).

One CFG note: batched and sequential CFG are bit-identical for `fp32`/`w8a32` and
inaudibly different (~80 dB) for `w16a32`, but under `w8a8-dyn` the batch=2
invoke shares activation-quantization scales across the cond/uncond rows, so batched
CFG yields a *different plausible sample* than sequential. Pass `--no-cfg-batched`
with `w8a8-dyn` when you need run-to-run reproducibility against sequential baselines.

```bash
# quarter-size models, same speed and near-identical quality on the decoder side
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --precision w8a32

# fastest CPU inference (quality tradeoff)
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --precision w8a8-dyn
```

**Encoder variants** (used by audio-to-audio / inpainting) exist at the same
precisions; int8 weights are GPTQ-calibrated on real audio, like the DiT and codec
(measured latent PSNR vs fp32, same-s / same-l): `w16a32` 66 / 71 dB (≈lossless),
`w8a32` 36 / 46 dB, `w8a8-dyn` 24 / 30 dB. For the most quality-sensitive
inpainting you can still pin `--encoder-precision fp32`.

`--dit-precision` / `--decoder-precision` / `--encoder-precision` override the shared
flag per component — useful because they react to quantization very differently (the
codec's precision maps directly to audio fidelity; the DiT's changes *which* sample
you get; the encoder's affects a2a/inpaint latents):

```bash
# fastest DiT, reference-quality codec
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s \
      --precision fp32 --dit-precision w8a8-dyn
```

## Install

```bash
./install.sh
```

`install.sh` is uv-based. On a fresh machine it will:

1. Install [uv](https://github.com/astral-sh/uv) via the official curl
   installer if it's missing (prompts y/N; `-y` skips the prompt).
2. Create a project-local `.venv/` with managed Python 3.11.
3. `uv pip install` the runtime deps into the venv (much faster than pip).
4. Ask which DiT bundles to download from HuggingFace
   (`stabilityai/stable-audio-3-optimized`). Each pick pulls its matching
   audio codec; T5Gemma (the shared text encoder) is downloaded once.
   Already-present weights are skipped.

End-to-end on a fresh machine: **~10 seconds** + weight downloads.

> Don't want to pre-pick bundles? Skip install entirely and just run
> `./sa3 --prompt …` — any missing model file is downloaded from HF on
> first use and symlinked into `models/tflite/` from the HuggingFace cache.

Portable CPU (no GPU required). Python 3.9+. `./install.sh --python 3.12` to
pin a different Python.

## Run

`./sa3` is a thin shell wrapper around `.venv/bin/python scripts/sa3_tflite.py
"$@"` that prompts to run `./install.sh` if uv or `.venv/` isn't set up.

```bash
# Text-to-audio
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --out lofi.wav

# Sound effects
./sa3 --prompt "footsteps on gravel" --dit sm-sfx --decoder same-s --out steps.wav

# Higher-quality music (medium DiT, chunked SAME-L decode)
./sa3 --prompt "A beautiful piano arpeggio grows into a cinematic climax" \
      --dit medium --decoder same-l --seconds 30 --out piano.wav

# Audio-to-audio variation (σmax 0.4-0.8 typical)
./sa3 --prompt "jazz fusion with electric piano" --dit sm-music --decoder same-s \
      --init-audio funk.wav --init-noise-level 0.7 --out funk_jazz.wav

# Inpaint seconds 4-7
./sa3 --prompt "explosive drum break" --dit sm-music --decoder same-s \
      --init-audio funk.wav --inpaint-range "4,7" --out funk_drums.wav

# CFG + negative prompt
./sa3 --prompt "ambient drone" --cfg 3.0 --negative-prompt "drums, vocals" \
      --dit sm-music --decoder same-s --out drone.wav

# Generate + play immediately (afplay; Ctrl-C stops both)
./sa3 --prompt "rainforest" --dit sm-sfx --decoder same-s --play

# All options + categorised examples
./sa3 --help
```

Omit `--dit` / `--decoder` for an interactive arrow-key picker. Omit
`--prompt` for a stdin prompt. Relative `--out` paths land in `output/`
(auto-created); absolute paths are honoured as-is. The output path is
printed prominently as a `▸ saved` line at the end of each run.

Use `--threads` to control the XNNPACK CPU thread count (default 8).

### Without the wrapper

```bash
.venv/bin/python scripts/sa3_tflite.py --prompt "..." --dit medium --decoder same-l
# or, after `source .venv/bin/activate`:
python scripts/sa3_tflite.py --prompt "..." --dit medium --decoder same-l
```

## Speed & memory

This is a **CPU** path — it trades the GPU releases' speed for portability. The
small models comfortably beat realtime on a modern laptop CPU; `medium` is slower
(its DiT is ~5.8 GB fp32 and it chunk-decodes SAME-L). Use ≥ ~20 s clips: very
short clips have too few latent tokens for the sampler to settle into a coherent
loop. Throughput scales with `--threads` up to your physical core count (4–8 is
the usual sweet spot; more threads on a short model adds overhead).

For sub-realtime latency on a supported device, prefer the GPU siblings:
[MLX](../mlx) on Apple Silicon, [TensorRT](../tensorRT) on NVIDIA.

## Flag reference

| Flag                  | Default  | Notes                                                                 |
|-----------------------|----------|-----------------------------------------------------------------------|
| `--prompt`            | (asks)   | Text prompt; empty string = unconditional                              |
| `--negative-prompt`   | —        | CFG uncond branch; only used when `--cfg ≠ 1.0`                       |
| `--dit`               | (asks)   | `sm-music`, `sm-sfx`, or `medium`                                     |
| `--decoder`           | (asks)   | `same-s` (pairs with sm-*) or `same-l` (pairs with medium)            |
| `--seconds`           | 30       | Output length (use ≥ ~20 s)                                          |
| `--steps`             | 8        | Pingpong sampler steps; 1 = single forward (fastest), 8 = sweet spot  |
| `--seed`              | random   | Set for reproducibility; the chosen seed is printed at the end        |
| `--cfg`               | 1.0      | Guidance scale; 1.0 = off, >1 toward prompt, <1 toward uncond. ≠1 runs cond+uncond each step |
| `--apg`               | 1.0      | Adaptive Projected Guidance; only matters when `--cfg ≠ 1`            |
| `--cfg-batched`       | on       | When `--cfg ≠ 1`, run cond+uncond as one batch=2 invoke on the variable-batch DiT (~7–29% faster on Apple-Silicon AMX). `--no-cfg-batched` → sequential batch=1 dual-pass. Bit-identical at `fp32`/`w8a32` — see [Precision variants](#precision-variants---precision) for `w16a32`/`w8a8-dyn` |
| `--init-audio`        | —        | WAV (any format via ffmpeg) input for audio-to-audio / inpaint       |
| `--init-noise-level`  | 1.0      | σmax; 0.4–0.8 typical for variation, 1.0 = full regen, >1 = overshoot |
| `--inpaint-range`     | —        | `START,END` seconds; regenerate that span, keep the rest              |
| `--threads`           | 8        | XNNPACK CPU threads (all TFLite models run on CPU)                    |
| `--free-models`       | on       | Free each model after its last use; `--no-free-models` keeps them resident |
| `--out` / `-o`        | (auto)   | Relative → `output/<file>`; absolute → as-is. 16-bit PCM stereo @ 44.1 kHz, trimmed to exactly `--seconds` |
| `--play`              | off      | After writing, play via `afplay` (macOS); Ctrl-C stops both           |

All `.tflite` models are **fp32** except T5Gemma, which is **fp16** (numerically
lossless there). There is no dtype knob: on CPU, int8/fp16 weights buy size, not
speed (XNNPACK dequantizes to fp32 to matmul), and int8 costs quality on the DiT
— so this release ships the fp32 graphs directly. (See "Notes on the design".)

## Files

```
sa3_tflite/
├── sa3                            ← shell wrapper (use this)
├── install.sh                     ← uv bootstrap (run once)
├── bootstrap.sh                   ← one-line curl installer
├── README.md
├── requirements.txt               ← ai_edge_litert, numpy, sentencepiece, soundfile, huggingface_hub
├── output/                        ← default landing zone for generated WAVs
├── scripts/
│   ├── sa3_tflite.py              ← orchestrator CLI (invoked by ./sa3)
│   ├── weights.py                 ← weights manifest + HF auto-download
│   ├── examples.py                ← shared examples block (--help + post-install)
│   └── install.py                 ← install.sh's Python half (bundle picker)
└── models/
    ├── tokenizer.model            ← SentencePiece model, BUNDLED (~4 MB; T5Gemma tflite is encoder-only)
    ├── defs/
    │   └── tflite_pipeline.py     ← Tokenizer + T5Gemma front-end + pingpong schedule + sampler + WAV
    └── tflite/                    ← .tflite models (auto-downloaded; ~2.3 GB small, ~9.5 GB medium)
        ├── t5gemma/encoder_fp16.tflite        564 MB   text encoder (fp16)
        ├── sa3-sm-music/dit_fp32.tflite       1.8 GB   small music DiT (conditioner baked in)
        ├── sa3-sm-sfx/dit_fp32.tflite         1.8 GB   small sfx DiT (conditioner baked in)
        ├── sa3-m/dit_fp32.tflite              5.8 GB   medium DiT (conditioner baked in)
        ├── same-s/{enc,dec}_fp32.tflite       ~220 MB each   shared sm-* codec
        └── same-l/{enc,dec}_fp32.tflite       ~1.8 GB each   medium codec
```

The DiT graphs are **baked-I/O**: the conditioner (prompt-padding + seconds
embedder) and the patch/unpatch are compiled into the graph, so the DiT takes the
raw T5Gemma output directly and the decoder emits audio directly. The two small
DiTs share the SAME-S codec (bit-exact between checkpoints), so only one set of
small-codec files is shipped.

## Auto-download from HuggingFace

Model files aren't bundled — they're pulled from
`stabilityai/stable-audio-3-optimized` (under `tflite/…`) on first use and
symlinked into `models/tflite/` from the HF cache. No duplication. Anonymous
downloads work but are rate-limited; `huggingface-cli login` with a free read-only
token lifts the cap. The SentencePiece tokenizer (`models/tokenizer.model`, ~4 MB)
is the one weight that IS committed, since the `.tflite` T5Gemma is encoder-only.

## Notes on the design

- **Baked-I/O varlen graphs.** Each `.tflite` is a single self-contained graph
  with the conditioner and patch/unpatch in-graph, accepting a variable sequence
  length — so one file serves any `--seconds`. The DiT is a 6-input graph
  (`x, t, t5_hidden, t5_mask, seconds, local_add_cond`); feed raw T5 outputs and
  the in-graph conditioner handles prompt-padding + seconds-embedding.
- **fp32 everywhere (except fp16 T5Gemma).** On CPU, quantizing buys size, not
  speed — XNNPACK dequantizes int8/fp16 weights to fp32 to matmul, so fp16 is
  actually *slower* and int8 gives no speedup. And the DiT will not go int8 at
  quality: per-step error compounds over the 8 chaotic sampling steps into a
  *different* (still plausible) sample, not a degraded one. So this release ships
  the fp32 graphs directly. T5Gemma fp16 is the sole exception — it's numerically
  lossless there and halves that file.
- **Monotonic audio-to-audio schedule.** The pingpong schedule applies the LogSNR
  shift to the normalized `[1→0]` grid, then scales by σmax, so audio-to-audio
  (σmax < 1) stays monotonically decreasing while keeping all N distilled steps.
  σmax = 1.0 (text-to-audio) is bit-identical to the classic schedule.
- **SAME-L chunked decode.** The SAME-L decoder's dense sliding-window-attention
  mask is O(T²), so long clips are decoded in overlap-8 windows of 64 latent
  tokens (the throughput optimum) and stitched. SAME-S has a narrow receptive
  field and decodes whole.
- **CFG (`--cfg ≠ 1`)** combines a cond and an uncond velocity in denoised space
  (optional APG). The canonical DiT is **variable-batch**, so by default cond+uncond
  run as **one batch=2 invoke per step** (`--cfg-batched`) — ~7–29% faster on
  Apple-Silicon (the AMX matrix unit amortizes the weight loads across both rows;
  measured on an M4 Pro MacBook Pro).
  `--no-cfg-batched` falls back to a sequential batch=1 dual-pass (like the TensorRT
  release, whose engine is static-batch=1); the two are bit-identical at `fp32`/`w8a32`
  (~80 dB at `w16a32`; `w8a8-dyn` diverges by design — batch-shared activation scales).

## License & attribution

Model weights derived from Stability AI's Stable Audio 3 checkpoints.
T5Gemma text encoder from Google.

Use of the Stable Audio 3 weights is governed by the **Stability AI
Community License**. Please refer to the full terms at
<https://stability.ai/license>.
