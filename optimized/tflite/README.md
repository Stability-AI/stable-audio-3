# sa3_tflite — Stable Audio 3 on CPU via LiteRT / TFLite

Portable CPU inference for **Stable Audio 3** — the LiteRT/TFLite sibling of the
[MLX](../mlx) (Apple Silicon) and [TensorRT](../tensorRT) (NVIDIA GPU) releases.
No PyTorch, transformers, or stable-audio-tools at runtime — just `ai_edge_litert`
(LiteRT) driving fully self-contained `.tflite` graphs through the XNNPACK CPU
delegate. Runs anywhere LiteRT runs: **macOS / Linux / Windows, x86 / ARM**
(Windows is x64-only — see [Windows](#windows)).

## Quick Install

One line on a fresh machine — installs everything and plays back ~30 seconds of
"Impending tribal, epic orchestral buildup":

```bash
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash
```

Windows — stock PowerShell (no bash/git needed; the curl|bash line above requires Git Bash or WSL on Windows):

```powershell
irm https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.ps1 | iex
```

Already cloned the repo? Run from inside `optimized/tflite/`:

```bash
# macOS / Linux / Windows Git Bash
./install.sh                                              # one-time setup
./sa3 --prompt "Impending tribal, epic orchestral buildup" --play           # generates + plays
```

```bat
:: Windows cmd / PowerShell
install.bat
sa3.bat --prompt "Impending tribal, epic orchestral buildup" --play
:: PowerShell users can use .\sa3.ps1 instead of sa3.bat — same flags, and
:: Ctrl-C exits cleanly (cmd's "Terminate batch job (Y/N)?" prompt is a .bat quirk)
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

## Precision variants

Precision is split between the **DiT** and the **SAME codec** — they react to quantization oppositely:

- **DiT** (`--dit-precision`, or the global `--precision`): `fp32` (default) · `w16a32` · `w8a32` · `w8a8-dyn`
  (wXaY = weight/activation bits; "16" = fp16). int8 *fails* on the DiT — its 8-step sampler is chaotically
  sensitive, so per-step error becomes a *different* (still plausible) sample, not a noisier fp32. Judge by ear;
  **fp32 is the pick**, and on CPU it's also the fast one.
- **Codec** (`--decoder-precision` / `--encoder-precision`): `fp32` · **`w8a8` (default)**. The SAME enc/dec ship as
  static **rung** models — see [docs/RUNGS.md](docs/RUNGS.md). They run once on a fixed latent, so int8 is
  quality-free: a real-music round-trip is identical to fp32. `w8a8` is ~3× faster and ~half the RAM on an
  int8-capable CPU; `fp32` is bit-exact and the pick for CPUs *without* int8 acceleration. (The old `w16a32`/`w8a32`
  codec variants added no audible quality and are retired to `tflite/same-*/legacy/`.)

`--precision` is a global default that sets the DiT directly and maps to the codec (`fp32`→`fp32`, any int8→`w8a8`).
When omitted, the defaults are **DiT fp32, codec w8a8**. All variants keep the full feature set (any `--seconds`,
odd/even L, batched/sequential CFG) and lazy-download from HuggingFace on first use.

| component | precisions | size | notes |
|---|---|---|---|
| **DiT** | `fp32` / `w16a32` / `w8a32` / `w8a8-dyn` | sm 1.8 / 0.9 / 0.45 / 0.45 GB · medium 5.8 / 2.9 / 1.5 / 1.5 GB | int8 → a *different* sample; fp32 is the pick |
| **codec** (rung) | `fp32` / `w8a8` | SAME-S enc/dec 0.22 / 0.25 → w8a8 0.06 / 0.09 GB · SAME-L 1.7 / 1.8 → w8a8 0.47 / 0.52 GB | w8a8 quality-free, ~3× faster + half RAM; needs litert ≥ 2.2.0 |
| T5Gemma | fp16 | 0.6 GB | single-precision |

*(int8 weights are per-channel; `w8a8-dyn` also quantizes activations, its int8×int8 matmuls being what make it
faster-than-fp32 on the DiT. Codec rungs use dynamic int8 too, but there the extra activation-quant error is far
below the AE's own loss. DiT speed factors measured on an Apple M4 Pro, XNNPACK, 8 threads.)*

One CFG note (DiT-only): batched vs sequential CFG is bit-identical at `fp32`/`w8a32`, ~80 dB apart at `w16a32`, but
under `w8a8-dyn` the batch=2 invoke shares activation scales across the cond/uncond rows → a *different plausible
sample*. Pass `--no-cfg-batched` with `w8a8-dyn` when you need run-to-run reproducibility.

```bash
# default: fp32 DiT + w8a8 codec (fastest codec, quality-free)
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s

# bit-exact codec (e.g. a CPU without int8 acceleration)
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --decoder-precision fp32

# cap codec RAM on a tiny-memory device (slower)
./sa3 --prompt "lofi house loop" --dit medium --max-rung 64
```

`--dit-precision` / `--decoder-precision` / `--encoder-precision` override the global flag per component
(the DiT changes *which* sample you get; the codec's precision is quality-free on the round-trip):

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

### Windows

The stack runs natively on Windows x64 — no WSL needed. `ai-edge-litert`
ships `win_amd64` wheels for Python **3.10–3.13** (x64 only; no Windows-ARM
wheels), so install a Python in that range from
[python.org](https://www.python.org/downloads/) (check *"Add python.exe to
PATH"*), then from `optimized\tflite\`:

```bat
install.bat                          :: one-time setup (python -m venv + pip)
sa3.bat --prompt "lofi house loop" --dit sm-music --decoder same-s
```

`install.bat` / `sa3.bat` are the Windows twins of `./install.sh` / `./sa3`
(plain venv + pip; uv not required). Notes:

- **ffmpeg is optional** — only needed for mp3 / 24-bit / non-44.1 kHz
  `--init-audio` inputs: `winget install ffmpeg` (or `choco install ffmpeg`).
- **Symlinks**: downloaded weights are exposed via symlinks where possible;
  without Developer Mode, Windows disallows creating them, so the downloader
  automatically falls back to a hardlink (zero-copy) or a plain copy. No
  action needed either way.
- **Long paths**: the HuggingFace cache nests deeply — if downloads fail with
  path-length errors, enable Windows long paths once (admin PowerShell):
  `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1`
  (the `LongPathsEnabled` registry switch), then restart the terminal.
- `--play` uses the stdlib `winsound` player (macOS uses `afplay`).

## Run

`./sa3` is a thin shell wrapper around `.venv/bin/python scripts/sa3_tflite.py
"$@"` that prompts to run `./install.sh` if uv or `.venv/` isn't set up.

```bash
# Text-to-audio
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --out lofi.wav

# Sound effects
./sa3 --prompt "footsteps on gravel" --dit sm-sfx --decoder same-s --out steps.wav

# Higher-quality music (medium DiT + SAME-L codec)
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

# Generate + play immediately (afplay/winsound/aplay; Ctrl-C stops both)
./sa3 --prompt "rainforest" --dit sm-sfx --decoder same-s --play

# All options + categorised examples
./sa3 --help
```

Omit `--dit` / `--decoder` for an interactive arrow-key picker. Omit
`--prompt` for a stdin prompt. Relative `--out` paths land in `output/`
(auto-created); absolute paths are honoured as-is. The output path is
printed prominently as a `▸ saved` line at the end of each run.

Use `--threads` to control the XNNPACK CPU thread count (default 8).

## Web UI (gradio)

A browser UI over the same TFLite backend — every generation mode plus a
spectrogram player, a history panel, and playback options (auto-play, infinite
radio with pre-generation, loop, hotswap, MP3/WAV save).

```bash
./sa3-gradio                       # sm-music, opens a public share link
./sa3-gradio --dit medium          # start on the medium DiT
./sa3-gradio --no-share            # local-only (http://localhost:7860)
./sa3-gradio --port 7871           # pick the port
```

`./sa3-gradio` is a wrapper around `.venv/bin/python scripts/sa3_gradio.py`; on
first run it offers to install the UI-only extras (`gradio`, `pillow`,
`soundfile`) into `.venv`. In the browser:

- **Precision** dropdown next to the model picker — `fp32` (default) / `w16a32`
  / `w8a32` / `w8a8-dyn`, applied to the DiT; the codec maps it to its rung tier
  (`fp32` → fp32, any int8 → w8a8).
- **LoRA** panel — upload a `.safetensors` adapter or pick one from
  `loras/<model>/` (e.g. `loras/sa3-medium/`), set its strength, stack several;
  settings are remembered per model. The panel is disabled under a quantized
  precision (see the LoRA note above) and has no per-step range (frozen graph).
- **Audio-to-audio / inpainting** — upload a reference clip; inpainting adds a
  start/end range. Both are combinable.

Generated WAV/MP3s land in `output/gradio/`. First use of a model/precision
loads its weights (cached after).

### Without the wrapper

```bash
.venv/bin/python scripts/sa3_tflite.py --prompt "..." --dit medium --decoder same-l
# or, after `source .venv/bin/activate`:
python scripts/sa3_tflite.py --prompt "..." --dit medium --decoder same-l
```

## Speed & memory

This is a **CPU** path — it trades the GPU releases' speed for portability. The
small models comfortably beat realtime on a modern laptop CPU; `medium` is slower
(its DiT is ~5.8 GB fp32). Use ≥ ~20 s clips: very
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
| `--lora`              | —        | A `.safetensors` LoRA adapter (SA3-native/underfit or PEFT) merged into the DiT, optional `strength=S`; repeat to stack. Requires `--dit-precision fp32` or `w16a32`. See [LoRA](#lora) |
| `--lora-strength`     | 1.0      | Default strength for `--lora` adapters without their own `strength=`  |
| `--init-audio`        | —        | WAV (any format via ffmpeg) input for audio-to-audio / inpaint       |
| `--init-noise-level`  | 1.0      | σmax; 0.4–0.8 typical for variation, 1.0 = full regen, >1 = overshoot |
| `--inpaint-range`     | —        | `START,END` seconds; regenerate that span, keep the rest              |
| `--threads`           | 8        | XNNPACK CPU threads (all TFLite models run on CPU)                    |
| `--free-models`       | on       | Free each model after its last use; `--no-free-models` keeps them resident |
| `--out` / `-o`        | (auto)   | Relative → `output/<file>`; absolute → as-is. 16-bit PCM stereo @ 44.1 kHz, trimmed to exactly `--seconds` |
| `--play`              | off      | After writing, play the WAV: `afplay` (macOS) / `winsound` (Windows) / `aplay` (Linux); Ctrl-C stops both |

All `.tflite` models are **fp32** except T5Gemma, which is **fp16** (numerically
lossless there). There is no dtype knob: on CPU, int8/fp16 weights buy size, not
speed (XNNPACK dequantizes to fp32 to matmul), and int8 costs quality on the DiT
— so this release ships the fp32 graphs directly. (See "Notes on the design".)

## LoRA

Apply a LoRA finetune by merging it into the DiT's weight buffers at load —
same adapters and semantics as the MLX backend's `--lora`:

```bash
# one adapter
./sa3 --dit medium --decoder same-l --prompt "progressive metal" \
      --lora myband.safetensors

# per-adapter strength; stack several
./sa3 --dit medium --decoder same-l --prompt "..." \
      --lora a.safetensors strength=0.8 --lora b.safetensors strength=0.5
```

The adapter is a `.safetensors` file (SA3-native `train_lora.py` / underfit
output, or a PEFT adapter directory) — pickle `.ckpt/.pt` is refused. All nine
adapter types (lora, dora-rows/cols, bora, the four -xs) are supported; the base
must match `--dit`. The merge is written into a cached copy of the DiT under
`models/tflite/lora_cache/` (keyed by adapter + strength content hash) and reused
on repeat runs, so the ~5–15 s patch cost is paid once. A medium fp32 cache entry
is ~5.4 GB — delete `lora_cache/` to reclaim.

Requires `--dit-precision fp32` (default) or `w16a32`. **LoRA on the quantized
DiTs (`w8a32` / `w8a8-dyn` / `w4a32`) isn't figured out yet** — those store
weights as GPTQ-calibrated int8, so merging would mean dequantize → add the LoRA
delta → re-quantize, and a naive re-quant throws away the GPTQ error-feedback
grid (you'd get a model that's both LoRA-adapted *and* degraded). Doing it well
needs a GPTQ pass over the merged weights; until then `--lora` refuses quantized
precisions (the CLI errors, the web UI disables the LoRA panel). Use fp32/w16a32
for LoRA — on CPU fp32 is the fast-and-accurate choice anyway, so this is rarely
a real constraint. **Per-step gating (`steps=`) is also MLX-only** — a frozen
TFLite graph merges weights once at load; use `optimized/mlx` for step-gated LoRA.

## Files

```
sa3_tflite/
├── sa3                            ← shell wrapper (use this)
├── install.sh                     ← uv bootstrap (run once)
├── sa3.bat / install.bat          ← Windows twins of sa3 / install.sh
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
        ├── same-s/{enc,dec}_{fp32,w8a8}.tflite   rung codec (fp32 ~0.22 GB · w8a8 ~0.07 GB); legacy/ = pre-rung
        └── same-l/{enc,dec}_{fp32,w8a8}.tflite   rung codec (fp32 ~1.75 GB · w8a8 ~0.5 GB);  legacy/ = pre-rung
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
- **SAME codec = static rung models.** Both SAME encoders and decoders are static
  **rung** `.tflite` (a ladder of fixed-size subgraphs sharing one copy of the weights,
  litert ≥ 2.2.0 `CompiledModel` + weight cache). `RungEncoder`/`RungDecoder`
  auto-dispatch the optimal rung per length and tile bit-exactly — no chunk-size knob,
  RAM flat. This replaced the dynamic-varlen graphs whose RAM blew up (SAME-L O(S²) →
  16 GB@256; SAME-S → ~50 GB@8192). See [docs/RUNGS.md](docs/RUNGS.md).
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
