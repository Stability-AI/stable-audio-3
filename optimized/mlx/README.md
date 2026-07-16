# sa3_mlx — Stable Audio 3 in pure MLX

Apple-Silicon-native inference for **Stable Audio 3**, with no PyTorch,
transformers, or stable-audio-tools at runtime.

## Quick Install

One line on a fresh Apple Silicon Mac — installs everything and plays
back ~2 minutes of "Impending tribal, epic orchestral buildup":

```bash
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh | bash
```

Already cloned the repo? Run from inside `optimized/mlx/`:

```bash
./install.sh                                              # one-time setup
./sa3 --prompt "Impending tribal, epic orchestral buildup" --play           # generates + plays
```

## Three models, three modes

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
> `./sa3 --prompt …` — any missing weight file is downloaded from HF on
> first use and symlinked into `models/mlx/` from the HuggingFace cache.

Apple Silicon only (MLX is Metal-backed). Python 3.10+. `./install.sh
--python 3.12` to pin a different Python.

## Run

`./sa3` is a thin shell wrapper around `uv run python scripts/sa3_mlx.py
"$@"` that prompts to run `./install.sh` if uv or `.venv/` isn't set up.

```bash
# Text-to-audio
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --out lofi.wav

# Sound effects
./sa3 --prompt "footsteps on gravel" --dit sm-sfx --decoder same-s --out steps.wav

# Higher-quality music (medium DiT)
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

# Apply a LoRA finetune (merged into the DiT at load; base must match --dit)
./sa3 --prompt "arabic maqam oud taqsim" --dit medium --decoder same-l \
      --lora ./my_lora.safetensors --lora-strength 1.0 --out maqam.wav

# Per-LoRA strength + sampling-step gating (skipping step 1 often helps).
# steps= is 1-based inclusive: 2-8, 2- (2..last), -4 (1..4), 3 (just step 3).
# Step-gated adapters are re-merged in place at the few step boundaries
# (~80 ms each on medium) — no extra memory, every step runs at full speed.
./sa3 --prompt "progressive metal instrumental" --dit medium --decoder same-l \
      --lora plini.safetensors strength=0.8 steps=2- \
      --lora rain_texture.safetensors steps=-4 --out prog.wav

# Generate + play immediately (afplay; Ctrl-C stops both)
./sa3 --prompt "rainforest" --dit sm-sfx --decoder same-s --play

# All options + categorised examples
./sa3 --help
```

Omit `--dit` / `--decoder` for an interactive arrow-key picker. Omit
`--prompt` for a stdin prompt. Relative `--out` paths land in `output/`
(auto-created); absolute paths are honoured as-is. The output path is
printed prominently as a `▸ saved` line at the end of each run.

### Without the wrapper

```bash
uv run python scripts/sa3_mlx.py --prompt "..." --dit medium --decoder same-l
# or, after `source .venv/bin/activate`:
python scripts/sa3_mlx.py --prompt "..." --dit medium --decoder same-l
```

## Web UI (gradio)

```bash
./sa3-gradio                  # public gradio.live share link by default
./sa3-gradio --no-share       # local-only (http://127.0.0.1:7860)
./sa3-gradio --dit medium     # initial model (switchable in the UI)
```

Every generation mode is wired: text-to-audio, CFG 0–10 (0 = the negative prompt
takes over, 0.5 = halfway between both prompts, >1 = extrapolate), audio-to-audio
(guide audio + σmax slider), and inpainting (separate reference audio + start/end
range sliders) — a2a and inpainting combine, so an inpainted span can regenerate
from the guide audio instead of noise. Each clip renders with a 3-band tinted stereo mel
spectrogram (bass=red / mid=green / high=blue). Models hot-swap from the
dropdowns and cache in unified memory, so switching back is instant; WAVs land
in `output/gradio/`. The UI needs two extra packages (`gradio`, `pillow`) —
`./sa3-gradio` offers to install them into `.venv` on first run.

## Speed & memory

Measured on **M1 8 GB** (the slowest M-chip). Newer chips are faster —
roughly **1.2–1.4× on M2**, **1.5–2× on M3**, **2–3× on M4**, especially
on the memory-bandwidth-bound seq=1 attention path.

| `--dit`               | Wall (10 s clip) | × realtime | Peak RAM |
|-----------------------|------------------|------------|----------|
| `sm-music` / `sm-sfx` | ~1 s             | ~10×       | 1.6 GB   |
| `medium`              | ~5 s             | ~2×        | 3.8 GB   |

Peak RAM assumes `--free-models` (default on) — T5Gemma is freed after
conditioning and the DiT after sampling, so the decoder never competes
with the upstream models for memory. 8 GB M1 is fine for everything;
disable with `--no-free-models` only if you have headroom and plan to
batch multiple runs in one process.

### Benchmark yourself

Reproduce the table above on your own machine — sweeps both models across
4 clip lengths (5 s / 30 s / 120 s / 380 s) and prints an ASCII summary.
Each cell runs in its own subprocess so peak RAM is measured cleanly per
run; weights are pre-warmed via `ensure_local()` so HF download time
doesn't pollute the timings. Takes ~3-10 min depending on your chip.

```bash
uv run --no-project python scripts/benchmark.py
```

Sample run on **M4 Pro / 48 GB**:

```
┌───────────┬─────────┬─────────┬───────────┬───────────┬────────────┐
│ model     │ decoder │ seconds │  wall (s) │ ×realtime │   peak RAM │
├───────────┼─────────┼─────────┼───────────┼───────────┼────────────┤
│ sm-music  │ same-s  │       5 │      0.80 │     6.27× │    1.62 GB │
│ sm-music  │ same-s  │      30 │      1.53 │    19.61× │    1.94 GB │
│ sm-music  │ same-s  │     120 │      4.12 │    29.13× │    2.38 GB │
│ sm-music  │ same-s  │     380 │     13.46 │    28.24× │    2.58 GB │
├───────────┼─────────┼─────────┼───────────┼───────────┼────────────┤
│ medium    │ same-l  │       5 │      2.20 │     2.27× │    3.82 GB │
│ medium    │ same-l  │      30 │      5.06 │     5.92× │    3.89 GB │
│ medium    │ same-l  │     120 │     14.68 │     8.17× │    5.21 GB │
│ medium    │ same-l  │     380 │     47.78 │     7.95× │    5.05 GB │
└───────────┴─────────┴─────────┴───────────┴───────────┴────────────┘
```

## LoRA training

Finetune SA3 on your own audio, entirely in MLX (no PyTorch) — a full training
CLI that replicates [underfit](https://github.com/dada-bots/underfit)'s
conventions, defaults, and checkpoint format (so it can run as underfit's
Apple-Silicon backend). Three steps: **pre-encode → train → generate**. See
`TRAINING_CONVENTIONS.md` for the complete convention inventory and the
torch(MPS)-vs-MLX forward+backward parity results.

**1. Pre-encode** your audio to SAME latents (once, offline — torch-free):

```bash
uv run python scripts/pre_encode_mlx.py \
    --audio-dir ~/my-clips --output-dir ~/my-latents --codec same-s
```

Each file becomes `<stem>.npy` latents `[D, T]` + a `<stem>.json` sidecar
(duration, padding mask, tags) — the exact format underfit's dataset uses.
Use `same-s` for sm-music/sm-sfx, `same-l` for medium.

**2. Train.** Training uses the **BASE** checkpoint (`stabilityai/stable-audio-3-*-base`,
rectified_flow), *not* the shipped ARC weights inference uses. The base-model npz is
**auto-downloaded** from HuggingFace on first run — no flag needed (override with
`--dit-weights <path>`; regenerate one yourself from a `-base` repo with
`scripts/export_base_npz.py`):

```bash
uv run python scripts/lora_train_mlx.py \
    --dit sm-music --latents-dir ~/my-latents --lr 1e-4 --name my-lora \
    --adapter-type dora-rows --rank 16 --max-steps 2000
```

Defaults mirror underfit (dora-rows / rank 16, AdamW, uniform sampler + the
"full" distribution shift, CFG-dropout 0.1, signal-only masked rectified-flow
loss, checkpoint every 1000 steps). To match the full dashboard template config,
add `--timestep-sampler trunc_logit_normal --use-effective-length --beta2 0.95
--weight-decay 0.01 --lr-scheduler inverse --lr-warmup 0.995`. Checkpoints land at
`output/runs/<name>/<uuid>/checkpoints/<name>-step=S-epoch=E.safetensors` with
`lora_config` metadata; resume with `--lora-ckpt-path <ckpt>`.

Optional training-time demos (RF-Euler inference → mp3, underfit's on-disk
format): `--demo-every 1000 --demo-config demos.json`, where `demos.json` is a
list of `{"prompt", "cfg", "seed", "steps", "lora_strength"?, "lora_interval_max"?}`.

**3. Generate** with your adapter — the checkpoint applies to the shipped ARC
model at inference via `--lora` (see "Apply a LoRA finetune" above; also
`scripts/sa3_gradio.py --lora <ckpt>` to open the web UI with it preloaded):

```bash
./sa3 --dit sm-music --decoder same-s --prompt "..." --out finetuned.wav \
      --lora "output/runs/my-lora/"*"/checkpoints/my-lora-step=2000-epoch="*.safetensors
```

### Building your own loop

The CLI is assembled from reusable pieces: `inject_trainable_lora` /
`save_lora_checkpoint` / `apply_lora_checkpoint` (`models/defs/lora.py` — all 9
LoRA/DoRA/BoRA/-XS types with the no-weight-materialization forward), the RF
loss + samplers + distribution shift (`models/defs/training.py`), the
pre-encoded dataset + prompt templates (`models/defs/latent_dataset.py`), and
`encode_audio` (`models/defs/audio_encoding.py`). Checkpoints use the same
safetensors keys + `lora_config` metadata as the PyTorch trainer, so adapters
are interchangeable in either direction.

## Flag reference

| Flag                  | Default  | Notes                                                                 |
|-----------------------|----------|-----------------------------------------------------------------------|
| `--prompt`            | (asks)   | Text prompt; empty string = unconditional                              |
| `--negative-prompt`   | —        | CFG uncond branch; only used when `--cfg ≠ 1.0`                       |
| `--dit`               | (asks)   | `sm-music`, `sm-sfx`, or `medium`                                     |
| `--decoder`           | (asks)   | `same-s` (pairs with sm-*) or `same-l` (pairs with medium)            |
| `--seconds`           | 30       | Output length                                                         |
| `--steps`             | 8        | Pingpong sampler steps; 1 = single forward (fastest), 8 = sweet spot  |
| `--seed`              | random   | Set for reproducibility; the chosen seed is printed at the end        |
| `--cfg`               | 1.0      | Guidance scale; 1.0 = off, >1 toward prompt, <1 toward uncond         |
| `--apg`               | 1.0      | Adaptive Projected Guidance; only matters when `--cfg ≠ 1`            |
| `--init-audio`        | —        | WAV (44.1 kHz, 16-bit PCM) input for audio-to-audio / inpaint         |
| `--init-noise-level`  | 1.0      | σmax; 0.4–0.8 typical for variation, 1.0 = full regen, >1 = overshoot |
| `--inpaint-range`     | —        | `START,END` seconds; regenerate that span, keep the rest              |
| `--dit-dtype`         | fp16     | DiT compute dtype (decoder always FP32; T5Gemma always fp16)          |
| `--lora`              | —        | A `.safetensors` LoRA adapter (SA3-native/underfit or PEFT) with optional `strength=S` and `steps=MIN-MAX` tokens; repeat the flag to stack adapters. Full-range adapters merge at load; step-gated ones re-merge in place at step boundaries. Pickle `.ckpt/.pt` is refused. Base must match `--dit` |
| `--lora-strength`     | 1.0      | Default strength for adapters without their own `strength=`; 0 = bit-exact bypass, >1 amplifies |
| `--free-models`       | on       | Progressive model freeing; `--no-free-models` keeps them resident     |
| `--out`               | out.wav  | Relative → `output/<file>`; absolute → as-is. 16-bit PCM stereo @ 44.1 kHz, trimmed to exactly `--seconds` |
| `--play`              | off      | After writing, play via `afplay`; Ctrl-C stops both processes         |

## Files

```
sa3_mlx/
├── sa3                            ← shell wrapper (use this)
├── install.sh                     ← uv bootstrap (run once)
├── README.md
├── TRAINING_CONVENTIONS.md         ← LoRA-training conventions + MPS-vs-MLX parity
├── requirements.txt
├── output/                        ← default landing zone for generated WAVs
├── scripts/
│   ├── sa3_mlx.py                 ← orchestrator CLI (invoked by ./sa3)
│   ├── weights.py                 ← weights manifest + HF auto-download
│   ├── examples.py                ← shared examples block (--help + post-install)
│   ├── install.py                 ← install.sh's Python half (bundle picker)
│   ├── test_all_configs.py        ← npz + CLI config sanity tests
│   ├── benchmark.py               ← wall-time + peak-RAM matrix across model × duration
│   ├── lora_train_mlx.py          ← LoRA training CLI (underfit conventions)
│   ├── pre_encode_mlx.py          ← audio → SAME-latent pre-encode (torch-free)
│   ├── sa3_gradio.py              ← web UI (invoked by ./sa3-gradio; --lora preload)
│   ├── test_lora_merge.py         ← adapter-math tests (weight-free; run in CI)
│   └── parity_forward_{torch,mlx}.py  ← torch(MPS)-vs-MLX fwd+bwd parity harness
└── models/
    ├── defs/
    │   ├── sa3_pipeline.py        ← sampler + conditioner + unpatch
    │   ├── t5gemma_mlx.py         ← T5Gemma encoder + SentencePiece wrapper
    │   ├── dit_mlx.py             ← small DiT (sm-music + sm-sfx)
    │   ├── dit_mlx_medium.py      ← medium DiT (differential attention)
    │   ├── same_s_{encoder,decoder}.py    ← small codec
    │   ├── same_l_{encoder,decoder}.py    ← large codec
    │   ├── lora.py                ← trainable LoRA/DoRA/BoRA adapters (9 types)
    │   ├── lora_merge.py          ← inference-time LoRA merge + per-step gating
    │   ├── training.py            ← RF loss + timestep samplers + distribution shift
    │   ├── latent_dataset.py      ← pre-encoded dataset + underfit prompt templates
    │   ├── audio_encoding.py      ← waveform → SAME latents (pre-encode bridge)
    │   └── demo_mlx.py            ← training-time RF-Euler demos → mp3
    └── mlx/                       ← .npz weights (auto-downloaded; ~8.4 GB total)
        ├── t5gemma_f16.npz                541 MB    text encoder + tokenizer
        ├── dit_sm-music_f16.npz           877 MB    DiT + conditioner baked in
        ├── dit_sm-sfx_f16.npz             877 MB    DiT + conditioner baked in
        ├── dit_medium_f16.npz             2.77 GB   DiT + conditioner baked in
        ├── same_s_{encoder,decoder}_f32.npz  ~210 MB each    shared sm-* codec
        └── same_l_{encoder,decoder}_f32.npz  ~1.7 GB each    medium codec
```

Each DiT `.npz` bundles its small conditioner under a `cond.*` key prefix
(learned `padding_embedding` + a tiny SecondsTotalEmbedder), so the
orchestrator only loads one file per model. The two small DiTs share the
same SAME-S codec weights — they're bit-exact between checkpoints, so
only one set of small-codec npz files is shipped.

## Notes on the design

- **Mixed precision**: DiT runs FP16 (validated transparent at ~50–57 dB
  PSNR vs FP32), the audio decoder always runs FP32 (SAME-S's differential
  attention catastrophically cancels in FP16), T5Gemma always runs FP16.
  `--dit-dtype fp32` reverts to all-FP32 for bit-exact reproducibility.
- **Padding embedding**: empty / short prompts don't see raw zeros for
  pad positions — they see a learned 768-dim `cond.padding_embedding`
  baked into each DiT npz. This is what makes "" (empty prompt) produce
  sensible unconditional audio instead of noise.
- **Auto-download**: weights aren't bundled — they're pulled from
  `stabilityai/stable-audio-3-optimized` on HF on first use and symlinked
  into `models/mlx/` from the HF cache. No duplication.
- **Chunked decoding**: SAME-L can't fit a long clip into one decode pass
  on 8 GB; the decoder splits into overlap-2 (SAME-S) or overlap-8 (SAME-L)
  chunks with a uniform-kernel edge treatment that stays bit-exact vs an
  un-chunked reference at the natural sizes.

## License & attribution

Model weights derived from Stability AI's Stable Audio 3 checkpoints.
T5Gemma text encoder from Google.

Use of the Stable Audio 3 weights is governed by the **Stability AI
Community License**. Please refer to the full terms at
<https://stability.ai/license>.
