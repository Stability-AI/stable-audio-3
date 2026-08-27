# sa3_trt — Stable Audio 3 on TensorRT

NVIDIA-native inference for **Stable Audio 3**. Full pipeline (T5 + DiT
8-step pingpong + decoder + narrow + DtoH) captured as one CUDA graph;
**~30 ms / 30 s clip** on H100 at sm-music + same-s.

## Quick Install

One line on a fresh Linux + NVIDIA box — installs everything and plays back
~2 minutes of "Death Metal":

```bash
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tensorRT/bootstrap.sh | bash
```

Already cloned? Run from inside `optimized/tensorRT/`:

```bash
./install.sh                        # one-time setup
./sa3 --prompt "Death Metal"        # generate
```

`bootstrap.sh` auto-installs `git` via the local package manager
(apt/dnf/yum/apk/pacman/zypper); falls back to a `curl + tar` download if
that fails. `install.sh` is arch-aware — it queries HF for a matching
`tensorRT/sm_*/` engine set and downloads it; if no prebuilt engines exist
for your GPU, it offers to compile fresh ones from the canonical ONNX
hosted alongside on HF (~30 min one-time build).

## Three models, three modes

| `--dit`    | model                     | best for                       |
|------------|---------------------------|--------------------------------|
| `sm-music` | sa3-sm-music (50 M block) | fast music generation          |
| `sm-sfx`   | sa3-sm-sfx   (50 M block) | sound effects                  |
| `medium`   | sa3-medium   (1.4 B)      | higher-quality music, slower   |

| mode             | flags                                                 | example                          |
|------------------|-------------------------------------------------------|----------------------------------|
| text-to-audio    | `--prompt P`                                          | new clip from a description      |
| audio-to-audio   | `--prompt P --init-audio IN.wav --init-noise-level σ` | variation of an existing clip    |
| inpainting       | `--prompt P --init-audio IN.wav --inpaint-range "S,E"`| regenerate one section, keep rest|
| CFG + negative   | `--cfg 3.0 --negative-prompt P_NEG`                   | steer toward / away from prompts |

```
prompt ─▶ T5Gemma encoder ─▶ DiT pingpong sampler ─▶ SAME-S/L decoder ─▶ WAV
                                       ▲
                  optional: encoder + init audio (audio-to-audio / inpaint)
```

The text-to-audio fast path captures all five stages in a single CUDA graph.
CFG and inpainting fall back to an eager Python sampler (still TRT-accelerated
per-stage, just not graph-fused).

## Install

```bash
./install.sh                  # interactive engine picker
./install.sh -y               # unattended: --engines all
./install.sh --engines medium # medium DiT + SAME-L (highest quality)
```

`install.sh` is uv-based. On a fresh machine it will:

1. Install [uv](https://github.com/astral-sh/uv) via the official curl
   installer if it's missing.
2. Create a project-local `.venv/` and `uv pip install -r requirements.txt`
   (TensorRT 10.15.1.29 pinned, torch nightly, triton).
3. Detect your GPU's compute capability → `sm_<cc>`, then query HF for a
   matching engine set.
4. Download the bundle(s) you ask for (medium / sm-music / sm-sfx / all)
   into `models/sm_<cc>/`. The T5Gemma tokenizer ships in-repo (it's
   arch-agnostic).
5. If your arch has no prebuilt engines on HF, you'll be offered:
   - **[B]** Build fresh engines from ONNX (~30 min, recommended)
   - **[D]** Download a non-matching arch anyway (engine may not load)
   - **[S]** Skip — build/download manually later

End-to-end on a fresh machine with prebuilt engines: **~3 min** (mostly
package install + a ~5 GB engine download).

## Run

`./sa3` is a thin wrapper that invokes the project venv's Python on
`scripts/sa3_trt.py` with your args:

```bash
# Text-to-audio
./sa3 --prompt "lofi house loop" --dit sm-music --decoder same-s --out lofi.wav

# Higher-quality music
./sa3 --prompt "A beautiful piano arpeggio grows into a cinematic climax" \
      --dit medium --decoder same-l --seconds 30 --out piano.wav

# Sound effects
./sa3 --prompt "footsteps on gravel" --dit sm-sfx --decoder same-s --out steps.wav

# Audio-to-audio variation (σmax 0.4–0.8 typical)
./sa3 --prompt "jazz fusion with electric piano" --dit sm-music --decoder same-s \
      --init-audio funk.wav --init-noise-level 0.7 --out funk_jazz.wav

# Inpaint seconds 4-7
./sa3 --prompt "explosive drum break" --dit sm-music --decoder same-s \
      --init-audio funk.wav --inpaint-range "4,7" --out funk_drums.wav

# CFG + negative prompt
./sa3 --prompt "ambient drone" --cfg 3.0 --negative-prompt "drums, vocals" \
      --dit sm-music --decoder same-s --out drone.wav

# Short-form (DiT engine supports L=1..4096 = ~93 ms .. ~6.3 min output)
./sa3 --prompt "kick drum hit" --seconds 1 --dit sm-music --decoder same-s

# All flags + examples
./sa3 --help
```

Omit `--dit` / `--decoder` for an interactive arrow-key picker. Relative
`--out` paths land in `output/`; absolute paths are honoured as-is.

### DiT precision (`--precision`)

`fp16` is the default for every model — no flag needed for the recommended setup:

| model            | default     | why                                                        |
|------------------|-------------|------------------------------------------------------------|
| `medium`         | `fp16` | FMHA-fused (96 fused attention nodes) **and** fp32-accurate at every length |
| `sm-music`/`sm-sfx` | `fp16` | standard attention — already fuses in fp16           |

`--precision` also takes `fp8` (all DiTs) and `fp32` explicitly:

- **`fp16`** — canonical **and default for every DiT** (formerly `fp16mixed`; every
  tier is mixed-precision, so the qualifier was dropped). FP16 trunk, FP32 islands
  around RMSNorm and RoPE generation, and an FP16 attention core (QK^T → Softmax →
  P·V) so TRT's FMHA fuser fires. Teacher-forced velocity cos **1.0000** vs the FP32
  engine at every length. Bit-reproducible run to run.
  <br>Medium used to default to a uniform-`bf16` engine because this engine's
  attention core was stuck in FP32 and therefore unfused — 0 fused MHA nodes.
  Bounding the RoPE island before QK^T fixed that (**4.3× faster at L=4096**), which
  retired the `bf16` tier entirely (it also drifted at long sequence — see below).
  Engines built before 2026-07 are the slow variant; rebuild with
  `build_from_onnx.py sa3-m`.
- **`fp8`** — *medium: the max-speed clean tier, calibrated; sm-music/sm-sfx: a clean
  weight-halving tier (see the end of this bullet).* On **medium**: fp8 E4M3 on the 176
  linear GEMMs + bf16 fused FMHA (96 nodes) + a **baked fp32 RoPE constant table** (position
  cos/sin computed host-side at build and frozen as a graph constant — no in-graph trig,
  so the island is precision-policy-robust and cross-runtime-stable). **~1.3× faster than
  `fp16` at every length** (H200, same-run round-robin: 1.40× @L129 / 1.32× @L1292 /
  1.34× @L4092 — also ahead of `bf16`) and **clean at long sequence** (latent std 0.86 vs
  eager 0.95, **0.000% clip** at 2-min and 6-min), so it stays clean exactly where `bf16`
  clips. The fp8 scales are **calibrated on real conditioning** (per-tensor activation amax +
  per-channel weight scales from @ryanontheinside's #47, captured via `make_calib.py`) — a
  **speed-free** change (31.2 vs the earlier uncalibrated 30.8 ms/fwd) that lifts worst-step
  velocity-cos vs fp32 on adversarial seeds **0.52/0.57/0.64 → 0.92/0.94/0.92** (steps 1–7
  match the fully-calibrated reference within ~0.001). It remains a **speed tier over an
  already-good default, not a fidelity upgrade over it**: single-step velocity cos ~0.92–0.94
  (below fp16's ~1.0) but the 8-step render stays coherent — it just no longer collapses
  at the highest-noise first step the way the uncalibrated engine did. **Capped at L≤4096** —
  the baked table is sized to the profile max, which is *also* the SAME-L decoder's own cap,
  so this is not a new end-to-end limit; longer renders are rejected (a re-bake would be
  needed). **Not seed-reproducible vs fp16.** Built weakly-typed `EXPLICIT_BATCH` +
  `BF16` + `FP8` + `OBEY_PRECISION_CONSTRAINTS` (rebuild: `build_from_onnx.py sa3-m-fp8`;
  producer: `build/build_dit_bf16.py` RoPE-baker + `build/transplant_scales.py` calibrated-scale
  transplant; identity check: `scripts/verify_fp8_rope.py`; calibration by @ryanontheinside,
  [#47](https://github.com/Stability-AI/stable-audio-3/pull/47)).
  <br>On **sm-music / sm-sfx** fp8 is a **different, simpler recipe** — fp8 E4M3 grafted onto
  the linear GEMMs of the fp16 graph (attention stays fp16-fused, fp32 islands untouched;
  no baked RoPE — these DiTs never had bf16's long-angle problem). It's a **clean weight-halving
  tier** (engine 479 vs 936 MB, velocity-cos **~0.99** vs eager, clip% at/below fp16) that
  is only **marginally faster (~1.1×)**: a small DiT's ~5 ms forward at batch 1 is overhead-bound,
  so fp8's GEMM savings barely show. Default stays fp16. Rebuild: `build_from_onnx.py
  sa3-sm-music-fp8` / `sa3-sm-sfx-fp8`; producer: `build/make_dit_fp8_smalldit.py`. Not
  seed-reproducible vs fp16.
- **`fp32`** — pure FP32, bit-equivalent to PyTorch eager (~2× slower, ~2× VRAM).

> **Retired: `bf16` (was medium-only).** A uniform-bf16 trunk was ~3% faster than
> `fp16` but **drifted at long sequence** — weakly-typed BF16 evaluated RoPE's
> rotation angle in bf16, which reaches ~4155 rad at L=4092 where bf16's spacing is
> 32 rad (> 2π), destroying position info for the fast-rotating dims (latent inflates
> ~2.5×, a 6-min render clips 2–3% of samples). `fp16` matches its FMHA fusion
> without the drift, so `bf16` was removed. `--precision bf16` (and the old
> `--precision fp16mixed`) now silently alias to `fp16`.

```bash
# medium defaults to fp16. fp8 is the max-speed clean tier (~1.3x faster at
# every length, clean at long sequence):
./sa3 --prompt "..." --dit medium --decoder same-l --precision fp8
```

The TRT DiT engines are static batch=1 (the ONNX bakes batch=1), so CFG runs as a
sequential cond+uncond dual-pass at batch=1 for every precision.

### SAME-L attention kernel

`--decoder same-l` uses a custom sliding-window-attention plugin. Prebuilt engines embed
an ahead-of-time PTX kernel, so nothing re-enters Python during inference and the engine
goes inside the full-pipeline CUDA graph. This matters on **sm_120**, where the older
JIT-dispatched build is not stream-capturable: the decode is silently dropped from the
graph and every render comes back as the same wash of noise, with exit code 0.

If you see byte-identical output across seeds, you are on an affected engine. Re-pull it
(the published `sm_120` engines were rebuilt 2026-07-31), or run with `--no-mega-graph` to
bypass capture. The `sm_90` engines are unaffected and still ship the JIT kernel, which is
why **Triton is required at inference on sm_90 but not on sm_120**. Building your own is
covered under "Choosing the SAME-L attention kernel" in
[`build/README.md`](build/README.md).

### Chunkable engines (canonical, both decoders)

`--decoder same-l` now resolves to `dec_fp16_chunkable_limiter.trt` / `enc_fp16_chunkable.trt`,
and `--decoder same-s` to `dec_bf16_chunkable_limiter.trt` / `enc_bf16_chunkable.trt`, with
`--dec-precision fp8` selecting SAME-L's fp8 pair. (SAME-S is **bf16** — SAME-L is the fp16 one.)
Two things changed.

**Two optimization profiles per engine.** TensorRT commits a context's scratch at
`create_execution_context()` — sized from the profile ceiling, *before any shape is bound* — so the
old single-profile decoder reserved **8143 MB whether it decoded five seconds or six minutes**. The
new engines carry a low band (decoder 256 latents, encoder 64) and a wide band (4096):

| | scratch | decode |
|---|---|---|
| `--chunking` (default), SAME-L | **509 MB** decoder, **521 MB** encoder | 256-latent windows |
| `--chunking` (default), SAME-S | **363 MB** decoder, **365 MB** encoder | 256-latent windows |
| `--no-chunking` | 8143 / 8330 MB (SAME-L), 5780 / 5800 (SAME-S) | one single-shot call |

Chunking is a **memory** mechanism, not a speed one. It wins on time only at exactly L=256, where
the whole request is one window; above that, single-shot is 10–20% faster. Whole-pipeline resident
VRAM for a six-minute render drops from ~21.5 GB to ~3.9 GB.

**A limiter with a runtime ceiling.** The decoder emits already-limited PCM (5.8 ms window,
sample-peak, ceiling 0.977 = −0.2021 dBFS) in place of the old hard clip. `--limiter-ceiling`
changes it per call with no rebuild and no recapture; a large value bypasses it and reproduces the
old behaviour exactly. ⚠ **This changes the audio.** To reproduce renders made before it landed,
use `--dec-precision legacy`, which resolves to the previous engines — still published, nothing was
moved.

The low band's ceiling is 256 latents on both decoders and both encoders. It is a real dial:
raising it costs scratch and cuts the window count, lowering it does the reverse. On SAME-S the
choice carries no quality axis at all — windowed decode there is *bit-identical* to single-shot,
because its receptive field fits inside the 16-latent trim. On SAME-L it does: encode accuracy
improves monotonically with window width, which is why the encoder's low band is 256 and not the
64 it originally shipped with (1.3&ndash;1.5&times; faster and cos 0.99969 against 0.99831).

Also worth knowing when building your own: a profile is `(min, opt, max)`, and the two ends do
different jobs. **`max` sets VRAM; `opt` sets which shapes get the good kernels** and costs nothing.
The inherited default of `min(1292, ceiling)` — 1292 latents being exactly 120.0 s — costs 17–23%
at short L on the wide band while buying nothing at long L. See
[`build/build_samel_chunkable.py`](build/build_samel_chunkable.py) for the tuned values.

⚠ The old `same-l/enc_fp8.trt` was **removed** from the model repo. Its activation quantisers were
calibrated with a doubly-conservative percentile plus a `1e-4` floor, putting the clip points at
0.04–1.22 where the decoder's plain `amax` puts them at 22.4; that cost up to 2.16 dB of round trip
on clean acoustic material. `enc_fp8_chunkable.trt` is the same weights recalibrated
([`quantize/recalib_enc_fp8.py`](quantize/recalib_enc_fp8.py)) and measures 30.9 dB of latent SNR
where the old one measured 18.3.

## Speed & memory

Measured on **H100 SXM 80 GB** at `--steps 8` (rf-denoiser sweet spot).
Numbers are end-to-end Inference (T5 + DiT + decoder + narrow + DtoH); WAV
save excluded as that's pure I/O.

| `--dit`               | 3 s clip  | 30 s clip | 120 s clip | Resident VRAM |
|-----------------------|-----------|-----------|------------|---------------|
| `sm-music` / `sm-sfx` | ~25 ms    | ~30 ms    | ~50 ms     | 8 GB          |
| `medium`              | ~45 ms    | ~75 ms    | ~150 ms    | 14 GB         |

The full-pipeline CUDA graph eliminates per-stage Python/dispatch overhead
— each replay completes in **literally identical wall-clock time** (zero
variance once the graph is built).

### Benchmark DiT step time across L values

```bash
.venv/bin/python scripts/bench_dit_profile.py \
    --engines "canonical=models/sm_90/sa3-sm-music/dit_fp16.trt" \
    --lvals 1,32,128,256,512,1024,1292,2048,4096 --warmup 3 --runs 7
```

## Flag reference

| Flag                 | Default     | Notes                                                                          |
|----------------------|-------------|--------------------------------------------------------------------------------|
| `--prompt`           | (asks)      | Text prompt; empty = unconditional                                             |
| `--negative-prompt`  | —           | CFG uncond branch; only used when `--cfg ≠ 1.0`                                |
| `--dit`              | (asks)      | `sm-music`, `sm-sfx`, or `medium`                                              |
| `--decoder`          | (asks)      | `same-s` (pairs with sm-*) or `same-l` (pairs with medium)                     |
| `--seconds`          | 30          | Output length (≈ 93 ms .. ~6.3 min)                                            |
| `--steps`            | 8           | Pingpong sampler steps; 1 = single forward, 8 = sweet spot                     |
| `--seed`             | random      | Set for reproducibility; the chosen seed is printed at the end                 |
| `--cfg`              | 1.0         | Guidance scale; 1.0 = off, >1 toward prompt, <1 toward uncond                  |
| `--apg`              | 1.0         | Adaptive Projected Guidance; only matters when `--cfg ≠ 1`                     |
| `--init-audio`       | —           | WAV (44.1 kHz, 16-bit PCM) for audio-to-audio / inpaint                        |
| `--init-noise-level` | 1.0         | σmax; 0.4–0.8 typical for variation, 1.0 = full regen                          |
| `--inpaint-range`    | —           | `START,END` seconds; regenerate that span, keep the rest                       |
| `--chunking`         | on          | SAME-L: windowed decode on the low profile (509 MB scratch). `--no-chunking` = single-shot on the wide profile, faster above L=256, ~5.4 GB more resident |
| `--limiter-ceiling`  | 0.977       | Decoder's output peak ceiling, linear (0.977 = −0.2021 dBFS). Large value = bypass, reproducing the old hard clip |
| `--dec-precision`    | canonical   | `canonical` (fp16), `fp8`, `fp8_fast` (same-s), or `legacy` (pre-limiter engines, for reproducing old renders) |
| `--quiet`            | off         | Suppress per-stage prints + NVML probes — saves ~4 ms                          |
| `--pinned-copy`      | on          | Pinned host buffer + non_blocking DtoH for Stage 5                             |
| `--free-models`      | off         | Free TRT engine memory after each stage's last use                             |
| `--out`              | out.wav     | Relative → `output/<file>`; absolute → as-is. 16-bit PCM stereo @ 44.1 kHz     |

## Files

```
optimized/tensorRT/
├── sa3                          ← shell wrapper (use this)
├── install.sh                   ← uv bootstrap + arch-aware engine download
├── bootstrap.sh                 ← curl|bash entry (installs git + clones + runs install.sh)
├── README.md
├── requirements.txt
├── output/                      ← default landing zone for generated WAVs
├── scripts/
│   ├── sa3_trt.py               ← entry point — full-pipeline CUDA graph
│   ├── sa3_trt_core.py          ← TRTRunner / DiTRunner + helpers, eager fallback sampler
│   ├── runtime.py               ← tokenizer + dist-shift loaders
│   ├── tokenizer.json           ← bundled T5Gemma tokenizer (34 MB, arch-agnostic)
│   ├── diff_attn_nocast_plugin.py
│   ├── triton_swa_v2.py         ← SAME-L SWA plugin kernel
│   └── bench_dit_profile.py     ← DiT-only timing across L values
├── build/
│   ├── README.md                ← how to build for a new GPU arch
│   ├── build.py                 ← interactive menu (default entry)
│   ├── build_from_onnx.py       ← one target → ONNX → TRT engine
│   └── build_dit_profile.py     ← DiT with custom (min, opt, max) profile shapes
└── models/                      ← .trt engines (auto-downloaded per arch; ~8 GB)
    └── sm_<cc>/                 ← arch dir matches `nvidia-smi --query-gpu=compute_cap`
        ├── t5gemma/t5gemma_fp16.trt
        ├── sa3-sm-music/dit_fp16.trt
        ├── sa3-sm-sfx/dit_fp16.trt
        ├── sa3-m/dit_fp16.trt      ← + dit_fp8.trt / dit_fp32.trt if selected
        ├── same-s/{enc,dec}_dynamic_bf16.trt
        └── same-l/{enc_fp16_chunkable,dec_fp16_chunkable_limiter}.trt
            (+ the fp8 pair, and the pre-limiter *_dynamic_triton_swa.trt as `legacy`)
```

DiT engines support a dynamic L range of **1 → 4096** at `opt=1292`
(~2 min, the most common output length). T5 hidden + mask + seconds_total
+ local_add_cond are all baked into the DiT engine, so a single TRT
invocation per sampling step handles everything.

## Notes on the design

- **Full-pipeline CUDA graph**: T5 encode + DiT 8-step loop (pingpong
  denoise/renoise math included) + decoder + int32→int16 narrow + DtoH
  copy to pinned host RAM are all captured into ONE `g.replay()`. End of
  replay, you have int16 PCM in pinned host memory — `wave.open` can
  consume it directly.
- **Per-arch engines**: TRT bakes SASS for the build arch into the engine.
  We publish prebuilt sm_90 engines on HF; install.sh queries for
  matching archs and falls back to compile-from-ONNX for everything else.
- **STRONGLY_TYPED T5Gemma**: built with an FP16-mixed graph (FP32
  attention island around softmax) — fixes a BF16 numerical bug where one
  specific cross-attention output token collapsed in magnitude.
- **PCM-baked SAME-S decoder**: the int16 narrow + transpose are folded
  into the decoder engine itself; saves ~3 ms of post-decode CPU work.
- **Mixed precision**: DiT runs FP16-mixed (FP16 trunk + FP32 RMSNorm/RoPE
  islands + FMHA-fused FP16 attention core), decoder int32→int16, T5Gemma
  FP16-mixed. `--quiet` skips per-stage NVML probes for an extra ~4 ms.
- **Auto-download**: missing engines are pulled from
  `stabilityai/stable-audio-3-optimized/tensorRT/sm_<cc>/` on first use.

## License & attribution

Model weights derived from Stability AI's Stable Audio 3 checkpoints.
T5Gemma text encoder from Google.

Use of the Stable Audio 3 weights is governed by the **Stability AI
Community License**. Please refer to the full terms at
<https://stability.ai/license>.
