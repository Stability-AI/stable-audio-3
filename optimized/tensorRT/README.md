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

Scratch, read per profile with `get_device_memory_size_for_profile_v2()` — never
`device_memory_size_v2`, which reports the max across profiles and so credits the low band with
the wide band's reservation:

| | decoder | encoder |
|---|---|---|
| SAME-L, `--chunking` (default) | **485 MB** | **496 MB** |
| SAME-L, `--no-chunking` | 7,766 MB | 7,944 MB |
| SAME-S bf16, `--chunking` | **346 MB** | **348 MB** |
| SAME-S bf16, `--no-chunking` | 5,512 MB | 5,531 MB |
| SAME-S fp8, `--chunking` | **327 MB** | **329 MB** |
| SAME-S fp8, `--no-chunking` | 5,204 MB | 5,225 MB |

These are flat in L: TensorRT sizes scratch from the profile ceiling, not from the shape you
bind, so the wide band costs the same 7.8 GB on a 3-second clip as on a six-minute one.

Chunking is a **memory** mechanism, not a speed one, and the cost is smaller than it looks in
isolation. Decode alone at L=4096 is 250.0 ms chunked against 195.6 ms single-shot (1.28×), but
the DiT dominates a render, so end-to-end it is **2–5%**: on medium fp16 + SAME-L a 380 s render
is 592 ms chunked against 565 ms single-shot, and resident VRAM is 5,756 MiB against 12,832 MiB.
Full curves in [`benchmarks/`](benchmarks/).

**A limiter.** The decoder emits already-limited PCM (5.8 ms window, sample-peak, ceiling
0.977 = −0.2021 dBFS) in place of the old hard clip. The ceiling is **baked**, so the engines keep
the `('latent', 'pcm')` signature of the pre-limiter builds and drop straight in — a baked engine
is bit-exact against a runtime-input one and about 4% faster. Rebuild with `--ceiling-input` if you
want it settable per call. ⚠ **This changes the audio.** To reproduce renders made before it landed,
the pre-limiter engines are no longer reachable from the runtime — rebuild one from the pristine
`onnx/same-l/dec_fp16.onnx` / `onnx/same-s/dec_bf16.onnx` if you need it.

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
the `bands` / `opts` columns of the `TARGETS` table in
[`build/build_autoencoders.py`](build/build_autoencoders.py) for the tuned values.

⚠ The old `same-l/enc_fp8.trt` was **removed** from the model repo. Its activation quantisers were
calibrated with a doubly-conservative percentile plus a `1e-4` floor, putting the clip points at
0.04–1.22 where the decoder's plain `amax` puts them at 22.4; that cost up to 2.16 dB of round trip
on clean acoustic material. `enc_fp8_chunkable.trt` is the same weights recalibrated
([`quantize/recalib_enc_fp8.py`](quantize/recalib_enc_fp8.py)) and measures 30.9 dB of latent SNR
where the old one measured 18.3.

## Building on hardware we don't publish for

Prebuilt engines exist only for the architectures we can build and verify on (currently
`sm_90`). On anything else — `sm_120`, `sm_89`, … — the autoencoders are a ~10 minute local
build, and the runtime asks rather than failing:

```
No prebuilt engines for sm_120 — 2 of the 4 file(s) this configuration needs are not published:
    same-l/dec_fp16_chunkable_limiter.trt
    same-l/enc_fp16_chunkable.trt

These are autoencoders — buildable here from the published ONNX in about 10 minutes.
Build them now for sm_120? [Y/n]
```

The whole file list is checked against the model repo *before* anything downloads, so you never
pay 3 GB for a DiT and T5 only to discover the decoder is absent. Answering yes runs the build;
declining, a non-tty, or an engine the AE script can't build (DiT, T5Gemma) exits with the exact
command instead.

To build ahead of time, or to rebuild after changing a profile:

```bash
python build/build_autoencoders.py                      # all eight, verified
python build/build_autoencoders.py --model same-s --kind dec --force
python build/build_autoencoders.py --list
```

It downloads the ONNX (the limiter is already grafted into the decoder graphs), compiles both
profiles, and verifies by default — decoder checks are an encode→decode round trip, so a
`--kind dec` build with no encoder present reports SKIP rather than a failure it isn't
responsible for. Engines land in `models/<arch>/` and are picked up automatically.

⚠ `build_from_onnx.py` no longer builds autoencoders itself. Its four AE targets delegate here:
it used to emit single-profile, limiter-less engines under the retired
`dec_dynamic_triton_swa.trt` / `dec_dynamic_bf16.trt` names, which the runtime registry can never
request — so `all` was quietly producing unusable files.

## Benchmarks

[`benchmarks/`](benchmarks/) has the full sweep of all eight autoencoder configurations —
`{SAME-S, SAME-L} × {16-bit, fp8} × {chunked, single-shot}` — across 32 latent lengths from L=1
to L=8192, as six charts plus the raw JSON and the scripts that produce both:

```bash
python scripts/bench_autoencoders.py --out b.json --music <dir|file|npy>
python scripts/make_ae_charts.py b.json charts.html
```

The decoder is the half most people are choosing between, so its three charts are below. The
encoder's are in [`benchmarks/`](benchmarks/) along with the interactive version, where hovering
a point gives its value.

**VRAM is flat in L** — TensorRT sizes scratch from the profile ceiling, not from the shape you
bind, so `--no-chunking` costs the full 7.8 GB on a three-second clip just as on a six-minute one.

![Decoder — VRAM scratch reserved](benchmarks/img/dec-vram.png)

**Latency** — chunked and single-shot are identical up to L=256, where the whole request is one
window; the step is where windowing begins. Single-shot ends at L=4096, the wide band's ceiling.

![Decoder — latency](benchmarks/img/dec-latency.png)

**Accuracy is flat**, and chunked lands on single-shot to three decimals at every length. Since
single-shot has no seams at all, that is the direct evidence the windowed overlap is exact. fp8
costs 0.014 dB on SAME-L and 0.052 dB on SAME-S. The fall below L≈16 is real but is not a seam
either: 0.09 s blocks encoded independently lose context at every boundary.

![Decoder — accuracy, content held fixed](benchmarks/img/dec-accuracy.png)

⚠ Two traps worth knowing before you measure these autoencoders yourself, both encoded in the
script. **Accuracy has to hold content fixed** — scoring the first L latents against the original
makes the L axis a content walk, since a longer L is a *different, longer piece of music*: at
fixed L, moving the excerpt swings SNR 9.4–15.0 dB where the whole L axis at a fixed offset moves
2.95 dB. And **real music only** — these are content-dependent (~16 dB on generated audio against
~5 dB on real masters), so the accuracy pass refuses to run without `--music` and will not fall
back to noise.

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
| `--chunking`         | on          | Windowed decode on the low profile (485 MB scratch, SAME-L). `--no-chunking` = single-shot on the wide profile: 2–5% faster end-to-end, ~7 GB more resident |
| `--limiter-ceiling`  | baked 0.977 | Only for engines built `--ceiling-input`; the shipped ones bake it |
| `--dec-precision`    | canonical   | `canonical` (fp16 for SAME-L, bf16 for SAME-S) or `fp8`. Both are chunkable and carry the limiter |
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
│   ├── sa3_gradio.py            ← web UI; gradio_ui.py is its backend-agnostic layer
│   ├── bench_dit_profile.py     ← DiT-only timing across L values
│   ├── bench_autoencoders.py    ← the eight AE configurations: VRAM, ms, accuracy
│   └── make_ae_charts.py        ← that sweep → one self-contained HTML page
├── benchmarks/                  ← the sweep: charts, raw JSON, and how to re-run it
├── build/
│   ├── README.md                ← how to build for a new GPU arch
│   ├── build.py                 ← interactive menu (default entry)
│   ├── build_autoencoders.py    ← the eight AE engines (2 profiles + baked limiter)
│   ├── build_from_onnx.py       ← one target → ONNX → TRT engine (AEs delegate above)
│   ├── build_dit_profile.py     ← DiT with custom (min, opt, max) profile shapes
│   └── limiter/                 ← limiter.onnx + the graft used to bake it in
└── models/                      ← .trt engines (auto-downloaded per arch; ~8 GB)
    └── sm_<cc>/                 ← arch dir matches `nvidia-smi --query-gpu=compute_cap`
        ├── t5gemma/t5gemma_fp16.trt
        ├── sa3-sm-music/dit_fp16.trt
        ├── sa3-sm-sfx/dit_fp16.trt
        ├── sa3-m/dit_fp16.trt      ← + dit_fp8.trt / dit_fp32.trt if selected
        ├── same-s/{enc_bf16_chunkable,dec_bf16_chunkable_limiter}.trt
        └── same-l/{enc_fp16_chunkable,dec_fp16_chunkable_limiter}.trt
            (+ the fp8 pair for each; `--dec-precision fp8`)
```

### The model repo

Engines and the ONNX they are built from live in
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized):

```
tensorRT/sm_<cc>/                 ← prebuilt engines, one dir per arch
├── t5gemma/t5gemma_fp16.trt
├── sa3-{m,sm-music,sm-sfx}/dit_{fp16,fp8,fp32}.trt
├── same-s/{enc_bf16_chunkable,dec_bf16_chunkable_limiter}.trt   (+ the fp8 pair)
├── same-l/{enc_fp16_chunkable,dec_fp16_chunkable_limiter}.trt   (+ the fp8 pair)
└── same-{s,l}/legacy/            ← superseded engines, kept but never resolved

onnx/                             ← sources; build_autoencoders.py / build_from_onnx.py read these
├── sa3-{m,sm-music,sm-sfx}/dit_{fp16,fp8}.onnx
├── t5gemma/encoder.onnx
├── same-l/{enc_fp16,dec_fp16,enc_fp8,dec_fp8}.onnx
├── same-l/{dec_fp16_limiter,dec_fp8_limiter}.onnx      ← what the decoders build from
├── same-s/{enc_bf16,dec_bf16,enc_fp8,dec_fp8}.onnx
├── same-s/{dec_bf16_limiter,dec_fp8_limiter}.onnx      ← what the decoders build from
└── same-{s,l}/legacy/            ← retired w8_bf16 and fp8_fast graphs
```

Three naming rules, once you know them the layout reads itself:

- **`_limiter` marks the grafted decoders.** `dec_fp16_limiter.onnx` is `dec_fp16.onnx` with the
  limiter spliced in — 99.8% the same graph, +66 `lim/` nodes replacing the old hard clip. The
  plain ones are kept as the pristine pre-graft sources, which is what
  [`build/limiter/graft_limiter.py`](build/limiter/graft_limiter.py) consumes if the limiter is
  ever re-grafted.
- **`chunkable` appears on engines, never on ONNX.** It is the two optimization profiles the TRT
  builder adds, so `dec_fp16_limiter.onnx` → `dec_fp16_chunkable_limiter.trt`. The encoders make
  the same point more plainly: `enc_fp16.onnx` → `enc_fp16_chunkable.trt`, identical graph, two
  profiles instead of one.
- **`legacy/` is never resolved by the runtime.** The registry only names the files above; nothing
  reads `legacy/`. It exists so an older checkout or an experiment can still find what it needs.

⚠ The SAME-L encoders build from an ONNX carrying a `samel::diff_attn_swa` plugin node. Whether
that becomes a JIT or an AOT kernel is decided at *build* time — `build_autoencoders.py` always
sets `PREFER_AOT_PYTHON_PLUGINS`, so the Triton PTX is compiled **into** the engine. A shipped
engine therefore deserializes and runs with no plugin import at all; a JIT one would not, and
could not go inside a CUDA graph.

DiT engines support a dynamic L range of **1 → 4096** at `opt=1292`
(~2 min, the most common output length), and the chunkable autoencoders match it — every one
reports a floor of 1 latent, so a 0.1 s render is legal. (Until recently it was not: the
runtime's floor check swallowed a `TypeError` and fell back to a hardcoded 32, rejecting
everything under ~3 s regardless of what the engine could do.) T5 hidden + mask + seconds_total
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
