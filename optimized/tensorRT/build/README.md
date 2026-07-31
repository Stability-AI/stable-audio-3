# Building TRT engines for a new GPU architecture

The TRT engines published at `huggingface.co/stabilityai/stable-audio-3-optimized/tree/main/tensorRT/sm_90/` were built on Hopper (H100/H200, compute capability 9.0 → `sm_90`). TRT engines are not portable across GPU architectures — to run on `sm_100` (Blackwell) or `sm_120` (RTX 50xx) you compile fresh engines from the canonical ONNX hosted on HuggingFace.

Run the build on the target GPU; TensorRT bakes the arch into the engine, so the arch you build on _is_ the arch the engine runs on.

## Two flows

**Consumer (what most people want):** download ONNX from HF, compile to TRT for the local GPU. **Lightweight deps** — no model checkpoints, no `stable-audio-tools`, just `tensorrt` + `torch` + `huggingface-hub`.

**Producer (Stability AI / model maintainers):** trace the PyTorch source → ONNX → TRT. Refreshes the canonical ONNX after a model retrain. Heavy deps (`stable-audio-tools`, model checkpoints, etc.).

```
                              consumer flow                producer flow
                              ─────────────                ─────────────
HuggingFace                    onnx/<engine>/  ←─────── publish (incl. dit_fp16mixed.onnx)
   tensorRT/<arch>/   ←─── compile + commit              source ckpts
                              │                              │
                              ↓                              ↓
                         build.py                      build_*.py
                         build_from_onnx.py            (build_t5gemma.py,
                            (just compile,              build_dit.py,
                             STRONGLY_TYPED;            build_dit_fp16mixed.py,
                             no graphsurgeon)            build_same_*.py)
```

The SA3 DiT ships both an FP32 canonical `dit.onnx` (regenerable from PyTorch source) and a pre-processed `dit_fp16mixed.onnx` (canonical + FP32 islands around RMSNorm and RoPE *generation*, FP16 attention core, rest converted to FP16). Consumers use the pre-processed one; producers refresh both when the model retrains.

## Consumer flow (default)

```bash
export CUDA_VISIBLE_DEVICES=0     # pick a free GPU
python build.py                   # interactive menu
```

`build.py` detects your GPU arch, shows which engines exist under `../models/<arch>/` (✓) and which are missing (✗), and dispatches each build through `build_from_onnx.py <name>` which:

1. `huggingface_hub.hf_hub_download` pulls the ONNX (and `.data` sidecar for sa3-m) from `stabilityai/stable-audio-3-optimized/onnx/`.
2. TRT compiles it with arch-appropriate kernels.
3. The `.trt` lands at `../models/<arch>/<engine>/<file>.trt` — same path `sa3_trt.py` reads from.

```
━━━ SA3 TRT engine build menu ━━━

  GPU arch:   sm_100
  Output dir: models/sm_100/

  [1] ✓  t5gemma  (text encoder + tokenizer)
        ✓  t5gemma/t5gemma_fp16mixed.trt  538.1 MB
        ✓  t5gemma/tokenizer.json     32.8 MB
  [2] ✗  same-s encoder
        ✗  same-s/enc_dynamic_bf16.trt  (missing)
  ...
  [A] Build all missing  (7 target(s))
  [Q] Quit
```

Direct, non-interactive:
```bash
python build_from_onnx.py t5gemma
python build_from_onnx.py same-l-decoder
python build_from_onnx.py sa3-sm-music
python build_from_onnx.py all          # every canonical (FP16-mixed) engine

# FP32 variants — opt-in. ~2× engine size, ~2× slower, bit-equivalent to PT eager.
# Pair with `sa3_trt --precision fp32` at inference.
python build_from_onnx.py same-l-decoder-fp32   # upcasts ONNX FP16→FP32 in-process
python build_from_onnx.py same-s-decoder-fp32   # canonical ONNX is already FP32
python build_from_onnx.py sa3-m-fp32            # reads HF dit.onnx (already FP32)
python build_from_onnx.py all-fp32              # every FP32 target
python build_from_onnx.py all-both              # canonical + FP32
```

### Consumer deps

- `tensorrt==10.15.1.29` — pinned (TRT 10.x engines aren't cross-minor-compatible)
- `torch` (TRT plugins use torch tensors; needed for SAME-L plugin verification)
- `triton` — for the SAME-L SWA plugin kernel (typically bundled with PyTorch on Linux)
- `huggingface-hub`
- `numpy`

That's it — no `stable-audio-tools`, no `transformers`, no model checkpoints.

## Choosing the SAME-L attention kernel

The SAME-L encoder/decoder use a custom sliding-window-attention plugin
(`samel::diff_attn_swa`). It ships **two implementations**, and which one the engine uses
is decided at build time and baked into the `.trt`. If you are bringing up an
architecture we have not published engines for, this is the one knob you may have to
touch.

```bash
SA3_SWA_PLUGIN=aot   # default. PTX kernel compiled into the engine.
SA3_SWA_PLUGIN=jit   # Triton dispatched through a Python callback per enqueue.
SA3_SWA_AOT=mma      # default AOT kernel: block-tiled, tensor cores.
SA3_SWA_AOT=ptx      # fallback AOT kernel: scalar FP32, one warp per query.
```

| | AOT (`mma`) | JIT |
|---|---|---|
| CUDA-graph capturable | **yes** | yes on sm_90, **no on sm_120** |
| Python/Triton callbacks per decode | **0** | 12 (one per attention layer) |
| Needs torch + Triton at inference | no | **yes** |
| Engine size | +0.1% | baseline |
| Build-time deps | Triton | Triton |

### What the published engines use

| published engines | kernel |
|---|---|
| `sm_120/same-l/` | AOT (block-tiled MMA) — required, see below |
| `sm_90/same-l/` | JIT — predates the AOT kernel, captures fine on Hopper, and AOT measured no faster there, so they were left alone |

Building `same-l` for sm_90 today gives you AOT rather than the published JIT engine. That
is fine — the two are at parity in speed and AOT is the more robust of the pair — but it
does mean your local engine will not be byte-identical to the downloaded one.

The runtime registers both implementations, so a JIT engine and an AOT engine both load
and run whatever GPU you are on. Verified: the published sm_90 JIT engine produces
byte-identical output before and after the AOT implementation was added.

### Why AOT is the default

**Correctness on sm_120.** A JIT engine is not stream-capturable there: `enqueueV3`
returns `False` under capture, TRT reports *"this TRT engine is not stream capturable"*,
and the stage is **silently omitted** from the graph. The decoder then returns the
pre-capture warmup decode of zero latents — a constant wash of noise, byte-identical
across seeds and prompts, with **exit code 0**. It captures fine on sm_90, but
re-entering Python inside a captured region is fragile rather than supported, so do not
assume a new architecture will tolerate it.

**Robustness.** AOT removes 12 GIL acquisitions per decode, which matters for
concurrent serving, and removes the runtime Triton dependency entirely.

### Measured (SAME-L decode, ms, median of 7 on an idle GPU)

| sm_120 · RTX PRO 4500 | L=256 | L=1024 | L=1292 | L=2048 | L=4096 |
|---|---|---|---|---|---|
| JIT | 55.1 | 227.2 | 289.1 | 457.7 | 920.2 |
| AOT `ptx` | 62.5 | 251.2 | 318.7 | 502.8 | 1008.8 |
| **AOT `mma`** | 55.4 | **223.6** | **283.7** | **448.4** | **899.3** |

| sm_90 · H200 | L=256 | L=1024 | L=1292 | L=2048 | L=4096 |
|---|---|---|---|---|---|
| JIT | 12.4 | 47.0 | 61.1 | 98.5 | 194.4 |
| AOT `ptx` | 12.7 | 47.3 | 60.7 | 96.6 | 195.5 |
| **AOT `mma`** | 12.4 | 48.4 | 62.6 | 100.0 | 196.8 |

Accuracy against the FP32 decoder at L=1292 on sm_120: AOT `mma` **51.45 dB**, JIT
51.43, AOT `ptx` 51.11 — the MMA kernel is the most accurate of the three.

So `mma` is at parity with JIT on both architectures (a few percent either way, faster
at long sequence on sm_120) while being capturable. `ptx` costs ~10% on sm_120 and is
kept only as a fallback if the MMA kernel will not build on some future target.

### If you are bringing up a new architecture

1. Build with the defaults and **check the decoder output is not constant** — render two
   different seeds and compare. Identical output is the non-capturable-engine signature
   described above, not a sampling quirk.
2. If the AOT build fails, try `SA3_SWA_AOT=ptx` (simpler kernel, no shared-memory or
   tensor-core requirements), then `SA3_SWA_PLUGIN=jit` as a last resort — and if you
   land on `jit`, verify capture explicitly rather than trusting it.
3. Two AOT constraints that bite on new targets:
   - The MMA kernel needs **40 KB of shared memory**. `BLOCK_N=64` would need 64 KB,
     over the 48 KB static limit, and fails at enqueue with *"Failed to enqueue status
     -1"*. If a target has less shared memory, lower `BLOCK_N` in
     `scripts/triton_diff_swa_mma.py` — but keep `BLOCK_KV >= BLOCK_N + 2*WINDOW`, or the
     K/V tile stops covering the window and attention contributions are silently dropped.
   - The PTX `.target` is rewritten to the plain `sm_XX` (Triton emits `sm_XXa`, which
     TRT's loader rejects). The `m16n8k8` TF32 `mma` used here needs sm_80+.

## Publishing TRT engines to HuggingFace

After building all 8 engines for a new `<arch>`, push them to HF so others on the same GPU don't need to rebuild:

```bash
HF=/path/to/stable-audio-3-optimized
mkdir -p $HF/tensorRT/<arch>
cp -r ../models/<arch>/* $HF/tensorRT/<arch>/
cd $HF
git lfs track "*.trt"  # already in .gitattributes
git add tensorRT/<arch>
git commit -m "Add <arch> TRT engines"
git push
```

Once pushed, `install.sh` on any matching machine auto-detects the new arch from the HF API and downloads — no script changes needed.

## Producer flow (refresh the canonical ONNX)

Only needed when the underlying SA3 model weights change. Re-exports ONNX from the PyTorch source, then publishes to HF.

### Required source checkpoints

| Engine | Source ckpt |
|---|---|
| `sa3-{m,sm-music,sm-sfx}/dit.onnx` | `<MODELS_ROOT>/SA3-{M-hf,sm-music,sm-sfx}/{model_config.json,model.safetensors}` |
| `same-s/{enc,dec}_dynamic_bf16.onnx` | `<MODELS_ROOT>/SAME-S/{SAME-S.ckpt,SAME-S.json}` |
| `same-l/{enc,dec}_dynamic_triton_swa.onnx` | `<MODELS_ROOT>/SAME-L/{SAME-L.ckpt,SAME-L.json}` |
| `t5gemma/encoder.onnx` | `google/t5gemma-b-b-ul2` (auto-downloaded via `transformers`) |

Default `MODELS_ROOT` is hard-coded in each `build_*.py`; edit the constants at top if yours differ.

### Producer deps (on top of the consumer set)

- `stable-audio-tools` (install via `pip install git+https://github.com/Stability-AI/stable-audio-tools` — heavy, ~1 GB of audio deps)
- `transformers` (for T5Gemma load)
- `onnx`, `safetensors`

### Producer build order

T5Gemma and SAME-S are independent. SAME-L encoder imports the decoder builder (shared `patched_diff_attention_forward`), so build the decoder first.

```bash
python build_t5gemma.py
python build_same_s_decoder.py
python build_same_s_encoder.py
python build_same_l_decoder.py
python build_same_l_encoder.py
python build_dit.py sa3-sm-music
python build_dit.py sa3-sm-sfx
python build_dit.py sa3-m
```

After the DiT ONNXes are exported, run the FP16-mixed precision-island surgery on each one (see `build_dit_fp16mixed.py`):

```bash
python build_dit_fp16mixed.py \
    --input  <HF_REPO>/onnx/sa3-sm-music/dit.onnx \
    --onnx   <HF_REPO>/onnx/sa3-sm-music/dit_fp16mixed.onnx \
    --engine ../models/<arch>/sa3-sm-music/dit_fp16mixed.trt
# repeat for sa3-sm-sfx and sa3-m
```

This wraps every RMSNorm chain, attention `Softmax`, and the RoPE region in `Cast(FP32) → op → Cast(FP16)` islands, converts the rest of the weights to FP16, then **bounds the RoPE island before QK^T** (`bound_attention_core()`) so the attention core — QK^T → Softmax → P·V — runs FP16, and finally compiles a `STRONGLY_TYPED` TRT engine. It writes BOTH the modified `dit_fp16mixed.onnx` (~half the size of the original) AND the TRT engine. Publishing the modified ONNX is what lets consumers compile their own engines with plain `build_from_onnx.py` (no `onnx-graphsurgeon` dependency on the consumer side).

The attention-core step is not cosmetic. Without it the RoPE island emits q/k in FP32 and nothing casts them back, so the whole O(L²) core stays FP32 and TRT's FMHA fuser cannot fire: **0 fused MHA nodes and 4.3× slower at L=4096** on the medium DiT, for no accuracy gain (teacher-forced velocity cos vs the FP32 engine is 1.0000 with the step, 0.9998 without). PyTorch does the same thing the step does — `apply_rotary_pos_emb` ends with `t.to(out_dtype)` and attention is then a fused `scaled_dot_product_attention` over FP16 inputs — and TRT's FMHA kernels accumulate softmax in FP32 internally anyway. `--no-bound-attn` skips it, only useful for reproducing the pre-2026-07 engines.

`STRONGLY_TYPED` is **mandatory**, not stylistic. Building the same graph weakly-typed with `BuilderFlag.FP16` fuses just as well and runs just as fast, but the FP16 flag also lets TRT re-cast the FP32 RMSNorm islands — i.e. it silently degrades to naive FP16 (measured: teacher-forced cos 0.88 @L=256, 0.77 @L=4092). Note this is the opposite of the fp8-GEMM recipe, where weakly-typed is required; the rule is per-pattern.

Naive `BuilderFlag.FP16` (without the surgery) catastrophically overflows in RMSNorm variance — the RMSNorm islands are mandatory. BF16 was tried earlier and is audibly degraded on long renders, but the mechanism is narrower than the "compounds quantisation error over 8 sampling steps" story recorded here previously: a per-op ablation showed bf16 arithmetic through the rest of the DiT is clean, and the single culprit is the **RoPE rotation angle**. `t · inv_freq` reaches ~4155 rad at L=4092, where bf16's spacing is 32 rad — larger than a full 2π rotation — so `cos`/`sin` of it carry no position information for the fast-rotating dims (9 of 16 frequency pairs destroyed). The latent then inflates ~2.5× over the 8 steps and the decoder clips 2–3% of samples. It is clean at short lengths. The general rule: **any Fourier or rotary feature whose angle can exceed ~2048 rad is unsafe in bf16 by construction** — that is where bf16's ULP first exceeds 2π.

Each script also writes the ONNX to `<HF_REPO>/onnx/<engine>/<file>.onnx`. After all 8 are done:

```bash
HF=/path/to/stable-audio-3-optimized
cd $HF
git add onnx/
git commit -m "Refresh canonical ONNX"
git push
```

## Medium `fp8` — the max-speed RoPE-baked engine, calibrated

On top of the `fp16mixed` default and the selectable `bf16`, the medium DiT ships an **`fp8`**
engine (`dit_fp8.trt`, `--precision fp8`): fp8 E4M3 on the 176 linear GEMMs + 96 bf16 fused
FMHA + the **same baked fp32 RoPE constant table** as bf16. Measured on H200 it is ~1.3×
faster than `fp16mixed` at every length (1.40× / 1.32× / 1.34× @L129/1292/4092, also ahead of
bf16) and clean at long sequence (latent std 0.86, 0.000% clip at 6-min), so it stays clean
exactly where bf16 clips.

The fp8 scales are **calibrated on real conditioning** (updated 2026-07-31). The scale VALUES —
per-tensor activation amax + per-channel weight scales — come from @ryanontheinside's fp8 work
([#47](https://github.com/Stability-AI/stable-audio-3/pull/47)): captured with his `make_calib.py`
and grafted onto this baked / bf16-fused structure by `transplant_scales.py`. Calibration is
**speed-free** (31.2 vs the earlier uncalibrated 30.8 ms/fwd, inside run noise) and lifts
worst-case per-step fidelity — worst-step velocity-cos vs fp32 on adversarial seeds goes
**0.52/0.57/0.64 → 0.92/0.94/0.92**, with sampling steps 1–7 tracking the fully-calibrated #47
reference within ~0.001. It is still a **speed tier over the `fp16mixed` default, not a fidelity
upgrade over it** (single-step velocity cos ~0.92–0.94 < fp16mixed's ~1.0; the 8-step render
stays coherent), but it no longer collapses at the highest-noise first step the way the
uncalibrated engine did. See the runtime README's precision section for positioning.

**Consumer (per-arch rebuild).** The `.trt` is `sm_90`-specific; `sm_89` / `sm_120` / `sm_100`
are a rebuild away (run on the target GPU):

```bash
python build_from_onnx.py sa3-m-fp8     # pulls the calibrated dit_fp8.onnx (+ dit_fp8lin.onnx.data) from HF
```

`sa3-m-fp8` is the `sa3-m-bf16` recipe **plus `BuilderFlag.FP8`**: weakly-typed
`EXPLICIT_BATCH` + `BF16` + `FP8` + `OBEY_PRECISION_CONSTRAINTS`, reusing the same
`_pin_fourier_fp32` island (on the RoPE-baked ONNX the only `Cos`/`Sin` left are the two
runtime Fourier chains). The fp8 E4M3 Quantize/Dequantize pairs — carrying the calibrated
scales — ride in the ONNX, so TRT fires fp8 tensor-core GEMMs on the Linears while attention
stays bf16 fused-MHA. The build recipe is **unchanged by calibration**: calibration lives in the
ONNX scale values, not the builder flags. Identity of the shipped engine — **176 fp8 GEMMs +
96 bf16 fused MHA + baked fp32 RoPE constant** — verify with
`python ../scripts/verify_fp8_rope.py <engine>.trt` (needs a DETAILED-verbosity build, as the
shipped engine is; a plain consumer rebuild renders identically but is not introspectable).

**Producer (refresh the ONNX).** Two independent pieces: RoPE-baking (structure) and
calibration (scale values).

RoPE-baking is the SAME step as bf16 — `build_dit_bf16.py` is the shared baker (it handles both
an external `inv_freq`, as in the fp32 `dit.onnx`, and an inline one, as in the fp8-linear ONNX):

```bash
python build_dit_bf16.py --input dit_fp8lin.onnx --output dit_fp8lin_ropebaked.onnx --max-t 4160
```

`--max-t` (4160 = profile max 4096 + 64 global tokens) sizes the baked table; rendering past
L=4096 would need a larger `--max-t` **and** a matching TRT profile — the runtime rejects
L>4096 up front, coinciding with the SAME-L decoder's own cap, so it is not a new end-to-end
limit.

Then graft the calibrated fp8 scales onto the baked graph with `transplant_scales.py`. It matches
every quantized Linear by weight-initializer name and swaps only the scale VALUES (per-tensor
activation amax + per-channel weight scales); the 5.8 GB fp32 weights are untouched and TRT
re-quantizes them at build:

```bash
python transplant_scales.py \
    --ours-onnx  dit_fp8lin_ropebaked.onnx \
    --calib-onnx <build_dit_fp8.py output>/dit_fp8_calib.onnx \
    --out        dit_fp8.onnx           # publish this as onnx/sa3-m/dit_fp8.onnx
```

The calibrated `--calib-onnx` comes from @ryanontheinside's fp8 pipeline
([#47](https://github.com/Stability-AI/stable-audio-3/pull/47)). A **from-scratch recalibration**
(only needed on a model retrain) reruns `make_calib.py` (real-conditioning capture, in this
repo) → `build_dit_fp8.py` (max-PTQ + per-channel weight scales; that builder lives in #47, not
merged here). Everyday consumers never recalibrate — they pull the published calibrated
`dit_fp8.onnx`.

## Small-DiT `fp8` — sm-music / sm-sfx (a different, simpler recipe)

`sm-music` and `sm-sfx` also ship a selectable **`fp8`** engine (`--precision fp8`), but it is
**not** the medium's baked-RoPE recipe — those DiTs use standard (non-differential) attention and
never had the bf16 long-angle RoPE problem, so there is nothing to bake. Their fp8 is a straight
**graft of fp8 E4M3 Q/DQ onto the linear GEMMs of the known-good `dit_fp16mixed.onnx`** — attention
stays fp16-fused and the fp32 RMSNorm/RoPE islands are left exactly as the fp16mixed producer made
them. Built `STRONGLY_TYPED` (`build_from_onnx.py sa3-sm-music-fp8` / `sa3-sm-sfx-fp8`); the QDQ
carry the precision. Identity: 186 fp8 GEMMs + fp16 fused attention + the fp16mixed fp32 islands.

Positioning is honest: this is a **clean weight-halving tier** (engine 479 vs 936 MB, velocity-cos
~0.99 vs eager, clip% at or below fp16mixed), only **marginally faster** (~1.10–1.17×) — a small
DiT's ~5 ms forward at batch 1 is overhead-bound, so fp8's GEMM-math savings barely show. Default
stays `fp16mixed`; fp8 is for when the smaller engine / weight footprint helps. Not seed-reproducible
vs fp16mixed.

> ⚠ Do **not** produce these with `build_dit_fp8.py` (#47's ModelOpt path): on the small graphs its
> island-flatten + reapply does not restore the fp32 islands correctly and the engine collapses to
> velocity-cos ~0.69 with clipping (the GEMMs are fine — it's the islands). Grafting onto the
> fp16mixed ONNX keeps the islands correct by construction.

**Producer (refresh the ONNX).** `make_dit_fp8_smalldit.py` calibrates per-linear activation scales
from the eager model (own-domain few-shot prompts + one full render) and grafts the fp8 Q/DQ. Two
fp16-trunk specifics vs the medium inserter: Q/DQ scales are **FLOAT16** (fp16 trunk → DQ must output
fp16) and floored at 1e-4 (fp16 underflows tiny scales to 0, which TRT rejects):

```bash
python make_dit_fp8_smalldit.py \
    --model-config <ckpt>/model_config.json --checkpoint <ckpt>/model.safetensors \
    --fp16mixed-onnx onnx/sa3-sm-music/dit_fp16mixed.onnx \
    --domain Music --out onnx/sa3-sm-music/dit_fp8.onnx      # --domain SFX for sm-sfx
```

## File map

| File | Role | Flow |
|---|---|---|
| `build.py` | Interactive menu (default entry point) | consumer |
| `build_from_onnx.py` | One target → download ONNX from HF + compile to TRT. Decoder builds remove the baked output Clip first. **For the SA3 DiTs, pulls `dit_fp16mixed.onnx` (the pre-processed island-wrapped graph)** so the consumer just needs to invoke `STRONGLY_TYPED` compilation — no `onnx-graphsurgeon` required | consumer |
| `decoder_output.py` | Rewrites decoder ONNX outputs to remove the baked `[-1, 1]` Clip and expose `pcm_unbounded`; runtime applies no-boost attenuation before INT16 narrowing | consumer + producer |
| `build_dit_profile.py` | Build a DiT with custom `(min, opt, max)` profile shapes (experimental — short-form / fixed-shape variants). Operates on either ONNX flavor. | consumer |
| `build_dit_fp16mixed.py` | **Producer-side** ONNX surgery: takes the canonical FP32 `dit.onnx`, finds RMSNorm chains + attention `Softmax` + RoPE region, wraps each in `Cast(FP32) ↔ Cast(FP16)` islands, converts non-island weights to FP16, then bounds the RoPE island before QK^T (`bound_attention_core()`, `--no-bound-attn` to skip) so the attention core runs FP16 and TRT's FMHA fuser fires — 96/96 attentions on the medium DiT, 4.3× at L=4096. Writes both the modified `dit_fp16mixed.onnx` AND the TRT engine, which **must** be `STRONGLY_TYPED` (weakly-typed + `BuilderFlag.FP16` re-casts the FP32 islands and silently degrades to naive FP16). Only re-run when the model retrains or the island recipe changes. Requires `onnx` + `onnx-graphsurgeon`. | producer |
| `build_dit_bf16.py` | **Producer-side** shared RoPE-baker for the medium `bf16` AND `fp8` engines: precomputes RoPE's cos/sin in fp64 on the host, freezes them as fp32 constant tables (`--max-t`), rewires the 96 trig sites and lets DCE delete the runtime angle chain — so the trunk runs bf16/fp8 without the long-angle drift. Weights are never loaded (keeps the input's `.data` sidecar). Handles both external `inv_freq` (fp32 `dit.onnx`) and inline (fp8-linear ONNX). Consumer compile: `build_from_onnx.py sa3-m-bf16` / `sa3-m-fp8`. Requires `onnx`. | producer |
| `make_calib.py` | **Producer-side** FP8 calibration capture: drives the model's own pingpong `generate()` and records the six DiT engine inputs at every sampling step into a `.npz` (real-conditioning, deployment-matched prompts). Feeds `build_dit_fp8.py` (#47). Only re-run for a from-scratch recalibration on a model retrain. Calibration tooling by @ryanontheinside (#47). Requires `torch` + `stable_audio_3`. | producer |
| `transplant_scales.py` | **Producer-side** fp8 scale transplant: grafts #47's calibrated activation + per-channel weight scales onto the RoPE-baked bakedmin ONNX (matches Linears by weight-initializer name, swaps only the scale VALUES; weights untouched). This is what makes the shipped `dit_fp8.onnx` calibrated while keeping bakedmin's bf16 fused-MHA speed. Scale values / calibration by @ryanontheinside (#47). Requires `onnx`. | producer |
| `../scripts/verify_fp8_rope.py` | EngineInspector identity check for the fp8 engine (176 fp8 GEMMs + 96 bf16 fused MHA + baked fp32 RoPE constant). Needs a DETAILED-verbosity build. | consumer |
| `build_t5gemma.py` | Trace + export T5Gemma encoder ONNX + build TRT | producer |
| `build_same_s_decoder.py` | Trace + export SAME-S decoder ONNX + build TRT | producer |
| `build_same_s_encoder.py` | Trace + export SAME-S encoder ONNX + build TRT | producer |
| `build_same_l_decoder.py` | Trace + export SAME-L decoder ONNX (Triton SWA) + build TRT | producer |
| `build_same_l_encoder.py` | Trace + export SAME-L encoder ONNX (Triton SWA) + build TRT | producer |
| `build_dit.py <NAME>` | Trace + export DiT FP32 ONNX (cond baked in) + build TRT BF16 engine (legacy; the BF16 output isn't suitable for inference — chain it with `build_dit_fp16mixed.py` afterwards) | producer |
| `_arch.py` | Shared: GPU arch detection + path helpers | both |
| `samel_loader.py` | Helper: load SAME-L from .ckpt | producer |
| `samel_{encoder,decoder}_onnx.py` | Helper: clean ONNX rewrites of SAME-L blocks | producer |
