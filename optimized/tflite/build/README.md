# Building the SA3 tflite models from scratch

Rebuilds every shipped tflite — the **SAME-AE** rungs (`same-{l,s}/{enc,dec}_{fp32,w8a8}.tflite`) and the
**medium DiT** rungs (`sa3-m/dit_{fp32,w8a8}.tflite`) — from the repo's OWN PyTorch model code plus the
original checkpoints. Nothing here depends on a pre-built artifact or a machine-specific path; you provide
the checkpoints and a work dir via environment variables.

## 1. Environments (two — they pin different `ai_edge_litert` versions)
- **export env** (extract / export / quantize / merge): `torch`, `ai_edge_torch`, `ai_edge_quantizer`,
  `ai_edge_litert==1.2.0`, `safetensors`, `numpy`, `flatbuffers`.
- **runtime env** (verify + inference): `ai_edge_litert>=2.2.0` (provides `CompiledModel` + the XNNPACK
  weight cache the rungs need). This is just `optimized/tflite/requirements.txt`.

These are kept separate on purpose: the runtime env stays minimal (fast path to inference); the heavy
export toolchain (torch + ai_edge_torch) lives only in the build env, used for the rare act of building.

## 2. Checkpoints (download from HuggingFace, then point env vars at them)
```
SA3_CKPT_MEDIUM   = sa3-medium checkpoint   (stabilityai/stable-audio-3-medium — the ARC .safetensors;
                    has model.model.* = the DiT, and pretransform.model.{encoder,decoder} = SAME-L)
SA3_CKPT_SMMUSIC  = sa3-sm-music checkpoint (stabilityai/stable-audio-3-sm-music — the SAME-S autoencoder)
SA3_BUILD_WORK    = output/scratch dir for all intermediates + the final rungs   (default: build/_work)
```
The medium DiT and SAME-L both come from the single `stable-audio-3-medium` ARC checkpoint. If
`SA3_CKPT_MEDIUM` is unset, the DiT export resolves it from the HuggingFace cache automatically.

## 3. Build
```bash
export SA3_CKPT_MEDIUM=/path/to/stable-audio-3-medium-ARC.safetensors
export SA3_CKPT_SMMUSIC=/path/to/sa3-sm-music/ckpt
export SA3_BUILD_WORK=/path/to/a/fresh/workdir
PY_EXPORT=/path/to/export-env/bin/python \
PY_RUNTIME=/path/to/runtime-env/bin/python \
  bash build_all.sh                 # SAME-AE + DiT; set BUILD_DIT=0 to skip the DiT
```
`build_all.sh` runs: **extract** (4 weight extractors, ckpt→npz) → **export** SAME rungs (torch→tflite) →
**quantize** (each → w8a8) → **merge** (weight-dedup into the 8 SAME files) → **build the DiT**
(`build_dit.sh`) → **verify** (structural: ladders + dispatch + shapes). Outputs land in
`$SA3_BUILD_WORK/same-{l,s}/` and `$SA3_BUILD_WORK/sa3-m/`.

The DiT stage is also runnable on its own (it is the slow part — each rung is a full ai_edge_torch convert):
```bash
PY_EXPORT=/path/to/export-env/bin/python DIT_JOBS=6 bash build_dit.sh
```
`DIT_JOBS` exports that many rungs in parallel (each convert peaks tens of GB RAM; default 1 = sequential).

## 4. Ladders / design
- **SAME-L** rungs `{1,2,4,8,12,16,32,64,128,256}`, **SAME-S** `{2,4,8,12,16,32,64,128,256}` (SAME-S is
  even-only: its attention tiles in 34-token = 2-latent chunks). Small rungs make tiny-L exact + fastest;
  `{16..256}` tiles longer L (overlap = the SWA receptive field: SAME-L 12, SAME-S 16 latents → bit-exact).
- **DiT** rungs `{8,48,96,192,416,704,1056,1496,2040,2824,3536,4096}` (latents; 48 & 96 ≈ 4.5 s / 8.9 s fill
  the 0.7→17.8 s gap — audio ≈ L/10.78 s). A render of real length L runs on
  the SMALLEST rung R ≥ L in ONE forward, with the (R−L) pad KEYS masked out of self-attention — so the
  result is EXACT, not tiled. Merged weight-shared, peak RAM is the high-water mark of the largest rung
  used, flat across length (the old varlen graph materialized `[heads,S,S]` and blew up at long L —
  ~40 GB @ L=4096 vs the rung's ~3.8 GB, a 10× RAM cut).
- **DiT global-cond preamble (the `gcond` signature).** The per-step adaLN global vector `gc[1,9216]` =
  `global_cond_embedder(to_global_embed(seconds) + timestep(t))` is computed OUTSIDE the rung graph and fed
  into the rung as an input, instead of living inside it. This is REQUIRED for the int8 rung: ai_edge_torch
  inlines that `global_cond_embedder` FC across all 24 blocks, and inside the merged int8 ladder under the
  XNNPACK weight cache the shared packed FC is mis-reused across rung subgraphs → corrupt output (~6 dB).
  Pulling it out is numerically bit-exact and keeps the rung cache-safe (999 dB cache vs no-cache). It is
  built as a tiny fp32 `gcond.tflite` and then FOLDED INTO each ladder as a separate `gcond` signature (via
  `SA3_MERGE_EXTRA`), so only `dit_{fp32,w8a8}.tflite` ship — one self-contained file each. It is a single
  non-shared subgraph, so it is cache-safe. The runtime (`rung_dit.py`) runs the `gcond` signature once per
  diffusion step (~0.5 ms, cached by t; shared across the CFG cond/uncond passes) and feeds `gc`.
- **Precision tiers.** SAME codec: `w8a8` (default, quality-free on the round-trip) + `fp32`. DiT:
  `fp32` (bit-exact reference, the current default) + `w8a8` (the speed tier that supersedes the old
  published `w8a8-dyn`; int8 on the DiT is NOT bit-identical — the distilled few-step sampler is chaotic —
  so it is a different, not necessarily worse, sample; ear-gate before making it the default).
- Full SAME rationale + CPU-support notes: `../docs/RUNGS.md`.

## 5. Files
```
build_paths.py         resolves $SA3_BUILD_WORK + checkpoints; makes torch_defs importable
extract/               4 SAME weight extractors (ckpt -> npz)
torch_defs/            checkpoint-faithful SAME torch defs + windowed_decoder (O(S) attention) + limiter
export/                torch -> fixed-size tflite rung:
                         export_windowed_param.py / export_enc_windowed.py / export_same_s_fixed.py  (SAME)
                         export_dit.py            (DiT — loads stable_audio_3.models.dit + the ARC ckpt;
                                                   `--gcond` exports the global-cond preamble gcond.tflite)
quant_merge/           quant_one (fp32->w8a8) + merge_rungs_generic (fixed rungs -> one weight-shared file)
tfl_surgery.py         flatbuffer helper used by the merge
verify_final.py        structural check of the built files (runtime env) — SAME enc/dec + DiT rungs
build_dit.sh           DiT-only orchestrator (export rungs + gcond -> quant -> merge, gcond folded in
                       as the 'gcond' signature via SA3_MERGE_EXTRA so each ladder is one file)
build_all.sh           top-level orchestrator (SAME-AE + DiT)
```
Quality (DiT renders / AE round-trip vs ground-truth audio) is a separate check — the ear is the gate.
