# Building the SAME-AE rung tflites from scratch

Rebuilds all 8 canonical rung models — `same-{l,s}/{enc,dec}_{fp32,w8a8}.tflite` — from the original
checkpoints. Nothing here depends on pre-built artifacts or any machine-specific path; you provide the
checkpoints and a work dir via environment variables.

## 1. Environments (two — they pin different `ai_edge_litert` versions)
- **export env** (extract / export / quantize / merge): `torch`, `ai_edge_torch`, `ai_edge_quantizer`,
  `ai_edge_litert==1.2.0`, `safetensors`, `numpy`, `flatbuffers`.
- **runtime env** (verify + inference): `ai_edge_litert>=2.2.0` (provides `CompiledModel` + the XNNPACK
  weight cache the rungs need).

## 2. Checkpoints (download from HuggingFace, then point env vars at them)
```
SA3_CKPT_MEDIUM   = sa3-medium checkpoint   (stabilityai/stable-audio-3-medium — the ARC .safetensors;
                    has pretransform.model.{encoder,decoder} = SAME-L)
SA3_CKPT_SMMUSIC  = sa3-sm-music checkpoint (stabilityai/stable-audio-3-sm-music — the SAME-S autoencoder)
SA3_BUILD_WORK    = output/scratch dir for all intermediates + the 8 final rungs   (default: build/_work)
```

## 3. Build
```bash
export SA3_CKPT_MEDIUM=/path/to/stable-audio-3-medium-ARC.safetensors
export SA3_CKPT_SMMUSIC=/path/to/sa3-sm-music/ckpt
export SA3_BUILD_WORK=/path/to/a/fresh/workdir
PY_EXPORT=/path/to/export-env/bin/python \
PY_RUNTIME=/path/to/runtime-env/bin/python \
  bash build_all.sh
```
`build_all.sh` runs: **extract** (4 weight extractors, ckpt→npz) → **export** (fixed windowed rungs,
torch→tflite, over the ladder) → **quantize** (each → w8a8) → **merge** (weight-dedup into the 8 canonical
files) → **verify** (structural: ladders + dispatch + shapes). Outputs land in `$SA3_BUILD_WORK/same-{l,s}/`.

## 4. Ladders / design
- SAME-L rungs `{1,2,4,8,12,16,32,64,128,256}`, SAME-S `{2,4,8,12,16,32,64,128,256}` (SAME-S is even-only:
  its attention tiles in 34-token = 2-latent chunks). Small rungs make tiny-L exact + fastest; the {16..256}
  ladder tiles longer L (overlap = the SWA receptive field: SAME-L 12, SAME-S 16 latents → tiling bit-exact).
- Two precision tiers: **`w8a8`** (int8 weights, default — faster + smaller, quality-free on the round-trip)
  and **`fp32`** (bit-exact reference / fallback for CPUs without int8 acceleration).
- Full rationale, validation, and CPU-support notes: `../docs/RUNGS.md`.

## 5. Files
```
build_paths.py     resolves $SA3_BUILD_WORK + checkpoints; makes torch_defs importable
extract/           4 weight extractors (ckpt -> npz)
torch_defs/        checkpoint-faithful torch model defs + windowed_decoder (O(S) attention patch)
                   + limiter.py (output limiter baked into the decoder graph; SA3_BAKE_LIMITER=0 to skip)
export/            torch -> fixed-size tflite rung (decoders get the limiter baked in)
quant_merge/       quant_one (fp32->w8a8) + merge_rungs_generic (fixed rungs -> one weight-shared file)
tfl_surgery.py     flatbuffer helper used by the merge
verify_final.py    structural check of the 8 built files (runtime env)
build_all.sh       orchestrator
```
Quality (round-trip vs ground-truth audio) is a separate check — decode a real clip through enc→dec and
compare to the original; the ear is the gate.
