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

## File map

| File | Role | Flow |
|---|---|---|
| `build.py` | Interactive menu (default entry point) | consumer |
| `build_from_onnx.py` | One target → download ONNX from HF + compile to TRT. **For the SA3 DiTs, pulls `dit_fp16mixed.onnx` (the pre-processed island-wrapped graph)** so the consumer just needs to invoke `STRONGLY_TYPED` compilation — no `onnx-graphsurgeon` required | consumer |
| `build_dit_profile.py` | Build a DiT with custom `(min, opt, max)` profile shapes (experimental — short-form / fixed-shape variants). Operates on either ONNX flavor. | consumer |
| `build_dit_fp16mixed.py` | **Producer-side** ONNX surgery: takes the canonical FP32 `dit.onnx`, finds RMSNorm chains + attention `Softmax` + RoPE region, wraps each in `Cast(FP32) ↔ Cast(FP16)` islands, converts non-island weights to FP16, then bounds the RoPE island before QK^T (`bound_attention_core()`, `--no-bound-attn` to skip) so the attention core runs FP16 and TRT's FMHA fuser fires — 96/96 attentions on the medium DiT, 4.3× at L=4096. Writes both the modified `dit_fp16mixed.onnx` AND the TRT engine, which **must** be `STRONGLY_TYPED` (weakly-typed + `BuilderFlag.FP16` re-casts the FP32 islands and silently degrades to naive FP16). Only re-run when the model retrains or the island recipe changes. Requires `onnx` + `onnx-graphsurgeon`. | producer |
| `build_t5gemma.py` | Trace + export T5Gemma encoder ONNX + build TRT | producer |
| `build_same_s_decoder.py` | Trace + export SAME-S decoder ONNX + build TRT | producer |
| `build_same_s_encoder.py` | Trace + export SAME-S encoder ONNX + build TRT | producer |
| `build_same_l_decoder.py` | Trace + export SAME-L decoder ONNX (Triton SWA) + build TRT | producer |
| `build_same_l_encoder.py` | Trace + export SAME-L encoder ONNX (Triton SWA) + build TRT | producer |
| `build_dit.py <NAME>` | Trace + export DiT FP32 ONNX (cond baked in) + build TRT BF16 engine (legacy; the BF16 output isn't suitable for inference — chain it with `build_dit_fp16mixed.py` afterwards) | producer |
| `_arch.py` | Shared: GPU arch detection + path helpers | both |
| `samel_loader.py` | Helper: load SAME-L from .ckpt | producer |
| `samel_{encoder,decoder}_onnx.py` | Helper: clean ONNX rewrites of SAME-L blocks | producer |
