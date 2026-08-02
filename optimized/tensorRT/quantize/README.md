# SA3 Encoder + Decoder — Quantized Tiers (SAME-S / SAME-L)

Train-free quantized variants of the Stable Audio 3 **encoders and decoders**, each a single
**self-contained ONNX** derived from the published bf16 export. TensorRT builds the engine; quality is
preserved by GPTQ weight-error compensation and per-op precision placement (details below).

Two knobs, three tiers:

- **`fp8`** names a **compute** format — the GEMM runs on fp8 tensor cores → real speedup.
- **`w8_bf16`** names a **weight** format — int8 weights, **bf16 compute**. Storage only: smaller file, *no* speedup.
- **`_fast`** = fp8 pushed into the attention projections too — faster, lower quality.

So `dec_fp8` is *faster*; `dec_w8_bf16` is just *smaller*.

---

## Tiers at a glance

dB is PSNR vs the **bf16 baseline** (a heuristic — confirm the near-transparent tiers by ear;
`_fast` will have audible artifacts). Speed is whole-decoder median at seq 1292 (~2 min audio).

### SAME-S — baseline `dec_dynamic_bf16.onnx`: 7.25 ms @1292 · 23.2 ms @4096 · 218 MB

| file | quantized | compute | size | speed | dB vs bf16 (music / sfx) |
|---|---|---|---|---|---|
| `dec_w8_bf16.onnx` | int8 weights (all linears) | bf16 | 59 MB | 1.00× | 38.2 / 50.0 · transparent |
| `dec_fp8.onnx` | fp8 weights + activations (FFN) | fp8 FFN GEMM | 58 MB | **1.14×** | 30.9 / 41.0 · near-transparent |
| `dec_fp8_fast.onnx` | fp8 weights + activations (all linears) | fp8 all GEMMs | 58 MB | **1.22×** | 26.1 / 36.5 · lossy |

### SAME-L — baseline `dec_dynamic_triton_swa.onnx`: 54.0 ms @1292 · 174.4 ms @4096 · 1192 MB

| file | quantized | compute | size | speed | dB vs bf16 |
|---|---|---|---|---|---|
| `dec_w8_bf16.onnx` | int8 weights (FFN) | bf16 | 937 MB | 1.00× | 51.4 · transparent |
| `dec_fp8.onnx` | fp8 weights + activations (FFN) | fp8 FFN GEMM | 937 MB | **1.15×** | 43.4 · near-transparent |

SAME-L has **no `_fast` tier**: its attention projections are fp32 islands (fp8 there collapses quality — see islands below).

### Encoders (same tiers, same grafters)

The encoders share the decoders' architecture, so the same tiers apply — but the **encoder is cheap**
(~1 ms SAME-S, ~7 ms SAME-L), so quant buys **size, not speed**. `w8_bf16` is the useful encoder tier.
dB/cos vs eager encode (the latent):

| file | SAME-S enc | SAME-L enc |
|---|---|---|
| `enc_w8_bf16.onnx` | 36.2 dB · cos 0.997 · transparent | 44.5 dB · cos 0.9995 · transparent |
| `enc_fp8.onnx` | 29.9 dB · cos 0.986 · near-transparent | 30.8 dB · cos 0.989 · near-transparent |
| `enc_fp8_fast.onnx` | 22.3 dB · cos 0.924 · lossy | — (fp32 islands) |

Encoder input is audio `(1,2,N)`; output latent `(1,256,N/4096)`. **Every encoder onnx carries a
silence-pad node** (see below), so any `N` is accepted.

---

## Sequence length, padding & building

The onnx graphs are **fully dynamic** — every model is correct at **any** length (verified L=1→8192
vs eager, odd/even/prime, no size-dependent error; quant tiers track bf16 identically at all lengths).
The only bound is the **TensorRT optimization profile** you build with. Two rules:

- **Build wide.** Use `min=1` and a `max` covering your longest track. Reference build uses
  **latent `[1, 1292, 8192]`** (decoders, ≈12:40) and **audio `[1, 2097152, 33554432]`** (encoders).
  `L=1–31` then work and match eager (these can't be chunked); long files run **natively, no chunking**
  (a 12:40 decode / 6:20 encode is a single call). The old `[32,4096]` was just a conservative default.
- **Encoders auto-pad.** The SAME-L encoder needs the audio length to be a multiple of 4096 (the
  downsample ratio); the built-in pad node silence-pads the tail up to the next multiple, matching what
  eager does internally. Callers feed any `N`. (SAME-S never needed it; padded anyway for a uniform API.)

The `encode_chunked` path in the runtime is a **stale workaround** — it predated wide profiles and was
chasing what turned out to be out-of-profile garbage, not a real divergence. Single-shot is accurate at
any length (SAME-L enc cos 0.9995 on a real 285 s track). Chunking is unnecessary.

---

## Where the quantization goes, per tier

Every decoder block is `latent_proj → [ attn: to_qkv → (QKᵀ·softmax·V) → to_out ] → [ ff: ff.0 → GELU → ff.2 ]`,
with RMSNorm + RoPE around attention. Each tier touches a different subset:

### `dec_w8_bf16` — int8 weight-only
- **Weights:** every linear (`ff.0`, `ff.2`, `to_qkv`, `to_out`, `latent_proj`) → int8, per-output-channel scale, **GPTQ**-compensated (Hessian captured from real-audio activations). *(SAME-L: the 24 FFN linears; attention projections stay fp32 islands.)*
- **Activations / attn core / norms / RoPE:** untouched.
- Weights dequantize to bf16 at build → **bf16 GEMMs**. int8 is storage only ⇒ smaller download, bf16 speed, the most transparent tier.

### `dec_fp8` — fp8 on the FFN
- **FFN** (`ff.0` up-proj, `ff.2` down-proj) `+ latent_proj`: **weight fp8 + activation fp8** (per-tensor clipping scale) → **fp8 GEMM**. Weights GPTQ-compensated.
- **Attention projections** (`to_qkv`, `to_out`): SAME-S → **weight-only fp8** (bf16 compute — kept out of fp8 math because they feed softmax); SAME-L → **fp32 islands** (untouched).
- **Attention core** (QKᵀ·softmax·V): bf16 flash (unchanged).
- **RMSNorm / RoPE:** fp32.

### `dec_fp8_fast` — fp8 everywhere it pays (SAME-S only)
- **All linears** (FFN + `to_qkv` + `to_out` + `latent_proj`): **weight fp8 + activation fp8** → fp8 GEMM.
- **Attention core** still bf16 — but now it's fed **fp8-rounded Q/K/V** from the projections. *That rounding is the quality cost* (the softmax path is fp8-sensitive), buying +0.08× speed over `dec_fp8`.
- **RMSNorm / RoPE:** fp32.

---

## The levers (reference)

### What you quantize — the two halves of every GEMM

| lever | format | compute or storage | speed | quality | notes |
|---|---|---|---|---|---|
| **Weights** | fp8 (per-tensor*) or int8 (per-channel) | **storage** (dequant→bf16 at build) | none | near-lossless; **GPTQ** compensates rounding | 1 byte either format — fp8 ≠ smaller than int8 |
| **Activations** | fp8 only (dynamic, per-tensor **clip** scale) | turns the GEMM into fp8 tensor-core compute | **unlocks the speedup** | the **lossy half** (~31 dB floor; GPTQ can't fix it) | calibration-critical — amax scale → collapse; use a clipping scale |

\* fp8 weight scale **must be per-tensor in a strongly-typed build** (per-channel silently collapses the GEMM). int8 is happily per-channel. int8 *activations* are a non-starter — softmax breaks.

### Where you apply it — which op

| lever | touches | fp8? | speed | quality | verdict |
|---|---|---|---|---|---|
| **FFN GEMM** | `ff.0`, `ff.2` | ✅ weight+act | 1.14× (both) / 1.07× (ff.0 only) | ff.0 transparent; **ff.2 sensitive** (±outliers) | the main speed win → `dec_fp8` |
| **Attn projections** | `to_qkv`, `to_out` | ✅ weight+act | +0.08× on top | lossy — fp8-rounds Q/K/V *before* attention | `dec_fp8_fast` (SAME-S) |
| **Attn core** | QKᵀ·softmax·V | ❌ no fp8 MHA kernel @ hd=64 | ~none | — | **dead at hd=64** (Sage 0.46–0.97×, fully-fp8 softmax slower); the real lever is algorithmic **windowing** (~1.6× on SAME-L long-seq), not quant |

### Leave-alone islands (constraints, not levers)

| op | rule | why |
|---|---|---|
| **RMSNorm** | keep fp32/bf16 | cheap; quantizing buys nothing |
| **RoPE** | **must stay fp32** | bf16 RoPE angle → long-sequence clip bug (≥2 min renders) |
| **Attn projections (SAME-L)** | fp32 islands | fp8 compute there collapses quality in the strongly-typed graph |

The mental model: **activations = speed (and the quality floor); weights = size (near-free with GPTQ); attn core = neither (use windowing).**

---

## Which to pick

- **Smallest + most transparent, speed doesn't matter** → `dec_w8_bf16`
- **Faster, near-transparent (default)** → `dec_fp8`
- **Fastest, quality-tolerant** → `dec_fp8_fast` (SAME-S)

## Reducing SAME-L size (optional)
Both SAME-L tiers are 937 MB because the attention projections stay fp32 (680 MB). Weight-only-quantizing
those islands (int8/fp8 weights, bf16 compute — no speed or quality cost) pulls both to ~500 MB.

## Building the engines

The tier **ONNX files ship on HuggingFace** (`stabilityai/stable-audio-3-optimized`, alongside the
existing `dec_dynamic_bf16.onnx` / `enc_dynamic_*`). To build an engine with the wide profile:

```bash
python build_tiers.py <tier.onnx> <out.trt> --arch {same-s|same-l} --kind {enc|dec} [--fp8]
# SAME-S fp8 / fp8_fast tiers need --fp8; SAME-L carries fp8 in-graph (strongly-typed) and needs no flag.
```

`build_tiers.py` bakes the **wide** profile (decoder `latent [1,1292,8192]`, encoder `audio [1,·,33.5M]`),
so L=1..31 work and long files run natively. SAME-L uses the `diff_attn_swa` plugin (PREFER_JIT).

## Reproducing the tier ONNX (grafters)

The quantized ONNX are produced by grafting QDQ / int8 / fp8-stored weights onto the published bf16 export:

| grafter | produces |
|---|---|
| `fp8_gptq.py` | SAME-S `dec_fp8` / `dec_fp8_fast` (fp8-stored + GPTQ, ORT Hessian capture) |
| `fp8_gptq_samel.py` | SAME-L `dec_fp8` (fp8-stored + GPTQ, eager Hessian capture — SWA blocks ORT) |
| `gptq_w8.py` / `gptq_samel.py` | `dec_w8_bf16` (int8 weight-only + GPTQ) |
| `enc_quant_sames.py` / `enc_quant_samel.py` | the encoder tiers |
| `pad_encoder.py` | adds the silence-pad node so any audio length is accepted |

> **Note:** the grafters reference campaign-local calibration data (real-audio latents) and model
> checkpoints via absolute paths at the top of each file — adjust those to reproduce. Calibration +
> Hessians are computed from real-audio latents. The ready-made ONNX on HF need none of this.
