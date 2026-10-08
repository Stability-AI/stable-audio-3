# LoRA / DoRA on the SA3-medium TensorRT DiT

One engine that serves **any** adapter, swapped at runtime without a rebuild. Adapters load
in milliseconds, stack, and have independent strength.

TensorRT bakes the GPU architecture into the engine, so you build it once on the GPU you
intend to serve from. That is what this directory is for.

---

## Build it

```bash
python optimized/tensorRT/scripts/lora/build_branch.py --model sa3-m --download
```

**You need:** an NVIDIA GPU, TensorRT 10.x, `huggingface_hub` (only for `--download`), about
9 GB of free disk (2.9 GB engine + 2.9 GB ONNX) and ~5 GB of free VRAM.
**It takes** a few minutes — measured at **7.4 min on an H200**, most of it TensorRT
searching for kernels, which is silent.
**You get** a 2.96 GB `.trt` engine plus a JSON map describing the layers it adapts.

The two lines worth reading in the output are the target counts:

```
[build] 216 block linears = 24 layers x 9 targets  + 13 non-block (all)
[surgery] branch on 229 linears (+688 inputs, rank dynamic 1..512)
```

**229 = 24 layers × 9 + 13 non-block.** If that is lower than expected, some layers silently
have no adapter attached — the build asserts rather than letting it through.

```bash
python .../lora/build_branch.py --check     # CI gate: engine ⇄ map agree, non-zero on any problem
```

---

## Use it

```bash
python optimized/tensorRT/scripts/sa3_trt.py --dit medium --lora my_adapter.safetensors
python .../sa3_trt.py --dit medium --lora a.safetensors:0.8 --lora b.safetensors:0.5
```

`--lora` is repeatable and takes an optional `:STRENGTH`. Passing it selects the LoRA engine
automatically. The gradio app exposes the same thing as `lora_load`, `lora_unload` and
`lora_strength` API endpoints.

---

## Where things live

Everything resolves relative to this checkout and is overridable by environment variable. A
missing file raises where it is asked for rather than resolving to the wrong model — that is
deliberate, because the failure it prevents is silent: an `-xs` adapter folded against a
different model's SVD bases lands in a rotated basis and nothing raises.

| | default | override |
|---|---|---|
| engine | `models/<arch>/sa3-m/dit_fp16_lora.trt` | `$SA3_ENGINE_DIR`, `$SA3_MODELS_DIR` |
| branch map | next to these scripts | `$SA3_BRANCH_MAP` |
| ONNX source | `onnx/sa3-m/dit_fp16.onnx` | `$SA3_ONNX_DIR` |
| base checkpoint | `models/sa3-medium/` | `$SA3_CKPT_DIR` |
| frozen SVD bases | `$SA3_CKPT_DIR/svd_bases.pt` | `$SA3_SVD_BASES` |

The engine goes where every other engine in this install goes — `models/<arch>/<model>/`,
since TensorRT bakes the GPU architecture into the plan and one install can hold several
side by side. The builder and the runtime read the **same** constant from `paths.py`, so
they cannot drift apart.

Only `bake_dora.py` and the `verify_*` harnesses need the checkpoint. The SVD bases are
needed **only** by `-xs` adapters. Normal adapter loading touches neither.

---

## Adapter types

The branch serves four of the eight variants:

| | |
|---|---|
| ✅ `lora`, `lora-xs`, `dora-rows`, `dora-rows-xs` | |
| ❌ `dora-cols`, `dora-cols-xs`, `bora`, `bora-xs` | the branch's rescale is per **output** row; a column normalisation needs a per-**input** scale, which is a different operand and therefore a rebuild |

**DoRA adapters must be baked first.** The fold needs `c = magnitude / ‖W₀ + s·LR‖_row`, and
computing that denominator means loading all 10.5 GB of base weights purely to take a row
norm. Baking it into the adapter removes that from every start:

```bash
python .../lora/bake_dora.py my_dora.safetensors        # -> my_dora.normbaked.safetensors
```

Plain `lora` adapters never need it.

---

## One layer is always ignored

```
RuntimeWarning: my_adapter.safetensors: adapts 1 EXCLUDED layer(s) --
conditioners.seconds_total.embedder.embedding.1 -- which are IGNORED.
```

Expected, and the adapter is fine to use. The seconds embedder is **global conditioning** —
its output modulates all 24 blocks — so a stack of two adapters that both rescale it inverts
the model's gain and renders **silence**, not noise. It is dropped from every adapter, for
every engine. The cost is 0.6–17.75% of that one adapter's effect. Retrain without it and
the warning goes away.

---

## Stacking

Stacked adapters are concatenated along the rank axis. For `lora` / `lora-xs` that is exact
and order-free. For DoRA the per-row rescales are **summed**, which matches
`merge_loras_into_base_model` and not `load_and_apply_loras`' chaining.

**That distinction only matters when the adapters are overfit**, and it was ear-tested:

| | mean `c` | ρ summed | ρ chained | audible difference |
|---|---|---|---|---|
| trained pair | 0.987 | 0.975 | 0.983 | none |
| overfit pair | 0.492 | **−0.016** | 0.427 | 14 dB, and 8 dB *below the base model* |

`c` is a DoRA's per-row rescale; a well-trained adapter sits near 1. On the overfit pair the
summed coefficient goes **negative on average** — the base weight is cancelled rather than
adapted, so the stack renders quieter than using no adapter at all. The runtime warns when
more than 1% of rows invert. **If you see that warning the adapter is the problem, not the
composition rule.**

---

## What it costs

Per DiT step on an H200, against the plain no-LoRA engine. Roughly **flat ~4–6 ms**, so it
matters least where it costs most:

| sequence length | plain | with LoRA engine | overhead |
|---|---|---|---|
| 323 | 7.4 ms | 11.1 ms | +51% |
| 1292 | 11.7 ms | 16.0 ms | +37% |
| 4096 | 40.4 ms | 45.8 ms | **+13%** |

You pay this with a zero stack too, so run the plain engine until an adapter is selected.

**Stacked rank is nearly free up to 32**, then grows (L=4096, measured against rank 8):

| Σrank | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| step time | — | 1.00× | 1.00× | 1.01× | 1.06× | 1.13× | 1.29× |
| operands | 21 MB | 41 MB | 83 MB | 166 MB | 331 MB | 663 MB | 1.33 GB |

Operand VRAM is **2.589 MB per unit of stacked rank**, exactly. The cap is 512 and over it
the load is refused rather than silently mis-bound.

**Live operations** at L=1292: load 1–3 adapters 110–400 ms (⅔ of it the host safetensors
parse, so it tracks adapter *bytes* not rank), strength change **6.5 ms**, same-rank swap
~157 ms. Changing *rank* also forces a CUDA-graph recapture (~0.6–0.9 s) on the next render;
changing only strength or weights does not.

---

## How it works

A low-rank branch on each adapted linear:

```
y = W₀·x·(1 + pout)  +  Bp·(srow ⊙ (A·x))
```

`A`, `Bt` and `P` are **network inputs**, not weights — so swapping an adapter is a buffer
write, not a rebuild, and `srow` gives each adapter in a stack its own strength for the cost
of an R-element write.

The graph never learns what "stacking" means. Any composition of the form
`ρ ⊙ W₀ + Σ γₖ ⊙ δₖ` fits it, with ρ folding into `P` and γₖ into that block's `Bt` rows —
verified to 3.3e-16 against an exact chained merge. The semantics live entirely in Python.

---

## Files

| | |
|---|---|
| `build_branch.py` | builds the engine; `--check` is the CI gate |
| `targets.py` | which 229 linears get a branch, and how to find them in the ONNX |
| `trt_build.py` | STRONGLY_TYPED builder + optimization profile |
| `branch_runtime.py` | the branch: operand fold, rank cap, finite guards |
| `refold_runtime.py` | `RefoldLora` — what the runtime actually uses |
| `merge.py` | `load_stack`, `EXCLUDED_LAYERS`, the reference merge |
| `bake_dora.py` | bakes a DoRA's row norms |
| `paths.py` `constants.py` `dit_loader.py` | path resolution, input shapes, the torch reference wrapper |
| `verify_medium.py` `verify_stack.py` | single adapters and stacks vs torch |
| `grep_gate.sh` | pre-publish leak check |

The merge math itself is **not** here: `optimized/tflite/scripts/lora_core.py` already
defines what an adapter means for all 8 variants, and is imported rather than forked.

---

## When it goes wrong

| message | what to do |
|---|---|
| `no ONNX found at …` | add `--download`, or point `$SA3_ONNX_DIR` at your ONNX directory |
| `… is small and …data is missing` | the weights live in a separate `.onnx.data` that must sit next to the proto — re-run with `--download` |
| `… already exists` | add `--force`, or build to a different `--engine` path |
| `expected 229 targets …, got N` | the ONNX node naming drifted; see `classify()` in `targets.py` |
| `N% of rows have a SIGN-FLIPPED base coefficient` | the stack cancels the base weight — see **Stacking** above. Lower strengths, drop the most-drifted adapter, or retrain nearer `c=1` |
| `… latent values are NaN/Inf -- this would have decoded to silence` | fp16 overflow inside the DiT. Lower strengths, or render with no LoRA to confirm the base engine is clean |
| `operand loraB::… has N non-finite entries` | the **fold** overflowed, usually a large strength on a drifted DoRA |
| `stack rank N exceeds this engine's maximum` | load fewer or lower-rank adapters, or rebuild with `--rank-max` |
| `… adapter has no baked row norms` | run `bake_dora.py` on it |
| `frozen SVD bases missing` | an `-xs` adapter needs the **same** basis file it was trained against; set `$SA3_SVD_BASES` |
| `only the LEGACY name is present` | your ONNX clone predates the `fp16mixed` → `fp16` rename. They are **different graphs** — fetch the current one with `--download` |

---

## Options

| flag | default | |
|---|---|---|
| `--targets` | `all` | `core` (169) / `full` (228) / `all` (229) |
| `--rank-max` | `512` | ceiling on the **total stacked** rank, not per adapter |
| `--rank-opt` | `32` | the rank TensorRT tunes kernels for |
| `--onnx` `--engine` `--map` | from the preset | explicit paths |
| `--workspace-gb` | `48` | build-time scratch |

Raising `--rank-max` is nearly free: 128 → 512 costs **+5.3 MB** of engine file (measured),
~20 MB of scratch and 0.1% of step time.

The design, the measurements and the traps are written up in
[`BUILDING_TRT_LORA.md`](BUILDING_TRT_LORA.md).
