# LoRA / DoRA on the SA3 TensorRT DiTs

One engine per DiT that serves **any** adapter for it, swapped at runtime without a rebuild.
Adapters load in milliseconds, stack, and have independent strength.

TensorRT bakes the GPU architecture into the engine, so you build it once on the GPU you
intend to serve from. That is what this directory is for.

All three DiTs are served. They are different networks, so an adapter is **not** portable
between them and each gets its own engine and map:

| `--model` | `--dit` | layers × width | targets | engine | build |
|---|---|---|---|---|---|
| `sa3-m` | `medium` | 24 × 1536 | 229 | 2.96 GB | 7.4 min |
| `sa3-sm-music` | `sm-music` | 20 × 1024 | 193 | 0.95 GB | 5.1 min |
| `sa3-sm-sfx` | `sm-sfx` | 20 × 1024 | 193 | 0.95 GB | 5.0 min |

(All measured on an H200. The small engines are +15 MB over their plain `dit_fp16.trt`.)

---

## Build it

```bash
python optimized/tensorRT/scripts/lora/build_branch.py --model sa3-m        --download
python optimized/tensorRT/scripts/lora/build_branch.py --model sa3-sm-music --download
python optimized/tensorRT/scripts/lora/build_branch.py --model sa3-sm-sfx   --download
```

**You need:** an NVIDIA GPU, TensorRT 10.x, `huggingface_hub` (only for `--download`), about
9 GB of free disk for medium (2.9 GB engine + 2.9 GB ONNX) or 2 GB for a small DiT, and
~5 GB of free VRAM.
**It takes** a few minutes, most of it TensorRT searching for kernels, which is silent.
**You get** a `.trt` engine plus a JSON map describing the layers it adapts.

The two lines worth reading in the output are the target counts:

```
[build] 216 block linears = 24 layers x 9 targets  + 13 non-block (all)
[surgery] branch on 229 linears (+688 inputs, rank dynamic 1..512)
```

**229 = 24 layers × 9 + 13 non-block** for medium; **193 = 20 × 9 + 13** for either small
DiT. If that is lower than expected, some layers silently
have no adapter attached — the build asserts rather than letting it through.

```bash
python .../lora/build_branch.py --check     # CI gate: engine ⇄ map agree, non-zero on any problem
```

---

## Use it

```bash
python optimized/tensorRT/scripts/sa3_trt.py --dit medium --lora my_adapter.safetensors
python optimized/tensorRT/scripts/sa3_trt.py --dit sm-sfx --lora my_sfx_adapter.safetensors
python .../sa3_trt.py --dit medium --lora a.safetensors:0.8 --lora b.safetensors:0.5
```

`--lora` is repeatable and takes an optional `:STRENGTH`. Passing it selects the LoRA engine
automatically.

The runtime side is already in place for a UI — `SA3Inference.set_lora()` and
`set_lora_strength()` take a stack and per-adapter strengths, and a same-rank swap keeps every
cached CUDA graph valid — but the gradio app does not expose it yet.

---

## Where things live

Everything resolves relative to this checkout and is overridable by environment variable. A
missing file raises where it is asked for rather than resolving to the wrong model — that is
deliberate, because the failure it prevents is silent: an `-xs` adapter folded against a
different model's SVD bases lands in a rotated basis and nothing raises.

`<slug>` below is `sa3-m`, `sa3-sm-music` or `sa3-sm-sfx`.

| | default | override |
|---|---|---|
| engine | `models/<arch>/<slug>/dit_fp16_lora.trt` | `$SA3_ENGINE_DIR`, `$SA3_MODELS_DIR` |
| branch map | beside the engine, else next to these scripts | `$SA3_BRANCH_MAP` |
| ONNX source | `onnx/<slug>/dit_fp16.onnx` | `$SA3_ONNX_DIR` |
| base checkpoint | `models/<slug-ish>/` (see `paths.MODELS`) | `$SA3_CKPT_DIR` |
| frozen SVD bases | `$SA3_CKPT_DIR/svd_bases.pt` | `$SA3_SVD_BASES` |

Those variables name **sa3-medium**. For the other two, suffix them — `$SA3_CKPT_DIR_SM_MUSIC`,
`$SA3_ENGINE_DIR_SM_SFX` and so on — so that pointing medium somewhere can never silently
retarget a different model. `paths.for_model("sm-sfx")` is the accessor; it takes any of the
three spellings (`sa3-sm-sfx`, `sm-sfx`, the `--dit` name).

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

**The problem.** A TensorRT engine is a compiled plan: the weights are baked in when you
build it. Normally, changing a weight means rebuilding (minutes of kernel search) or
refitting (seconds, but it rewrites 2.6 GB of weights and cannot vary per sampler step).
Neither is a thing you can do while someone moves a slider.

**The trick is to stop treating the adapter as a weight.** A LoRA delta is low rank —
`ΔW = (α/r)·B·A` with `B` being `[out, r]` and `A` being `[r, in]` — so it does not have to
live in the plan at all. It can arrive as *activation data*. The build adds, to each of the
229 adapted linears, three extra **network inputs**:

```
A  [1, R, in ]     the down-projection
Bt [1, R, out]     the up-projection, stored rank-major
P  [1, 1, out]     a per-output-row rescale of the base path
```

plus one `srow [1, 1, R]` shared by the whole network. The adapted linear then computes

```
y = W₀·x·(1 + P)  +  Btᵀ·( srow ⊙ (A·x) )
    └── base ───┘     └──── low-rank branch ────┘
```

`R` is a **dynamic** dimension, 1..512, so one engine serves any rank.

**That is the whole mechanism, and everything else follows from it:**

- **Swapping an adapter is a buffer write.** No rebuild, no refit — microseconds of host
  work plus the fold. The plan never changes, so a captured CUDA graph stays valid as long
  as the buffer *addresses* hold.
- **Stacking is concatenation along `R`.** Two rank-16 adapters become one rank-32 operand
  pair. Because `srow` is indexed per rank-slot, scaling slots 0–15 scales the first adapter
  and 16–31 the second — per-adapter strength for the cost of a 32-element write.
- **`P` is what DoRA needs.** A DoRA is not just a delta: it renormalises each weight row and
  reimposes a learned magnitude, `c = magnitude / ‖W₀ + δ‖_row`. That is a rescale of the
  *base* path, which no amount of low-rank branch can express — hence `P = c − 1`. Without
  it the engine silently computes plain LoRA for a DoRA checkpoint, missing 32% of the
  adapter's effect at 1% magnitude drift and 86% at 5%.
- **The graph does not know what "stacking" means.** It evaluates the formula above and
  nothing else. Any composition of the form `ρ ⊙ W₀ + Σₖ γₖ ⊙ δₖ` fits it — `ρ` folds into
  `P`, `γₖ` folds into that adapter's rows of `Bt` — so the *semantics* of combining
  adapters live entirely in Python and can change without touching the engine. Verified by
  writing an exact chained composition into the same operands and reproducing a dense
  chained merge to 3.3e-16.

**What it costs, and why.** The branch is two small GEMMs per adapted linear — `[S,in]×[in,R]`
then `[S,R]×[R,out]`. At low rank that is memory- and launch-bound rather than arithmetic-
bound, which is why the overhead is a flat few ms per step rather than a percentage, and why
stacked rank is free up to about 32 before the GEMMs start to matter. Being a flat cost, its
*relative* price depends on the DiT: +13% at L=4096 and +37% at 1292 on medium, but +34% /
+60% / +83% at L=4096 / 1292 / 323 on the small DiTs, whose base step is ~2.7× cheaper.

**Rank is rounded up to a multiple of 8, and that is a real speedup.** Those two GEMMs run
~40% faster when the bound rank is a multiple of 8 — 8 fp16 values is 16 bytes, one
vectorised load, and the fp16 tensor-core fragment wants K/N in multiples of 8, so the kernel
only reaches its fast path when the rank dimension fills whole fragments. It is **not** a
threshold: rank 9 and 12 are exactly as slow as rank 1. So `pad_rank()` rounds the
*concatenated* stack rank up and leaves the extra rows zero in `A`, `Bt` and `srow`, which
contributes nothing and returns bit-identical output. Measured on an r4 adapter at L=1292:
9.79 → 8.33 ms/step, max|diff| exactly 0. It holds down to L=8, where it is ~16% of the step.
`rank` as reported is always what the adapters carry; the padded value is `rank_bound`.

The flip side of operands-as-inputs: **TensorRT refuses to enqueue with any input unbound**,
so a branch engine must be handed at least a zero stack before it will run at all — and a
zero stack still pays the full forward cost. There is no "rank 0" either: the profile minimum
is 1 and `set_input_shape` rejects a 0 dimension, so "no adapter" is a padded rank of zeros.
That is why `dit_fp16.trt` remains the default and the LoRA engine is only selected when an
adapter is actually wanted.

**Why not refit instead?** Refitting the real weights gives a 0% forward overhead and
supports every adapter variant including the ones this branch cannot express
(`dora-cols`, `bora`, which need a per-*input* rescale). It costs a multi-second weight
rewrite per change and cannot do per-step strength at all. The crossover is around 36
generations per adapter: below that the branch wins, above it the refit does. The branch is
the interactive path.

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
