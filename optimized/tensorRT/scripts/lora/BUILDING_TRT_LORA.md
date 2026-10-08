# Building a TensorRT DiT with live LoRA/DoRA support

Everything learned building runtime-swappable LoRA into the SA3 TensorRT DiTs
(medium = 229 targets, large = 182). Written so the next person does not re-pay for any of it.

> Scope note: this covers the **branch** approach — adapters as network *inputs*, swappable at
> runtime with no rebuild. The alternative (merge weights + `refit`) costs 0% forward but needs
> a refit per adapter; see "Why branches, not refit" at the end.

---

## 1. The core idea

Replace every adapted linear `y = W₀·x` with

```
y = W₀·x · (1 + pout)  +  Bp · ( srow ⊙ (A · x) )
```

where **`A`, `Bp` (as `Bt`) and `pout` (as `P`) are network INPUTS, not weights**. That single
decision is what makes adapters hot-swappable: the engine is built once, and loading an adapter
is a buffer write. `srow` is a per-rank-block strength vector, so a stack of adapters gets
per-adapter strength for the cost of an R-element write, with the captured graph replayed
unchanged.

Mapping to the adapter families:

| family | `A` | `Bp` | `pout` |
|---|---|---|---|
| `lora` | `lora_A` | `scaling · lora_B` | 0 |
| `lora-xs` | `Vᵀ` | `scaling · U·M_xs` | 0 |
| `dora-rows` | `lora_A` | `c ⊙ (scaling · lora_B)` | `c − 1` |
| `dora-rows-xs` | `Vᵀ` | `c ⊙ (scaling · U·M_xs)` | `c − 1` |

with `scaling = lora_alpha / rank` and

```
c = magnitude / ‖W₀ + scaling·δ‖_row        ← the DoRA row rescale
```

At rank 1 with one adapter this reproduces the canonical parametrisation exactly:
`W = c ⊙ (W₀ + scaling·δ)`.

### Design decisions that were paid for in prototypes

* **Per-layer inputs, NOT one grouped tensor sliced per layer.** A dynamic slice costs a real
  copy plus a fusion-blocking shape chain: measured **+194% vs +16%** forward overhead.
* **`Bt` stored RANK-MAJOR `[R, out]`** and consumed with `MatrixOperation.NONE`. Feeding a
  column slice of an `[out, R]` buffer silently computes the wrong branch, and fp16 hides it.
* **`srow` broadcast over S** as `[1, 1, R]`.

---

## 2. Choosing what to branch

Enumerate every 2-D weight in the checkpoint — that is the set of adaptable linears. For
sa3-medium: **226 linears + 2 convs inside the DiT, + 1 in the conditioner = 229**.

```
24 × 9 block linears           self_attn.to_qkv/to_out, cross_attn.to_q/to_kv/to_out,
                               ff.ff.0.proj, ff.ff.2, to_local_embed.0/.2      = 216
to_timestep_embed.0/.2, to_cond_embed.0/.2, to_global_embed.0/.2,
global_cond_embedder.0/.2, project_in, project_out                             =  10
preprocess_conv, postprocess_conv                                              =   2
conditioners.seconds_total.embedder.embedding.1                                =   1
```

**Survey the actual adapters before deciding.** Across 181 real sa3-medium adapters in both run
trees: 100% target the 168 core block linears **and the seconds-total embedder**; ~10% also
target `to_local_embed` / the embed stack / the convs. Building for only the core 168 leaves a
layer every adapter trains unserved — worth **0.6–17.75%** of an adapter's effect, and it grows
with training.

**Full coverage is free.** 168 → 229 targets measured: +7 MB engine file, +5 MB operands at
r16, no measurable VRAM, ≤1 ms/step at L=4096 (four builds were within 1.46 ms of each other in
an interleaved run). Build for everything.

---

## 3. Three activation layouts, one operand layout

Not every adapted op is `[1, S, D]` fp16. Converge them all so the runtime's buffer writes and
the optimisation profile stay uniform at fp16 `[1, R, *]`:

| kind | example | activation | handling |
|---|---|---|---|
| sequence | block linears | `[1, S, D]` | direct |
| rank-2 | seconds embedder (`/Gemm`) | `[1, 256] → [1, 768]` | reshape to `[1,1,D]`, back after |
| conv | `preprocess_conv` | NCHW `[1, C, L, 1]` | transpose to `[1, S, C]`, back after |

Traps:

* **fp32 islands.** The seconds embedder lives in the fp32 sinusoidal chain
  (`Clip→Div→Mul→Cos/Sin→Concat→Gemm`). Feeding fp16 operands there makes TRT build a
  mismatched elementwise — it *warns*, then computes in the wrong type. Cast operands to the
  target tensor's dtype at the branch.
* **Convs must be bias-free.** A TRT `CONVOLUTION` fuses its bias, so a DoRA `pout` rescale on
  its output would scale the bias — which canonical DoRA never does, since it parametrises
  `.weight` only. SA3-medium's two convs are kernel-1×1 with `bias_count=0`, which is the only
  reason they are safe to branch. **Re-check this if the architecture changes.**
* Linears parsed from ONNX `Gemm`/`MatMul` keep their bias as a *separate* add, so `T` is
  pre-bias and the rescale is correct there.
* Check whether the target's output is a **network output** before rewiring consumers
  (SA3's convs feed a squeeze, so they are not — but `postprocess_conv` is the last op and
  looks like it should be).

---

## 4. Build recipe

```bash
python build_branch.py --onnx <dit.onnx> --engine <out.trt> --map <map.json> \
       --targets all --rank-max 512 --rank-opt 32
```

`--targets core|full|all` = 169 / 228 / **229**. Node matching: block linears by
`/layers.N/` + a fragment; non-block by **exact** layer name. Assert that every layer
contributes the full per-layer set and that every named extra was found — an unmatched extra is
otherwise silent (engine builds, adapter loads, that layer just never gets its branch).

The map records, per target: `in`, `out`, `ckpt_suffix` (for SVD-bases lookup) and
**`adapter_key`** (absolute). The absolute key matters because targets do not share one prefix —
block linears are `model.transformer.…`, the embed stack is `model.…`, the seconds embedder is
`conditioners.…`.

### Rank ceiling

`--rank-max` caps the **concatenated stack rank**, not per-adapter rank. Cost of raising it:

```
ceiling 128 → 512:  +5 MB engine file, +20 MB scratch, +0.1% step time at equal loaded rank
```

Nearly free — build wide. The cost that matters is the rank you *load*:
**2.59 MB/rank** (229 targets), time flat to rank ~64, +5.7% at 128.

Multiple optimisation profiles (one per rank band) are the fallback if a ceiling ever does get
expensive; scratch is committed per profile. Not needed at 512.

---

## 5. DoRA specifics

### `c`, and why it matters everywhere

`c = magnitude / ‖W₀ + scaling·δ‖_row` is the per-row factor applied *after* the update.
`c = 1` ⟺ magnitude equals the norm the update already produced ⟺ DoRA has degenerated to LoRA
on that row. Training starts at exactly `c = 1` (magnitude init = `‖W₀‖_row`, δ = 0) and drifts.

Measured on real adapters (774,912 rows each):

| adapter | c mean | c p5 | c p95 | c<0 |
|---|---|---|---|---|
| overfit A — dora r16 | 0.292 | 0.103 | 0.486 | 0.1% |
| overfit B — dora-xs r16 | 0.242 | 0.049 | 0.428 | 2.1% |
| overfit C — dora-xs r128 | 0.513 | 0.400 | 0.677 | 0% |
| **trained — dora r16** | **0.984** | 0.975 | 0.993 | 0% |

A heavily-trained adapter can push the low-rank delta *larger than the base weight*
(`|LR|/‖W₀‖ = 1.27` measured), inflating `‖W₀+δ‖` while magnitude stays near `‖W₀‖` — so `c`
lands near 0.3, i.e. DoRA shrinks each row ~3× after the update. **Anything that assumes
`c ≈ 1` breaks on such adapters.** Two things in this system did.

### Baked norms

`bake_dora.py` precomputes `baked_vnorm_row = ‖W₀ + scaling·δ‖_row` into the adapter so the
runtime never loads base weights (~10 GB). Gate: `need_w0 = type.startswith("dora") and
"baked_vnorm_row" not in p`.

* **`magnitude` trains NEGATIVE** — 16,442 rows in one real adapter. The recovered row norm is
  `|magnitude|`; comparing against the signed value reports a constant rel error of exactly 2.0
  on every negative row and hides everything else behind it.
* The norm is **base-model specific**. Record which checkpoint it was computed against; medium's
  `base` and `arc` reconstruct each other's norms at rel ~2.2e-3, so an unlabelled norm makes a
  variant mismatch undetectable.
* Ideal: bake at *training* save time (`‖W₀‖` and δ are both in hand); ~3.5 MB fp32 for 229
  layers. Watch `save_lora_safetensors`' blanket `.half()` — fp16 storage of the norm costs a
  measured worst-case rel 4.9e-4 on `c`.

### Strength

The shipping knob is a **linear fade in weight space**: scale **both** `pout` and `Bp` by `s`.

```
W(s) = W₀·(1 + s(c−1)) + s·c·δ = W₀ + s·(W_full − W₀)     exact at s=0 and s=1
```

Scaling only `srow` silently degrades DoRA to plain LoRA (misses 32% of the adapter's effect at
1% magnitude drift, 86% at 5%, and no strength value recovers it).

**The magnitude must fade too.** Canonical (`V = W₀ + s·δ; W = magnitude·V/‖V‖`) scales only the
*direction*, so as `s→0` it tends to `magnitude·Ŵ₀`, not `W₀` — measured **rel 0.644 from W₀ at
s=1e-4**. Fix: `mag(s) = ‖W₀‖_row + s·(magnitude − ‖W₀‖_row)`.

Exact per-step DoRA strength (recomputing `‖V(s)‖` each step) needs `W₀` — i.e. the ~10 GB the
baking exists to avoid. Bake `n0`, `d`, `g` per layer if you want it base-weight-free.

---

## 6. Stacking

A stack concatenates along the rank axis: `A = [A₁;…;A_K]`, `Bp = [Bp₁|…|Bp_K]`,
`srow = [s₁·1_{r₁},…]`, and `pout = Σ sₖ(cₖ−1)`. So

```
W = W₀ ⊙ (1 + Σ sₖ(cₖ−1))  +  Σ sₖ·cₖ ⊙ δₖ
```

**This is bit-exact (1.5e-16 in fp64) against the repo's `merge_loras_into_base_model()`**,
whose `application_weight` is exactly our per-adapter strength. It is **not**
`load_and_apply_loras()`, which registers one parametrisation per adapter and therefore
**chains** them: `W = c₂′⊙(c₁⊙(W₀+δ₁) + δ₂)` with `c₂′` normalised against the already-rescaled
weight. The two torch paths disagree with **each other** by rel 1.27 on a real adapter pair —
this is a semantics choice, not a defect. Chaining is order-dependent; summing is not.

The gap between them is the product of the deviations, `∏cₖ − (1+Σ(cₖ−1))` → `(c₁−1)(c₂−1)` for
two. So adapters near `c=1` make it vanish (a trained adapter: 2.6e-04) and drifted ones make it
enormous (an overfit pair: 0.45, with the base-weight coefficient going 0.20 under chaining vs −0.35
under summing — both drastic).

Verified exact: **any number of `lora`/`lora-xs`, to rank 384**; one DoRA alone; mixed families
with different layer sets.

### ⚠ Our own two engines disagree here

The **branch** engine sums (above). The **merge/refit** engine does not: `merge.merge_layer` /
`merge_layer_torch` feed each adapter the *previous* adapter's merged weight as its base, which
is chaining — the same as `load_and_apply_loras()` and the same as the shipped MLX path. So for a
stack of ≥2 DoRAs the two SA3-medium LoRA engines compute different weights (rel ~1.27 on a real
an overfit pair). Single adapters and any number of `lora`/`lora-xs` are unaffected.

### Exact chaining IS reachable from the branch operands (derived, not built)

The obstacle has always been that `c₂′ = m₂/‖W₁ + s₂δ₂‖_row` normalises against the **composed**
matrix, so it seemed to need `W₀` at fold time — the 10.5 GB the baked norms exist to avoid. It
does not. DoRA's own invariant gives it away: after adapter 1, `‖W₁ᵢ‖ = |m₁ᵢ|` **exactly**, since
that is what the reparametrisation sets. Expanding the second norm:

```
‖W₁ + s₂δ₂‖²_row = m₁²  +  2 s₂ c₁ (⟨W₀, δ₂⟩_row + s₁⟨δ₁, δ₂⟩_row)  +  s₂²‖δ₂‖²_row
```

and then `W₂ = c₁c₂′ ⊙ W₀ + c₁c₂′s₁ ⊙ δ₁ + c₂′s₂ ⊙ δ₂` — all per-row scalars, so the branch can
express it with no new operands: `pout = c₁c₂′ − 1`, `Bp₁ ← c₁c₂′s₁B₁`, `Bp₂ ← c₂′s₂B₂`. K > 2
recurses the same way (each step's row norm is the previous adapter's magnitude).

Every term is cheap or already present:

| term | where it comes from |
|---|---|
| `m` | the adapter |
| `‖δ‖²_row` | the low-rank factors (`refold_runtime` already computes it as `g`) |
| `⟨W₀, δ⟩_row` | needs `W₀` **once, at bake time** — same pass `bake_dora.py` already makes for `baked_vnorm_row` (`refold` computes it as `d`) |
| `⟨δⱼ, δₖ⟩_row` | cross-adapter, from the factors alone: `rowsum(Bⱼ (AⱼAₖᵀ) Bₖᵀ)`, one small GEMM per layer per pair |

Measured on an overfit r16 + r4 pair, base-weight coefficient vs the full
matrix chain, worst layer of 169 (fp64):

| composition | max rel error |
|---|---|
| this closed form | **3.4e-16** (exact) |
| naive `Πcₖ` with the baked norms | 3.8e-01 – 7.5e-01 |
| summation (what the branch does now) | 1.06 – 4.66 |

So "compose multiplicatively" is only correct in this exact form; a bare product of the baked
`c`s is 38–75% wrong. The cross term from the factors matches the dense one to 1.8e-08.

Cost of switching: the summation anchor (bit-exact vs `merge_loras_into_base_model`) is traded
for a chaining anchor (bit-exact vs `load_and_apply_loras` / MLX / our own refit engine), the
stack becomes order-dependent, each adapter needs one more baked vector, and per-step strength
for multi-DoRA stacks has to go through `set_strengths_exact` — the cheap one-write
`set_strengths` cannot express a product. Not built; nothing currently *fails* without it.

Bookkeeping trap: read **each adapter's rank from its own tensors up front**, and give layers an
adapter does not target an explicit **zero block**. Inferring rank from "the last matching layer"
breaks the moment a stack mixes adapters with different layer sets (a 229-layer lora with a
169-layer dora), producing short buffers or `mat1 and mat2 shapes cannot be multiplied`.

### ⚠ The seconds embedder is EXCLUDED from every adapter — and why

Until 2026-10-07 the shipping 229-target engine would return an **entirely NaN** latent, with no
error raised anywhere, for many two-adapter DoRA stacks. The root cause is not rank, not the
operand range, and not the fp16 branch accumulation — it is one layer plus the summation
semantics above.

**`conditioners.seconds_total.embedder.embedding.1`** is a `[256]→[768]` linear whose output is
**global conditioning**: it feeds the adaLN modulation of all 24 blocks. Every other target is a
residual-stream projection. Then:

1. Every overfit DoRA drives `c` at that layer far below 1 — mean 0.149–0.618 (the trained one,
   trained near `c≈1`, sits at 0.973).
2. Stacking **sums**: `1 + Σ(cₖ−1)`. Two adapters each asking for ≈0.2× sum to `2c−1 ≈ −0.5…−0.7`
   — **sign-flipped**. Chained composition (`c₁·c₂ ≈ 0.04`) would have stayed positive.
3. An inverted *global* gain multiplies every token in every block, the activations run past the
   fp16 range, and the whole output comes back NaN.

A synthetic sweep forcing `1+pout` at that one layer, with no adapters anywhere else:

| forced `1+pout` | result |
|---|---|
| ≥ −0.05 | fine, max‖v‖ ≈ 5 |
| −0.20 | finite, max‖v‖ **17.2** (3×) |
| ≤ −0.50 | **all 330752 outputs NaN** |

Supporting evidence: enabling that single layer's branch reproduces the full NaN while all 228
others together are clean; zeroing `P` fixes it and zeroing `A`/`Bp` does not; the **228-target
engine was immune all along** (it never served the layer); NaN occurred at rank **20** and not at
rank 208. All operands were finite throughout (max‖A‖ 0.9, ‖Bp‖ 0.4, ‖P‖ 2.1) — this is a
semantics bug that fp16 converts into NaN, where fp32 would have been merely wrong.

Sign flips on the *other* 228 targets are harmless: measured across 43 real stacks, up to **100%
of a layer's rows** flipped and every render stayed finite.

**Fix (2026-10-07).** `merge.EXCLUDED_LAYERS` strips that layer from every adapter at load, in
`merge.load_stack()`, so branch engine, merge/refit engine and the reference merge all agree. The
adapter still loads; a `RuntimeWarning` names the layer once per adapter. Measured over the same
43 stacks (7 singles, all 21 pairs, 15 triples):

| | before | after |
|---|---|---|
| all-NaN renders | **20 / 43** | **0 / 43** |

Cost of the exclusion: 0.6–17.75% of one adapter's effect (`verify_medium.py`'s `L169` column).
Adapters should not train this layer in the first place — it encodes duration, not style.

### Guards

Two, because they catch different things and neither subsumes the other:

* `BranchLora.check_operands_finite()` — one device sync per **fold** (`set_stack`,
  `set_stack_gpu`, `set_strengths_exact`), raising with the offending layer named. Catches an
  operand that overflowed fp16 during the fold, e.g. an extreme strength on a drifted DoRA. It
  would **not** have caught the defect above: those operands were all finite.
* `sa3_trt_core.assert_finite_latent()` — one sync per **render**, on the final latent. This is
  the backstop that turns a NaN anywhere inside the 8 steps into an error instead of a file that
  plays silence. It has to be wired into **three** places, not one:
  `sample_flow_pingpong`, `GraphPingpongSampler.sample`, and `FullPipelineGraph.run`
  (`sa3_trt.py`). ⚠ **The mega-graph is the one that matters and the one that is easy to miss**:
  it captures DiT + decode + the device-to-host copy in a single graph, so the latent is never a
  tensor the samplers see, and it is the CLI/gradio default. Guarding only the samplers looks
  right and changes nothing — verified by rendering with `SA3_LORA_NO_EXCLUDE=1`, which wrote a
  12-second WAV of **literal zeros**, exit 0. A NaN latent decodes to int16 zero, so the symptom
  is *exact digital silence*, not noise. The mega-graph check reads `decoder_in_buf` after
  `stream.synchronize()`.

`SA3_LORA_NO_EXCLUDE=1` restores the pre-fix behaviour for A/B — it is how the two bullets above
were tested, and it will happily produce that silent file if the guard is ever removed.

A threshold guard on `min(1+pout)` was **considered and rejected on the data**: a single adapter
that renders perfectly already reaches −0.15 (an overfit r4) and −1.055 (an overfit
-xs r16), so no threshold separates working stacks from broken ones. The layer
identity, not the value, is what mattered.

---

## 7. Runtime integration (the part that bites)

* **LoRA bindings are PER-CONTEXT.** Any second execution context — the eager/CFG path, a
  batched CFG context — needs its own `attach_lora`. Without it TRT cannot resolve the output
  shape and the render dies on `tensor with negative dimension -1: [1, 256, -1]`, *with or
  without an adapter loaded*.
* **A branch engine will not enqueue with operands unbound**, even unused. Bind a zero stack at
  construction, **before** the first graph capture.
* **Rank change ⇒ buffers reallocate ⇒ every captured graph dies.** Drop them. A **same-rank**
  swap can keep them by copying into the *existing* buffers — but take `dict(B.bufs)`, not
  `B.bufs`: `set_stack_gpu` mutates that dict in place, so holding a reference makes the copy a
  no-op and leaves the graph pointing at freed memory. It renders **silence**, reporting success.
* **`set_input_shape` returns `False`; it does not raise.** Exceed the rank cap and TRT rejects
  every operand shape while the render proceeds on the *previous* bindings — confident, wrong
  audio. Read the cap from `get_tensor_profile_shape("lora_srow", 0)` and refuse. Put the check
  in the **runtime**, not a caller: CLI and UI reach `set_stack*` by different routes.
* **Shared scratch that re-sizes will free a buffer live CUDA graphs still address.** A graph
  replays recorded *addresses*, bypassing the context, so rebinding contexts is not enough. This
  fires when a lazily-loaded engine (e.g. the a2a encoder) grows the requirement past the initial
  allocation — the plain DiT stayed under it, the branch DiT (779 vs 748 MB) tipped it over. It
  surfaces far away, as a **T5 Myelin CUDA 700**. Drop cached graphs when the buffer moves.
* **Warn about unserved layers.** An adapter targeting something the engine does not branch loads
  "successfully" and is silently ignored — name it, once per (adapter, engine).

### Engine load: use `mmap`

```
f.read() + deserialize   2.77 s        mmap    0.68 s      (2.95 GB engine)
```
The cost is the **3 GB host allocation**, not the filesystem (RAM `/dev/shm` reads at the same
1.92 GB/s as the network FS) and not TRT (deserialize alone runs at 7.6 GB/s; `Runtime.max_threads`
changes nothing; `IStreamReaderV2` is no better). **Hold the mapping for the engine's lifetime** —
releasing it after deserialize gives `malloc_consolidate(): invalid chunk size`,
nondeterministically. Cold start 17.2 s → 12.4 s.

---

## 8. Measured performance (sa3-medium, 229 targets, H200)

| | |
|---|---|
| engine file | 2957 MB (plain 2922) |
| scratch | 799 MB (plain 748) |
| operands | 2.59 MB per rank unit — 43 MB @ r16, 333 @ r128, 1327 @ r512 |
| forward overhead | flat **~3.5–4.5 ms**: +35% @L=1292, +10.7% @L=4096, **independent of rank** to ~64 |
| no-adapter cost | you pay the full overhead with a zero stack — run the plain engine until an adapter is selected |
| cold fold, norm-baked | 319 ms |
| strength change | **5 ms**, no recapture |
| same-rank swap | 157 ms, no recapture |
| rank change | 62–288 ms fold **+ 0.6–0.9 s graph recapture on the next render** |
| duration change | 0.56–0.85 s first time per `T_lat`, 92 ms cached |

Rank-max zero-padding would remove every recapture (rank is nearly free) at ~+290 MB and
0–6% step time — evaluated, not built.

---

## 9. Verifying it

Compare against canonical `load_and_apply_loras` on identical fixed inputs. Essential controls,
each of which caught a real bug:

1. **Zero-adapter control** — engine with no adapter vs torch base. Establishes the engine's own
   fp16 floor (~1.3e-02) and catches a *baseline* mismatch masquerading as a fold error. The
   published sa3-m ONNX is built from **ARC**, not the RF base — folding against the wrong
   variant gave 27% error that read exactly like a fold bug.
2. **Fold error relative to the signal it belongs to.** Normalising a strong adapter's error by
   the *base* norm inflates it and produces false alarms.
3. **Delta metric** — `(adapted − zero)` engine vs `(adapted − base)` torch. The common fp16
   error cancels, so a weakly-trained adapter is measured on its own contribution. Essential:
   an adapter whose whole effect is near the fp16 floor cannot be validated any other way.
4. **Ablation** — zero one branch's operands and re-measure to attribute error to it.
5. **Stacks**, mixed family and mixed layer sets. A single-adapter test cannot catch a
   block-offset error.
6. **End-to-end through the real pipeline**, incl. a2a/inpaint/CFG, and an audio-level check
   that unload returns to base. ⚠ the SAME-S decoder is stochastic — establish its noise floor
   by rendering the same thing twice, or you will read AE noise as a LoRA defect.

Pin adapter checkpoints in a manifest. `ls | sort | tail -1` puts `step=9000` after
`step=15000`; use the numeric step.

---

## 10. Why branches, not refit

`refit` (merge into weights) costs **0%** forward and is bit-exact for every family including
`bora`/`dora-cols`, but needs a refit per adapter (~seconds) and cannot do per-step strength.
Branches cost ~+35% at musical lengths and buy µs-class swaps and a free strength knob.
Crossover measured on large at ~36 generations per adapter. Ship both if you can; the branch is
the interactive path.

---

## Files

```
lora/build_branch.py       build (targets core|full|all, --rank-max)
lora/build_refit.py        TARGETS / LOCAL_TARGETS / EXTRA_TARGETS / CONV_TARGETS, classify()
lora/branch_runtime.py     BranchLora: buffers, set_stack, set_strengths, guards
lora/refold_runtime.py     RefoldLora: GPU fold (~470× faster), set_stack_gpu, exact strength
lora/bake_dora.py          baked_vnorm_row + sign-aware self-check
lora/verify_medium.py      single adapters vs canonical, with all the controls above
lora/verify_stack.py       arbitrary stacks vs canonical
lora/bench_medium.py       footprint, per-L, live ops
lora/bench_rank_vram.py    VRAM/time vs loaded rank
lora/bench_engines.py      engine-to-engine comparison
pipeline/models.py         branch_engine() / engine_variant() — one resolve, no hand-pairing
```
