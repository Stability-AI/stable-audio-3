# SAME autoencoder — static "rung" TFLite models

The SAME audio encoder and decoder (SAME-S and SAME-L) ship as **static rung** `.tflite` models instead of the
old dynamic-varlen graphs. This doc explains what a rung model is, why we switched, and the decisions baked in.

## The problem: dynamic-varlen graphs blow up RAM
The encoders/decoders are transformers over an internal sequence of `L·17` tokens (L = latent frames). Exported as
a single **dynamic-shape** graph (`[1,256,-1]`), TFLite/XNNPACK can't reuse activation buffers across shapes, so
peak RAM explodes with length:
- **SAME-L** attention is O(S²) → the dense-varlen decoder hits **~16 GB at L=256** and can't run long clips at all.
- **SAME-S** is block-local O(L), but the varlen graph *still* reuses buffers poorly → RAM grows linearly to
  **~50 GB at L=8192**.

Manual chunking of the varlen graph helped RAM but added a chunk-size knob and overlap waste.

## The fix: a ladder of fixed-shape subgraphs in one file
A **rung model** is several **fixed-size** subgraphs — `{1,2,4,8,12,16,32,64,128,256}` for SAME-L,
`{2,4,8,12,16,32,64,128,256}` for SAME-S (even-only) — merged into ONE `.tflite`, sharing a single copy of the
weights (deduped by content). It's loaded via **litert ≥ 2.2.0 `CompiledModel` + the XNNPACK weight cache**
(`xnnpack_weight_cache_path`), which packs the weights once. `RungEncoder` / `RungDecoder`
(`scripts/rung_encoder.py`, `scripts/rung_decoder.py`) auto-discover the rungs from the file's `s<N>` signatures
and, for any length L, **dispatch to the rung that minimises total decoded latents**, tiling on it when L exceeds
the largest rung.

Result: RAM is **flat** — SAME-L ~2–3 GB (fp32) / ~1.5 GB (w8a8), SAME-S ~0.3–0.8 GB — from L=1 to L=8192, and it's
**faster** than the varlen graph (static shapes → a tight reused arena, exactly what varlen couldn't do). There is
**no chunk-size knob**: dispatch is automatic and optimal on any CPU (min compute = min time).

### Tiling overlap (bit-exact)
When L exceeds the largest rung, `RungDecoder`/`RungEncoder` tile with an overlap equal to the model's receptive
field, so the stitched output is **bit-exact** vs a single whole pass:
- **SAME-L: 12 latents** (12 blocks × ±1 latent-group of sliding-window attention).
- **SAME-S: 16 latents** (its 34-token / 2-latent block-local attention + midpoint shift widens the field).

### Small rungs make tiny-L exact + fastest
The small rungs `{1,2,4,8,12}` let a short clip decode **exactly L** instead of padding up to rung-16 — lossless and
fastest (e.g. SAME-S L=2: 15 ms vs 20 ms whole vs 59 ms pad-to-16; SAME-L L=1: 75 ms vs 380 ms). They cost only
+2 MB (SAME-S) / +5 MB (SAME-L) since the weights dedup. This removes the last case where whole-decode beat rungs,
so rungs are strictly best at **every** length.

## Two precision tiers: fp32 + w8a8
The codec runs **once on a fixed latent** (no sampler chaos), so int8 is quality-free: on a real-music round-trip
(audio → encode → decode → recon vs original), **w8a8 ≈ fp32** — the int8 weight error sits ~27 dB below the AE's
own ~10–14 dB loss. So the codec ships just two tiers:
- **`w8a8` (default)** — int8 weights; ~3× faster and ~half the RAM on an int8-capable CPU, quality-free.
- **`fp32`** — bit-exact reference, and the pick for CPUs *without* int8 acceleration.

`w16a32` and `w8a32` are dropped (they sit between w8a8 and fp32 in fidelity, so they add nothing audible while
being bigger/slower); the pre-rung varlen models — including those — live under `tflite/same-*/legacy/`.
(The **DiT** keeps its full precision ladder — int8 *does* fail there, because the 8-step sampler is chaotically
sensitive and int8 becomes a different sample.)

### `--max-rung N`
The one hardware knob: cap the largest rung used (e.g. `--max-rung 64`) to roughly halve peak codec RAM on a
tiny-memory device, at some speed cost. Default is auto (uncapped = fastest); no realistic target needs it.

## CPU support
`w8a8` runs (correctly) everywhere and is *faster* than fp32 wherever the CPU has an int8 dot/matrix instruction:
x86 AVX-512-VNNI / AVX-VNNI / AMX-INT8; ARM NEON-dotprod (v8.2+) / i8mm (v8.6+) / SVE2 / SME. That includes the
**Raspberry Pi 5** (Cortex-A76, has `asimddp`) — where the rungs are also what make SAME-L *fit* (the old varlen's
16 GB@256 would OOM it). On a CPU with *no* int8 dot (very old x86, ARMv8.0 like the Pi 4), w8a8 still runs but
won't beat fp32 — use `fp32` there. Needs a 64-bit OS with `ai_edge_litert>=2.2.0`.

## How the models are built
The encoder torch sources were lost in a July 2026 cleanup, so they were re-derived from the checkpoints; the full
from-scratch build (checkpoints → the 8 canonical `.tflite`) lives in [`../build/`](../build/) — see its README.
Key encoder detail (vs the decoder): the summary token sits at the **end** of each 17-group, the FF has **no** sin
gate, and the bottleneck is `(z·scaling_factor + bias) / running_std`.

## Validation
- Rung fp32 is **bit-exact** vs the shipped tflite (cos = 1.0), and tiling is bit-exact at the overlaps above.
- Round-trip loss vs ground-truth audio is **identical to the official Stability-AI stable-audio-3 AE** (SAME-L
  ~10.5–10.9 dB, SAME-S ~14.0–14.5 dB) — i.e. the loss is the autoencoder's own ceiling, not the runtime; w8a8
  matches fp32.
- Ear-checked: shipped-dense / rung-fp32 / rung-w8a8 are indistinguishable.
