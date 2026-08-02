# Lessons learned — building the cpu-amx pipeline

Hard-won findings from porting SA3 to torch-free C++/AMX on CPU. Read this before touching the engine
`.cpp`s or the quantization — most of these cost real time to discover, and several are non-obvious
enough that they nearly produced *wrong conclusions*.

## AMX is the whole game (and it doesn't transfer)

- **The bf16 speedups are AMX-BF16-specific.** The bf16 decoder engines are fast because Sapphire/Emerald
  Rapids has an AMX-BF16 matrix unit. On a non-AMX CPU (older Xeon, M-series, Raspberry Pi) those same
  `.so`s are *not* faster than fp32 — the win doesn't transfer. Don't quote the bf16 numbers as "CPU"
  numbers; they're "AMX CPU" numbers.
- **int8's win scales with GEMM width AND the ISA's int8 strength.** AMX-INT8 (`tdpbssd`) is huge; VNNI
  is solid; **AVX2 has no int8 nanokernel at all** in triton-cpu — int8 there is a wash or worse. So
  "int8 is faster" is only true on VNNI/AMX. Narrow models (SAME-S, dim 768) get a smaller int8 win than
  wide ones (SAME-L/DiT, dim 1536) because the GEMM has to be wide enough to amortize the requant.
- **The wall is memory/Q-DQ traffic, not FLOPs.** The single biggest structural win was making the int8
  path *fully fused* — requant folded into the GEMM epilogue, activations staying int8 between GEMMs, no
  per-GEMM quant/dequant round-trips. Building the engine properly (arena allocation, fused epilogues,
  a linear band-scan instead of an O(T²) dense mask) dissolved a whole set of "problems" (chunking,
  O(T²), memory-bound) that looked fundamental when they were really artifacts of a naive implementation.

## fp32 islands and attention numerics

- **Differential attention (the decoders) needs fp32 islands.** SAME-S/SAME-L attention computes
  `softmax(Q1K1ᵀ)V − λ·softmax(Q2K2ᵀ)V` — subtracting two similar-magnitude softmax outputs. In int8 (or
  even bf16 in the wrong place) this catastrophically cancels; int8 attention collapses the SAME-S decode
  by ~12 dB. Keep the softmax + the norms in fp32. This is the "mandatory fp32 attention island."
- **Standard softmax attention (T5Gemma) does NOT.** T5Gemma is ordinary softmax attention (+ a logit
  softcap), not differential — so the QK and PV matmuls tolerate bf16-AMX, and the fp32 softcap+softmax
  island in the middle absorbs the rounding. This is why the T5Gemma engine runs attention in bf16 (28 ms)
  instead of fp32 (141 ms, which would be *slower* than the TFLite it replaces). One caveat: a rare token
  with razor-peaked attention + very large output norm can bf16-flip an argmax key (cos 0.913 on that one
  token) — it's precision, not a bug (two independent fp32/fp16 references disagree by the same amount).
- **RMSNorm must accumulate in fp32/double** and apply Gemma's `(1 + w)` scale (weights are centered on 0
  — forgetting the `1+` looks like a totally broken model, ~0 dB). Softcap is `C·tanh(x/C)`, applied to
  the *scaled* logits before the mask add.

## The bf16-RoPE long-sequence bug (cost 2-minute renders)

The shipped bf16 medium-DiT **clipped every render ≥ ~2 minutes**, and the cause was subtle: **bf16 RoPE
angle overflow**. At L≈4092 the RoPE position angle reaches ~4155 radians; in bf16 (8-bit mantissa) that
has no fractional precision left, so ~16 of 32 frequency pairs are destroyed. It was *not* differential-
attention cancellation (bf16 arithmetic everywhere else is clean). **Keep RoPE in fp32.** Corollary: any
"vs bf16" long-sequence baseline taken before this fix was measured against a broken baseline.

## int8 quantization: what actually helps

- **The decoders are activation-limited, not weight-limited.** Once activations are int8, the weight
  quantization method barely matters — GPTQ *alone* buys +0.2–0.3 dB. The gain comes from **SmoothQuant**
  (migrating per-in-channel activation outliers into the weights), which does the heavy lifting (+2.2 dB
  on SAME-S). GPTQ's real value is that it lets you push SmoothQuant's α higher (0.5→0.9) without the
  rescaled weights degrading. Never SmoothQuant `proj_out` — its ±214 outliers backfire.
- **The SQ+GPTQ decoder gain is real and transfers to real music — for SAME-S.** +1.5 dB on calibration
  latents, and **+1.2 dB on real-music latents across every genre**. For SAME-L the gain is negligible
  (+0.04 dB on real music) because its 12-block residual stream already averages out per-layer int8 noise.
  We ship "improved" for both anyway (drop-in, never worse), but the quality win is all on SAME-S.
- **GPTQ HURTS the DiT — don't quantize-improve it.** On the medium DiT, GPTQ *lowers* per-forward cosine
  (0.994 → 0.985) and it's robust to every mitigation (damping, cross-attn exclusion, SmoothQuant). Cause:
  the DiT's self-attn qkv is adaLN-modulated (activation stats shift per timestep + per condition) and the
  8-step sampler is chaos-sensitive, so a single trajectory-averaged Hessian overfits the calibration and
  generalizes *worse* than calibration-agnostic RTN. The DiT stays naive-int8; its real speed/quality lever
  is bf16/fp8-on-GPU, never CPU weight-PTQ.
- **int4 is unshippable on all four models** — every lever (naive, GPTQ, Hadamard/QuaRot rotation) lands
  below the usable floor. These weights aren't outlier-dominated, so there's nothing for rotation to spread.

## Real music reframes the whole quality question

- **On real music the autoencoder is the bottleneck, not the decoder precision.** Encoding real songs with
  the fp32 AE and decoding loses a lot on its own (mean ~24 dB vs the original; dense/transient material
  like distorted rock or percussion drops to corr 0.77–0.91). Against that, fp32/bf16/int8/improved land
  within ~0.1 dB of each other vs the original. So on real content **int8 is essentially free** — the AE
  ceiling dwarfs the quant delta.
- **dB is a heuristic; the ear is the arbiter.** Audio-PSNR is harsh — a 24 dB AE reconstruction or a
  "low-dB" int8 decode can be perceptually transparent (benign spectral error, masked noise). Every
  quant/precision decision was settled by rendering comparable clips and listening (A/B listening rooms
  with the same fixed latent through each config), never by a dB gate.

## C++ `.so` gotchas that will bite you

- **The fused int8 `.so` keeps weights in GLOBAL statics.** `<engine>_init(weights_base)` loads into a
  process-global `std::map` — so instantiating two engines with different weights **in one process**
  silently makes them share the last-loaded weights (a `ctypes.CDLL` of the same path is one `dlopen`).
  This produced bit-identical "current" and "improved" clips and nearly shipped a wrong "the improvement
  doesn't transfer" conclusion. **Rule: one weight-set per process** (or `dlopen` separate `.so` copies).
- **The DiT `.so` has a multithread heap race + a teardown double-free.** At high thread counts it can
  heap-corrupt mid-sample (glibc "double free / corrupted size"); and even at threads=1 it double-frees at
  process exit. Workarounds: default the DiT to threads=1 for reliability, and call **`os._exit(0)`** after
  writing output to skip the crashing teardown. (The decoder + T5Gemma engines are bit-deterministic and
  race-free — this is specific to the DiT `.so`.)
- **The decoder torch *port* is not the true AE.** `same_l_decoder_torch` (what the C++ engines match at
  ~62 dB) differs from the real stable-audio-tools AE decoder by ~38 dB (corr 0.998 — likely transparent,
  but real). Every decoder "vs fp32" number in the campaign is vs the port, not the true autoencoder.

## Methodology that paid off

- **Numpy-reference-first.** For every new engine, write a pure-numpy fp32 forward and validate it against
  the groundtruth (should hit ~90+ dB) BEFORE writing any C++. That's where you cheaply catch the
  weight-transpose, RMSNorm-`(1+w)`, RoPE-layout, softcap, and embed-scale bugs — not in a compile loop.
- **One source, CPU + GPU.** The Triton-CPU reimplementation of the DiT runs the same kernels the GPU path
  can use; a single source targeting both is worth more than two hand-tuned ones.
- **Verify independently, don't trust the "done."** Re-running validation with a *different* harness (and
  a skeptical eye) caught the global-static-weights bug and the port-vs-AE gap. "It passed" from the thing
  that built it is not the same as "it's correct."
- **Group name has a space.** On this filesystem the group is "Domain Users" (two words), which shifts
  `ls -la | awk '{print $5}'` by one column — every size read that way is wrong. Use `stat -c%s`.
