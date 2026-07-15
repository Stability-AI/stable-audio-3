# Underfit training conventions → MLX port reference

Working reference for the `mlx-lora-training` branch: every training convention and
default in [underfit](https://github.com/dada-bots/underfit) (the SA3 LoRA trainer),
inventoried from its source (2026-07-14), plus the gap analysis against the MLX
primitives this branch starts from (PR #51 by @betweentwomidnights). The goal:
the MLX runtime can run as an underfit backend — same behaviors, same defaults,
same checkpoint format. Trim/fold this file into README before the PR finalizes.

## 1. Adapter configuration

| Convention | Underfit value | Source |
|---|---|---|
| adapter_type (config-layer default) | `"lora"`; legacy `"dora"` → `"dora-rows"` | training/lora.py:34, resolve_adapter_type |
| adapter_type (product/dashboard default) | **`dora-rows`** | dashboard index.html:1336 |
| rank (config-layer default) | 8 | training/lora.py:32 |
| rank (product/dashboard default) | **16** | README.md:157 |
| alpha default | = rank | training/lora.py:33 |
| targets | every `nn.Linear` + `nn.Conv1d` under BOTH `model.model` (DiT) and `model.conditioner`, filtered by include/exclude | training/lora.py:53-64 |
| include/exclude | substring match + `[i-j]` numeric-range expansion; dashboard default EXCLUDES all conditioner + "other" modules (embedders/projections/convs) | dashboard index.html:5741-49 |
| adapter dropout | none | — |
| trainable dtype | LoRA params forced fp32; base stays fp16 (`base_precision`) | loop.py:344-347 |

## 2. Checkpoint format

- Keys: `{**get_lora_state_dict(model.model), **get_lora_state_dict(model.conditioner)}`
  → DiT keys come out `model.transformer.layers...` (inner `.model` on the DiT wrapper),
  conditioner keys `conditioners.seconds_total.embedder.embedding.1...`; per-key shape
  `<layer>.parametrizations.weight.<lora_index>.{lora_A,lora_B,magnitude,magnitude_r,magnitude_c,M_xs,U,V}`
  (index 0 for the finetune adapter).
- Tensors saved **fp16** (`save_lora_safetensors` casts `.half()`); DoRA magnitudes
  squeezed to 1-D on load (`prepare_dora_state_dict`).
- Metadata `lora_config` JSON: `rank, alpha, adapter_type, include, exclude` +
  injected `step` (int), `epoch` (int), `base_model` (str, from model_config).
- Filename: `{run_label}-step={step}-epoch={epoch}.safetensors` (run_label = run name
  with trailing 14-digit timestamp stripped); dir `<save_dir>/<name>/<uuid8>/checkpoints`.
- -xs: only `M_xs` (+ magnitudes) trained/saved; U/V bases precomputed OFFLINE
  (compute_svd.py) from fp32 base weights into a separate `.pt`
  (`{key: {"U": fp16, "V": fp16, "S": fp32, "shape"}}`, sign-canonicalized by max-abs
  U-column entry), loaded via `svd_bases_path` from the model registry; per-layer SVD
  at attach time is the fallback.
- Validation contract (lora_validate.py): `.safetensors` only, ≥1 parametrization key,
  `lora_config` present with `rank` + `adapter_type` (alpha:=rank, include/exclude:=[]
  back-filled), adapter type inferable from key fingerprints.

## 3. Loss (loop.py:561-613 + loss.py)

- Objective from the model: `rectified_flow` → `alphas, sigmas = 1−t, t`,
  `noised = x·α + noise·σ`, `target = noise − x` (`"v"` objective also exists:
  cos/sin schedule, `target = noise·α − x·σ`).
- MSE per-element; `loss_normalization` default `"none"` (timestep/sample/sample_channel
  variance-normalization modes exist, eps 1e-6, detached variance).
- Masking: `mask_loss_weight` default 0.0 and `mask_padding_attention` default False →
  **signal-only** loss; reduction is per-sample mean over valid positions×channels,
  then batch mean. Inpainting further restricts the mask to the regenerated region.
- No sigma/logSNR loss weighting. Runs under fp16 autocast; no explicit casts.

## 4. Timesteps

- Sampler default **`uniform`** (`torch.rand`); also logit_normal,
  trunc_logit_normal (left 0.075, flipped `1−t`), log_snr (mean −1.2, std 2.0),
  log_snr_uniform (min −6, max 5).
- Distribution shift comes from the MODEL config: SA3 models ship
  `{"type": "full", "min_length": 256, "max_length": 4096}` (base_shift 0.5,
  max_shift 1.15 defaults) — applied as `t = dist_shift.shift(t, seq_len)` with
  seq_len = the latent time dim (crop length) by default
  (`use_effective_length_for_schedule` False; when True: `ceil(seconds_total·sr/4096)`).

## 5. CFG dropout (dit.py:441-450)

- `cfg_dropout_prob` default **0.1**, applied INSIDE the model forward when
  `cfg_scale == 1.0`: per-sample bernoulli mask replaces the whole
  `cross_attn_cond` (prompt+seconds concat) with **zeros**; `global_cond` is NOT
  dropped. (Matches inference: the uncond branch zeroes cross_attn, keeps global.)

## 6. Optimizer / schedule / loop mechanics

| Convention | Underfit value |
|---|---|
| optimizer | AdamW, single param group (LoRA params only) |
| lr | **no default — must be supplied** (`--lr` or optimizer_configs) |
| betas/eps/weight_decay | torch defaults: (0.9, 0.999) / 1e-8 / 0.0 |
| LR schedule | none by default (InverseLR + any torch scheduler configurable) |
| grad clip | off by default (`gradient_clip_val = 0.0`); when set: clip_grad_norm_ |
| grad accumulation | none (defaults.ini key exists but inert) |
| EMA / validation / early stopping / save_top_k | none (inert keys) |
| precision | `"16-mixed"`: fp16 autocast + GradScaler; base fp16, LoRA fp32 |
| batch_size | 1 (defaults.ini) |
| seed | 42 |
| steps | step-driven; `max_steps` absolute global-step target (default ~1e10) |
| checkpoint cadence | every 1000 steps + final save; save BEFORE demos |
| resume offsets | explicit config > safetensors metadata `step/epoch` > filename `step=/epoch=` tokens > step//steps_per_epoch |
| logging | train/loss, train/lr, grad_norm + lora_magnitude (fp32, post-clip); `loss_by_timestep.bin` binary telemetry (step, t_mean, loss_mean) flushed every 10 steps |

## 7. Data pipeline

- **Pre-encode** (offline, once): whole files (no clip splitting), up to 600 s
  (aligned down to ds_ratio=4096 samples), 44.1 kHz stereo (mono→stereo by channel
  repeat, resample via torchaudio defaults, NO loudness/peak norm), encoder fp32,
  `chunked=True` for >30 s. Saved per file: `<stem>.npy` latents `[D,T]` fp32
  **unscaled** (training divides by pretransform scale) + `<stem>.json` with
  `seconds_total` (full duration, rounded 3dp), `seconds_start: 0`,
  `audio_samples`, `latent_shape`, latent-resolution 0/1 `padding_mask`, and all tags.
- **Training dataset** (PreEncodedDataset semantics): crop/pad every item to
  `latent_crop_length` — **1300** latents (sm-music/sfx, ≈120 s) / **4096** (medium,
  ≈380 s); `random_crop=True` (dashboard default) choosing a start within the valid
  region; short items padded with `silence.npy` latent if present else zeros, mask
  extended with zeros. Fixed-length batches + padding mask (no bucketing).
  **`seconds_total` stays the FULL song duration after cropping** (not recomputed).
- **Small datasets**: if `len(ds) < batch_size`, RandomSampler with replacement,
  `num_samples = batch_size*100` (~100 steps/epoch) — the core underfit workflow.
- **Prompts** (prompt_templates.py, per-sample at train time): tag-based by default
  (`use_tags`, tags→`Title:/Artist:/Album:/Genre:/Label:/Year:/Composer:/BPM:/Prompt:`
  joined ", "); augmentation: 50% shuffle-all vs 50% random-subset (1..n tags);
  optional trigger token prepended with prob **80%**; path/fixed prompt sources with
  weighted balance (tags 50/paths 50/fixed 0); `lyrics=""` always. Tag priority:
  JSON sidecar > `.txt` sidecar (whole content = `prompt`) > embedded ID3
  (`title,artist,album,genre,label,date,composer,bpm`).
- **Demos**: `demo_every` 1000 (template; loop default 0=off) + baseline at step 0;
  `demo_steps` 50 (ARC 8), cfg [7] (ARC 1), per-entry seed/duration/lora_strength/
  `lora_interval_max` (σ-interval demo path uses an explicit Euler loop);
  peak-normalized int16 → mp3 (`ffmpeg -q:a 0`), trimmed to seconds_total+5 s,
  `demo_<i>_<step:08d>.mp3` + json sidecar, idempotent per step.

## 8. Gap analysis: PR #51 primitives vs the above

Already aligned: adapter math for all 9 types (torch-parity tested), rf loss target
+ masked per-sample-then-mean reduction, the 5 timestep samplers + full/flux/logsnr
shifts with matching defaults, fp16 checkpoint tensors + `lora_config` metadata,
`.safetensors`-only trust boundary, fp32 adapter params over a fp16 base, SVD sign
canonicalization, `[i-j]` include/exclude expansion, whole-file chunked encoding
with validity masks.

To build (roughly in order):
1. **lora.py**: save with underfit key roots (`model.` + `conditioners.`) instead of
   bare DiT keys; adapter for the baked seconds-conditioner Linear
   (`cond.seconds_total_weight` ↔ `conditioners.seconds_total.embedder.embedding.1`);
   match include/exclude against checkpoint-convention names so dashboard filter
   strings work verbatim; config-layer defaults rank 8 / alpha=rank / lora;
   metadata `step/epoch/base_model` injection helpers; accept `svd_bases_path`
   `.pt`-equivalent (or keep recompute fallback, which underfit also has).
2. **conditioning + dropout**: training-time conditioning builder (T5Gemma frozen,
   prompt padding, seconds token; MLX runtime pieces exist) + per-sample
   cross_attn→zeros dropout at prob 0.1 (keep global_cond).
3. **trainer loop** (`lora_train`-equivalent CLI): AdamW (torch-default betas/eps/wd,
   lr required), no schedule, no clip, batch 1, seed 42, step-driven with
   checkpoint-every-1000 + final save, underfit filename convention, resume-offset
   ladder, loss/lr/grad_norm logging, `loss_by_timestep.bin`.
4. **dataset**: PreEncodedDataset-equivalent (npy+json pairs, latent_crop_length
   1300/4096 defaults, random_crop, silence/zero pad, mask handling, seconds_total
   NOT recomputed, oversample-with-replacement ×100 for tiny sets) +
   prompt_templates port (50/50 shuffle/subset, trigger 80%, display-name map).
5. **pre-encode CLI** on top of `audio_encoding.py` (600 s cap, fp32, unscaled fp32
   npy + json sidecar with the exact metadata fields).
6. **demos** (optional, last): pipeline-based demo step with underfit cadence/format.

## 9. Training-forward conventions discovered by parity testing (2026-07-14)

**Harness** (reusable, 2026-07-15): `scripts/parity_forward_torch.py` (sa3 venv —
builds the fp32 base model, injects shared latents/noise/timesteps, saves
prediction/loss/cross/global/scale) + `scripts/parity_forward_mlx.py` (MLX venv —
same forward, reports three bug-localizing PSNRs). Fresh run (sm-music base, B=3,
t={0.25,0.5,0.75}): **pretransform.scale=1.0 asserted against the real model**
(confirms the softnorm no-scale convention — a prime hidden-bug candidate),
DiT-only forward (torch cross/global injected → isolates the DiT) **79.0 dB**,
conditioning cross/global **88.5/77.6 dB**, end-to-end **79.0 dB**, loss torch
4.716201 vs MLX 4.716814 (**Δ 6.1e-4**). **Backward also checked**: ∂loss/∂noised
through the full DiT (torch cross/global injected) agrees at **77.5 dB** — the same
floor as the forward, so the backward sweep every adapter gradient is built from
matches too (the local adapter-param VJP was separately verified within-framework to
≤2.4e-7 MLX / ≤4.4e-7 torch during the DoRA reformulation). A convention/autograd bug
(RoPE/attn-mask/adaLN/local_add_cond layout/scale) reads ≪40 dB; ~78 dB is the
cross-framework fp32 reduction-order floor through 20 layers. Confirms fwd+bwd parity,
no bug.

Verified by a controlled forward (identical latents crop, numpy-seeded noise,
t=0.5, fixed prompt, fp32) through underfit's torch loop on MPS vs the MLX
trainer — final agreement **loss 3.9829 vs 3.9823, prediction PSNR 80.7 dB**
(bounded by the fp16 npz weights; torch MPS-vs-CPU agrees at 3e-6 rel):

- **Train on the BASE checkpoint** (`stabilityai/stable-audio-3-*-base`,
  rectified_flow), NOT the shipped ARC (rf_denoiser) weights that inference
  uses. `lora_train_mlx.py --dit-weights` + a 441-key npz conversion
  (strip `model.model.`, `.gamma`→`.weight`, conv (out,in,k)→(out,k,in),
  `to_local_embed.{0,2}`→`.seq.{0,2}`, conditioner keys → `cond.*`).
- **local_add_cond during training = all-ONES inpaint mask + zero context**
  (the models are the diffusion_cond_inpaint variant). The inference path's
  `local_add_cond=None` (≡ zeros) is an inference-only convention — using it
  in training changes the loss by ~4x (wrong conditioning regime).
- **Cross-attention runs over all 257 tokens at training time too**: the
  torch conditioner substitutes the learned padding_embedding into padded
  positions (exactly like the optimized runtimes) and the attention mask is
  effectively pass-through. Do NOT mask/slice to valid tokens.

## 10. MPS vs MLX benchmark (M4 Pro 48 GB, sm-music base, 30 steps,
## dora-rows r16 α16 = 141 layers / 9.2M params, batch 1, crop 256, fp16 base)

| | MPS (underfit loop, torch 2.13) | MLX (lora_train_mlx.py) |
|---|---|---|
| steady it/s | 0.668 (1498 ms/step) | **1.65 (2.47x faster)** |
| peak memory | 3.67 GiB driver-allocated | (not instrumented; comparable class) |
| loss regime | 1–43, spikes at low t | same (0.9–35, spikes at low t) |
| checkpoint | underfit-valid, 423 keys | underfit-valid, identical key set |
