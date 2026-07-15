"""SA3 LoRA training on Apple Silicon — pure MLX, underfit conventions.

The MLX counterpart of underfit's raw-PyTorch training loop
(github.com/dada-bots/underfit, underfit/training/loop.py), built on the
trainable adapters in models/defs/lora.py. Every training convention and
default mirrors underfit (see ../TRAINING_CONVENTIONS.md):

  - rectified-flow velocity target (noise − x), signal-only masked MSE,
    per-sample-then-mean reduction
  - uniform timestep sampler + the SA3 models' "full" distribution shift
    (min_length 256, max_length 4096)
  - CFG dropout 0.1: per-sample, the whole cross-attention conditioning
    (prompt + seconds token) is zeroed; global_cond is kept
  - AdamW, single param group, torch defaults (betas 0.9/0.999, eps 1e-8,
    weight decay 0.0), no schedule, no clipping; LR has NO default and must
    be supplied (--lr)
  - adapters train in fp32 over a frozen fp16 base
  - step-driven loop; checkpoints every --checkpoint-every (1000) steps and
    at the end, saved BEFORE anything else can fail, named
    {run_label}-step={step}-epoch={epoch}.safetensors with lora_config
    metadata {rank, alpha, adapter_type, include, exclude, step, epoch,
    base_model}
  - resume offsets: explicit flags > checkpoint metadata > filename tokens
  - per-step loss_by_timestep.bin telemetry (struct "Iff": step, t_mean,
    loss_mean), flushed every 10 steps

Dataset: pre-encoded latents from scripts/pre_encode_mlx.py (npy + json
pairs), cropped to --latent-crop-length with random crop, tiny datasets
oversampled with replacement (batch_size*100 draws/epoch) — the underfit
overfitting workflow. Prompts are built per-sample with underfit's tag
augmentation. seconds_total stays the FULL source duration after cropping
(deliberate underfit convention).

Performance:
  - the train step (loss + grad + optimizer update) is mx.compile'd by
    default (MLX docs "Compile > Training graphs" pattern; --no-compile for
    the eager path). Batch shapes are constant, so it compiles once.
  - T5Gemma prompt conditioning is cached per exact prompt string (per-run,
    in-memory; --no-t5-cache to disable) — the tiny-dataset workflow repeats
    a handful of prompts, so the hit rate is ≈100% after the first epoch.
  - --grad-checkpoint recomputes each transformer block's activations in the
    backward pass (mlx-lm style mx.checkpoint) — memory for big crops on
    small Macs, at the cost of roughly one extra forward.
  - wired memory limit is raised to the device's recommended working-set
    size at startup (mx.set_wired_limit).

Usage:
    python scripts/lora_train_mlx.py --dit sm-music \
        --latents-dir output/latents/my-set --lr 1e-4 \
        --name my-lora --save-dir output/runs --max-steps 2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from sa3_mlx import DIT_CHOICES, load_dit  # noqa: E402
from weights import ensure_local  # noqa: E402
from sa3_mlx import T5GEMMA_NPZ_REL  # noqa: E402
from models.defs.sa3_pipeline import (  # noqa: E402
    apply_prompt_padding, load_conditioner_from_npz,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from models.defs.training import (  # noqa: E402
    rectified_flow_loss, sample_training_timesteps, shift_training_timesteps,
)
from models.defs.lora import (  # noqa: E402
    TrainableSecondsEmbedder, inject_from_lora_config, iter_trainable_lora_layers,
    load_lora_checkpoint, load_trainable_lora_state, save_lora_checkpoint,
    underfit_lora_config,
)
from models.defs import demo_mlx as demo  # noqa: E402
from models.defs.latent_dataset import (  # noqa: E402
    PreEncodedLatentDataset, iterate_batches,
)

# underfit registry conventions per model
BASE_MODEL_NAMES = {"sm-music": "sa3-sm-music", "sm-sfx": "sa3-sm-sfx",
                    "medium": "sa3-medium"}
LATENT_CROP_DEFAULTS = {"sm-music": 1300, "sm-sfx": 1300, "medium": 4096}
DIST_SHIFT_DEFAULT = {"shift_type": "full",
                      "options": {"min_length": 256, "max_length": 4096}}
T5_CACHE_CAP = 512  # prompt-conditioning cache entries (oldest evicted first)


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.

    Verbatim port of mlx-lm's mlx_lm/tuner/trainer.py:grad_checkpoint —
    activations of every instance of the layer's class are recomputed during
    the backward pass instead of being kept alive, trading ~one extra forward
    for a much smaller peak working set.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


class TrainBundle(nn.Module):
    """DiT + (optional) trainable seconds conditioner under one module root so
    a single value_and_grad covers every adapter parameter. The bundle's
    trainable_parameters() are the LoRA params only (bases frozen at inject)."""

    def __init__(self, dit, secs):
        super().__init__()
        self.dit = dit
        self.secs = secs  # TrainableSecondsEmbedder or None


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", required=True, choices=list(DIT_CHOICES.keys()))
    ap.add_argument("--dit-weights", default=None,
                    help="Override the DiT weight file — REQUIRED in practice for "
                         "training parity with underfit: train on the BASE "
                         "(rectified-flow) checkpoint, not the shipped ARC "
                         "(rf_denoiser) weights that inference uses. Convert the "
                         "HF *-base model.safetensors to npz first (441-key "
                         "layout; see TRAINING_CONVENTIONS.md).")
    ap.add_argument("--latents-dir", required=True,
                    help="Root of pre-encoded npy+json pairs (scripts/pre_encode_mlx.py)")
    ap.add_argument("--lr", type=float, required=True,
                    help="Learning rate (underfit has NO default — must be supplied)")
    ap.add_argument("--name", default="underfit-run", help="Run name (defaults.ini)")
    ap.add_argument("--save-dir", default=str(REPO / "output" / "runs"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=10_000_000_000,
                    help="Absolute global-step target (resume-aware)")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    # adapter config — underfit dashboard defaults
    ap.add_argument("--adapter-type", default="dora-rows")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None, help="Default: = rank")
    ap.add_argument("--include", default=None,
                    help="Comma-separated substring filters (underfit convention)")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated; default = underfit's dashboard exclude list")
    ap.add_argument("--no-conditioner-lora", action="store_true",
                    help="Skip the seconds-conditioner adapter (underfit adapts it "
                         "by default — e.g. the plini checkpoint carries its delta)")
    ap.add_argument("--bora-mode", default="speed", choices=["speed", "memory"],
                    help="bora/bora-xs forward: 'speed' (default) caches W0² per "
                         "adapted layer (+1 weight copy) for the reformulated "
                         "no-materialize forward; 'memory' keeps the original "
                         "full-weight forward with no cache. Ignored for other "
                         "adapter types.")
    # data
    ap.add_argument("--latent-crop-length", type=int, default=None,
                    help="Default per model: 1300 (sm-*) / 4096 (medium)")
    ap.add_argument("--random-crop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prompt-config", default=None,
                    help="JSON file with underfit prompt_config (trigger, tag options…)")
    # schedule / conditioning
    ap.add_argument("--timestep-sampler", default="uniform",
                    choices=["uniform", "logit_normal", "trunc_logit_normal",
                             "log_snr", "log_snr_uniform"])
    ap.add_argument("--dist-shift", default="full",
                    choices=["none", "full", "flux", "logsnr"])
    ap.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    # performance
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="mx.compile the train step (loss+grad+optimizer update; "
                         "MLX 'Compile > Training graphs' pattern). "
                         "--no-compile restores the eager path.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Gradient-checkpoint every transformer block (mlx-lm "
                         "style mx.checkpoint): activations are recomputed in "
                         "the backward pass — much lower peak memory for big "
                         "crops, ~one extra forward of compute")
    ap.add_argument("--t5-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="Cache the padded T5Gemma prompt conditioning per exact "
                         f"prompt string (per-run, in-memory, max {T5_CACHE_CAP} "
                         "entries)")
    # resume
    ap.add_argument("--lora-ckpt-path", default=None)
    ap.add_argument("--step-offset", type=int, default=None)
    ap.add_argument("--epoch-offset", type=int, default=None)
    # demos (underfit convention: RF Euler inference on the base model + trained
    # LoRA, decode → mp3, baseline at step 0 then every --demo-every steps)
    ap.add_argument("--demo-every", type=int, default=0,
                    help="Generate demos every N steps (0 = off). A baseline "
                         "set is also rendered at step 0.")
    ap.add_argument("--demo-config", default=None,
                    help="JSON list of demo entries. Each: {prompt, cfg=7, "
                         "seed=0, steps=50, lora_strength?, lora_interval_max?}.")
    ap.add_argument("--demo-decoder", default=None,
                    choices=["same-s", "same-l"],
                    help="Decoder for demos (default: the DiT's default_decoder).")
    ap.add_argument("--demo-dir", default=None,
                    help="Where to write demo_<i>_<step>.mp3 (default: cwd, "
                         "which the dashboard sets to <run>/demos/).")
    return ap.parse_args()


def resolve_offsets(args, ckpt_config) -> tuple[int, int]:
    """Underfit's resume ladder: explicit flags > checkpoint metadata >
    filename step=/epoch= tokens > zero."""
    if args.step_offset is not None or args.epoch_offset is not None:
        return int(args.step_offset or 0), int(args.epoch_offset or 0)
    if ckpt_config:
        step = ckpt_config.get("step")
        epoch = ckpt_config.get("epoch")
        if step is not None:
            return int(step), int(epoch or 0)
    if args.lora_ckpt_path:
        name = Path(args.lora_ckpt_path).name
        ms = re.search(r"step=(\d+)", name)
        me = re.search(r"epoch=(\d+)", name)
        if ms:
            return int(ms.group(1)), int(me.group(1)) if me else 0
    return 0, 0


def build_conditioning(t5, padding_emb, prompts, max_len=256,
                       cache=None, cache_stats=None):
    """Frozen prompt-side conditioning: T5Gemma embeddings with the learned
    padding embedding applied. Returns [B, 256, 768] fp32 (the trainable
    seconds token is concatenated later, inside the grad scope).

    When ``cache`` (a dict) is given, the padded embedding is cached per
    EXACT prompt string. The cache is per-run only (in-memory, never
    persisted) and stores evaluated [1, max_len, 768] fp32 mx arrays; it is
    capped at T5_CACHE_CAP entries, evicting oldest-inserted first. The
    tiny-dataset underfit workflow repeats a handful of prompts, so after
    the first epoch the T5Gemma forward is skipped entirely (~100% hits)."""
    prompts = list(prompts)
    if cache is None:
        embeds, mask = t5.encode(prompts, max_len=max_len)
        mx.eval(embeds, mask)
        padded = apply_prompt_padding(embeds.astype(mx.float32), mask,
                                      padding_emb.astype(mx.float32))
        return mx.stop_gradient(padded)

    missing = [p for p in dict.fromkeys(prompts) if p not in cache]
    if cache_stats is not None:
        n_miss = sum(1 for p in prompts if p in set(missing))
        cache_stats["misses"] += n_miss
        cache_stats["hits"] += len(prompts) - n_miss
    if missing:
        embeds, mask = t5.encode(missing, max_len=max_len)
        padded = apply_prompt_padding(embeds.astype(mx.float32), mask,
                                      padding_emb.astype(mx.float32))
        mx.eval(padded)
        for i, p in enumerate(missing):
            while len(cache) >= T5_CACHE_CAP:
                cache.pop(next(iter(cache)))  # evict oldest
            row = padded[i:i + 1]
            mx.eval(row)
            cache[p] = row
        print(f"  t5-cache: encoded {len(missing)} new prompt(s) "
              f"({len(cache)} cached)")
    return mx.stop_gradient(mx.concatenate([cache[p] for p in prompts],
                                           axis=0))


def main():
    args = parse_args()
    np_rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    # Let Metal wire (pin) up to the recommended working set — avoids paging
    # stalls when the training working set approaches physical memory.
    if hasattr(mx, "set_wired_limit") and hasattr(mx, "device_info"):
        wired = mx.device_info().get("max_recommended_working_set_size")
        if wired:
            mx.set_wired_limit(int(wired))

    alpha = float(args.alpha) if args.alpha is not None else float(args.rank)
    lora_config = underfit_lora_config()
    lora_config.update({"adapter_type": args.adapter_type, "rank": args.rank,
                        "alpha": alpha})
    if args.include is not None:
        lora_config["include"] = [s.strip() for s in args.include.split(",") if s.strip()]
    if args.exclude is not None:
        lora_config["exclude"] = [s.strip() for s in args.exclude.split(",") if s.strip()]

    crop_len = args.latent_crop_length or LATENT_CROP_DEFAULTS[args.dit]
    base_model = BASE_MODEL_NAMES[args.dit]

    # ── run dirs (underfit layout: <save>/<name>/<uuid8>/checkpoints) ────────
    run_label = re.sub(r"-\d{14}$", "", args.name) if args.name else None
    session = uuid.uuid4().hex[:8]
    ckpt_dir = Path(args.save_dir) / args.name / session / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {args.name}  session {session}")
    print(f"checkpoints → {ckpt_dir}")

    # ── dataset ──────────────────────────────────────────────────────────────
    prompt_config = None
    if args.prompt_config:
        prompt_config = json.loads(Path(args.prompt_config).read_text())
    dataset = PreEncodedLatentDataset(
        args.latents_dir, crop_len, random_crop=args.random_crop,
        prompt_config=prompt_config, seed=args.seed)
    print(f"dataset: {len(dataset)} latent file(s), crop {crop_len} latents"
          + ("  (oversampling with replacement — tiny dataset)"
             if len(dataset) < args.batch_size else ""))

    # ── models ───────────────────────────────────────────────────────────────
    t0 = time.time()
    if args.dit_weights:
        import importlib
        mod = importlib.import_module(DIT_CHOICES[args.dit]["loader"])
        dit_model = mod.load_dit(args.dit_weights, T_lat=crop_len,
                                 dtype=mx.float16, compile_=False)
        cond_src = args.dit_weights
    else:
        print("WARNING: training on the shipped (ARC/rf_denoiser) weights — "
              "underfit convention is to train on the BASE checkpoint "
              "(pass --dit-weights)")
        dit_model, _ = load_dit(args.dit, T_lat=crop_len, dtype=mx.float16)
        cond_src = str(ensure_local(DIT_CHOICES[args.dit]["ckpt"]))
    print(f"DiT loaded ({time.time()-t0:.1f}s, fp16 base, T_lat={crop_len})")
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))  # frozen
    padding_emb, secs_embedder = load_conditioner_from_npz(cond_src, prefix="cond.")

    # ── inject adapters ─────────────────────────────────────────────────────
    report, saved_config = inject_from_lora_config(
        dit_model, lora_config, checkpoint_prefix="model.",
        bora_mode=args.bora_mode)
    print(f"lora: {report.layer_count} DiT layer(s), {report.adapter_type}, "
          f"rank {args.rank}, alpha {alpha:g} "
          f"({report.trainable_parameters/1e6:.2f}M trainable)")

    if args.grad_checkpoint:
        # Patch the TransformerBlock class (both DiT defs expose the blocks at
        # dit.transformer.layers) so every block recomputes in backward.
        grad_checkpoint(dit_model.transformer.layers[0])
        print(f"grad-checkpoint: on "
              f"({len(dit_model.transformer.layers)} transformer blocks)")

    secs_module = None
    if not args.no_conditioner_lora:
        secs_module = TrainableSecondsEmbedder(secs_embedder.W, secs_embedder.b)
        try:
            cond_report, _ = inject_from_lora_config(
                secs_module, lora_config,
                checkpoint_prefix="conditioners.seconds_total.",
                bora_mode=args.bora_mode)
            print(f"lora: conditioner seconds embedder adapted "
                  f"({cond_report.layer_count} layer)")
        except ValueError:
            secs_module = None  # filtered out by include/exclude
    bundle = TrainBundle(dit_model, secs_module)

    # ── resume ───────────────────────────────────────────────────────────────
    step_offset = epoch_offset = 0
    if args.lora_ckpt_path:
        sd, ckpt_cfg = load_lora_checkpoint(args.lora_ckpt_path)
        restored = load_trainable_lora_state(bundle, sd)
        step_offset, epoch_offset = resolve_offsets(args, ckpt_cfg)
        print(f"resumed {restored} adapter layer(s) from "
              f"{Path(args.lora_ckpt_path).name} (step {step_offset}, "
              f"epoch {epoch_offset})")
    else:
        step_offset, epoch_offset = resolve_offsets(args, None)

    max_new_steps = max(0, args.max_steps - step_offset)

    # ── optimizer: AdamW, torch defaults, single group, no schedule ──────────
    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.999],
                            eps=1e-8, weight_decay=0.0)

    # ── training-time local conditioning (underfit/upstream convention) ──────
    # The SA3 models are the diffusion_cond_inpaint variant: the trainer feeds
    # pure generation as an all-ONES inpaint mask + zero context (verified at
    # 80.7 dB forward parity vs the torch loop; the inference path's
    # local_add_cond=None ≡ zeros is an inference-only convention and trains
    # against the wrong conditioning regime — 4x loss difference).
    lac_const = mx.array(
        np.concatenate([np.ones((1, 1, crop_len), np.float32),
                        np.zeros((1, 256, crop_len), np.float32)],
                       axis=1).transpose(0, 2, 1)).astype(mx.float16)

    # ── loss (adapters in the bundle are the only trainable params) ──────────
    # `seconds_in` is the raw seconds batch [B] fp32 when the trainable
    # seconds conditioner is active; with --no-conditioner-lora it is the
    # PRE-computed frozen seconds token [B, 1, 768] (built outside the step so
    # the compiled step stays pure mx — no numpy inside the traced graph).
    def loss_fn(bundle, latents, timesteps, loss_mask, prompt_cond,
                seconds_in, drop_mask):
        if bundle.secs is not None:
            sec_tok = bundle.secs(seconds_in)                 # [B, 1, 768] fp32
        else:
            sec_tok = seconds_in                              # frozen, precomputed
        cross = mx.concatenate([prompt_cond, sec_tok.astype(mx.float32)], axis=1)
        # CFG dropout (dit.py:441): whole cross_attn → zeros per sample;
        # global_cond (the seconds token) is kept.
        cross = mx.where(drop_mask[:, None, None], mx.zeros_like(cross), cross)
        global_cond = sec_tok[:, 0, :]
        cross16 = cross.astype(mx.float16)
        global16 = global_cond.astype(mx.float16)

        def model_fn(noised, t):
            return bundle.dit(noised, t.astype(noised.dtype), cross16, global16,
                              local_add_cond=lac_const)

        return rectified_flow_loss(model_fn, latents, timesteps,
                                   loss_mask=loss_mask)

    value_and_grad = nn.value_and_grad(bundle, loss_fn)

    # ── train step (compiled by default — MLX "Compile > Training graphs") ──
    # Captured state: model params (the LoRA adapters the optimizer mutates),
    # optimizer moments, and the global PRNG key — rectified_flow_loss draws
    # its noise INSIDE the step. Everything np/python (timestep sampling, CFG
    # dropout draw, T5 encode, telemetry) stays OUTSIDE the step; batch
    # shapes/dtypes are constant (fixed crop + batch), so this traces once.
    # optimizer.init() materializes the moment arrays up front so the captured
    # state structure never changes (no recompile on step 2).
    optimizer.init(bundle.trainable_parameters())
    state = [bundle.state, optimizer.state, mx.random.state]

    def train_step(latents, timesteps, loss_mask, prompt_cond, seconds_in,
                   drop):
        loss, grads = value_and_grad(bundle, latents, timesteps, loss_mask,
                                     prompt_cond, seconds_in, drop)
        optimizer.update(bundle, grads)
        return loss

    if args.compile:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)

    # ── telemetry (underfit's loss_by_timestep.bin, struct "Iff") ────────────
    tele = open("loss_by_timestep.bin", "ab")

    def checkpoint(step, epoch):
        fname = (f"{run_label}-step={step}-epoch={epoch}.safetensors"
                 if run_label else f"step={step}-epoch={epoch}.safetensors")
        path = ckpt_dir / fname
        save_lora_checkpoint(
            bundle, path,
            include=lora_config.get("include"),
            exclude=lora_config.get("exclude"),
            extra_config={"step": int(step), "epoch": int(epoch),
                          "base_model": base_model})
        print(f"\n  ✓ checkpoint {path.name}")
        return path

    # ── train loop (step-driven; epoch = one dataloader pass) ────────────────
    raw_step = 0
    epoch = epoch_offset
    last_saved_step = None
    t5_cache = {} if args.t5_cache else None
    t5_stats = {"hits": 0, "misses": 0}

    # ── demos (underfit RF path) ──────────────────────────────────────────────
    # RF Euler inference on the base model + trained LoRA, decoded to mp3. Baseline
    # at the start, then every --demo-every steps + a final render. Idempotent.
    demo_entries = None
    if args.demo_every > 0:
        if args.demo_config:
            demo_entries = json.loads(Path(args.demo_config).read_text())
            print(f"demos: {len(demo_entries)} prompt(s) every {args.demo_every} "
                  f"step(s) → {args.demo_dir or '.'}")
        else:
            print("WARNING: --demo-every set but no --demo-config — demos disabled")
    demo_dir = args.demo_dir or "."
    demo_decoder_name = args.demo_decoder or DIT_CHOICES[args.dit]["default_decoder"]
    _demo_dec = {}  # lazy: (decoder, chunk_fn, chunk_cfg)

    def run_demos(step):
        if not demo_entries:
            return
        os.makedirs(demo_dir, exist_ok=True)
        pending = [(i, e) for i, e in enumerate(demo_entries)
                   if not os.path.exists(
                       os.path.join(demo_dir, f"demo_{i}_{step:08d}.mp3"))]
        if not pending:
            return
        if "dec" not in _demo_dec:
            from sa3_mlx import load_decoder
            t_d = time.time()
            _demo_dec["dec"] = load_decoder(demo_decoder_name, dtype=mx.float32)
            print(f"  demo decoder {demo_decoder_name} loaded ({time.time()-t_d:.1f}s)")
        decoder, chunk_fn, chunk_cfg = _demo_dec["dec"]
        adapters = list(iter_trainable_lora_layers(bundle))
        T_lat = crop_len
        seconds_val = T_lat * demo.SAMPLES_PER_LATENT / demo.SAMPLE_RATE
        for i, entry in pending:
            prompt = entry.get("prompt", "")
            cfg = float(entry.get("cfg", 7))
            seed = int(entry.get("seed", 0))
            steps = int(entry.get("steps", 50))
            strength = entry.get("lora_strength")
            interval = entry.get("lora_interval_max")
            prompt_cond = build_conditioning(t5, padding_emb, [prompt],
                                             cache=t5_cache, cache_stats=t5_stats)
            if bundle.secs is not None:
                sec_tok = bundle.secs(mx.array([seconds_val], dtype=mx.float32))
            else:
                sec_tok = secs_embedder([seconds_val])
            sec_tok = mx.stop_gradient(sec_tok.astype(mx.float32))
            cross = mx.concatenate([prompt_cond, sec_tok], axis=1).astype(mx.float16)
            gcond = sec_tok[:, 0, :].astype(mx.float16)
            null = mx.zeros_like(cross) if cfg != 1.0 else None
            model_fn = demo.make_rf_model_fn(bundle.dit, cross, gcond, cfg=cfg,
                                             null_cross_attn=null)
            sigmas = demo.build_sigmas(steps)
            noise = mx.random.normal((1, 256, T_lat), dtype=mx.float16,
                                     key=mx.random.key(seed))
            snap = before = None
            if interval is not None:
                snap = demo.snapshot_adapters(adapters)
                s_on = float(strength) if strength is not None else 1.0

                def before(_i, sigma, s_on=s_on, interval=interval, snap=snap):
                    demo.scale_adapters(snap, s_on if sigma <= interval else 0.0)
            elif strength is not None and float(strength) != 1.0:
                snap = demo.snapshot_adapters(adapters)
                demo.scale_adapters(snap, float(strength))
            try:
                latent = demo.rf_euler_sample(model_fn, noise, sigmas,
                                              before_step=before)
                audio = demo.decode_latents(decoder, chunk_fn, chunk_cfg, latent, T_lat)
                meta = {"prompt": prompt, "cfg": cfg, "seed": seed,
                        "steps": steps, "step": int(step)}
                if strength is not None:
                    meta["lora_strength"] = strength
                if interval is not None:
                    meta["lora_interval_max"] = interval
                demo.save_demo_mp3(audio, i, step, demo.SAMPLE_RATE, demo_dir, meta)
                print(f"  ♪ demo {i} @ step {step}: {prompt[:48]!r} "
                      f"→ demo_{i}_{step:08d}.mp3", flush=True)
            except Exception as e:
                print(f"  demo {i} failed: {e}", flush=True)
            finally:
                if snap is not None:
                    demo.scale_adapters(snap, 1.0)  # restore trained factors
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()  # measure the training loop, not weight loading
    t_start = time.time()
    print(f"training: max {max_new_steps} new step(s) "
          f"(global target {args.max_steps}), lr {args.lr:g}, "
          f"batch {args.batch_size}, cfg-dropout {args.cfg_dropout_prob}, "
          f"compile {'on' if args.compile else 'off'}")
    run_demos(step_offset)  # baseline (untrained / resumed state)
    try:
        while raw_step < max_new_steps:
            for batch in iterate_batches(dataset, args.batch_size,
                                         shuffle=True,
                                         seed=args.seed + epoch):
                if raw_step >= max_new_steps:
                    break
                global_step = raw_step + step_offset + 1

                latents = mx.array(batch["latents"]).astype(mx.float16)
                loss_mask = mx.array(batch["padding_mask"])
                seconds = mx.array(np.asarray(batch["seconds_total"],
                                              dtype=np.float32))

                # timesteps: sampler + model-config distribution shift
                t_np = sample_training_timesteps(
                    args.timestep_sampler, latents.shape[0], rng=np_rng)
                if args.dist_shift != "none":
                    t_np = shift_training_timesteps(
                        t_np, latents.shape[-1], shift_type=args.dist_shift,
                        options=DIST_SHIFT_DEFAULT["options"]
                        if args.dist_shift == "full" else None)
                timesteps = mx.array(t_np)

                prompt_cond = build_conditioning(t5, padding_emb,
                                                 batch["prompt"],
                                                 cache=t5_cache,
                                                 cache_stats=t5_stats)
                drop = mx.array((np_rng.random(latents.shape[0])
                                 < args.cfg_dropout_prob))
                if bundle.secs is not None:
                    seconds_in = seconds
                else:
                    # frozen conditioner: build the seconds token OUTSIDE the
                    # (possibly compiled) step — see loss_fn
                    seconds_in = mx.stop_gradient(secs_embedder(
                        [float(s) for s in batch["seconds_total"]]))

                t_step = time.time()
                loss = train_step(latents, timesteps, loss_mask, prompt_cond,
                                  seconds_in, drop)
                mx.eval(state, loss)
                step_ms = (time.time() - t_step) * 1000.0

                raw_step += 1
                loss_v = float(loss)
                tele.write(struct.pack("Iff", global_step,
                                       float(t_np.mean()), loss_v))
                if raw_step % 10 == 0:
                    tele.flush()
                rate = raw_step / max(time.time() - t_start, 1e-9)
                print(f"step {global_step}  train/loss {loss_v:.6f}  "
                      f"train/lr {args.lr:.3e}  epoch {epoch}  "
                      f"({rate:.2f} it/s, {step_ms:.0f} ms)", flush=True)

                if args.checkpoint_every > 0 and \
                        global_step % args.checkpoint_every == 0:
                    checkpoint(global_step, epoch)  # save BEFORE demos
                    last_saved_step = global_step
                if args.demo_every > 0 and global_step % args.demo_every == 0:
                    run_demos(global_step)
            epoch += 1
    except KeyboardInterrupt:
        print("\ninterrupted — saving final checkpoint")

    global_step = raw_step + step_offset
    if raw_step > 0 and global_step != last_saved_step:
        checkpoint(global_step, epoch)
    if raw_step > 0:
        run_demos(global_step)  # final render
    tele.close()
    if t5_cache is not None:
        print(f"t5 cache: {t5_stats['hits']} hit(s) / "
              f"{t5_stats['misses']} miss(es), {len(t5_cache)} cached")
    print(f"peak memory: {mx.get_peak_memory()/2**30:.2f} GB"
          + ("  (grad-checkpoint on)" if args.grad_checkpoint else ""))
    print(f"done: {raw_step} step(s) in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
