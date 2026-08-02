#!/usr/bin/env python3
"""Capture FP8 calibration data for the SA3 DiT (producer-side).

``build_dit_fp8.py`` needs a calibration ``.npz``: real DiT inputs sampled
across the pingpong sampling schedule. This script produces one from nothing
but the model checkpoint, by driving the model's own ``generate()`` and
recording the six DiT engine inputs at every sampling step.

FP8 is the only quantization path in this repo that needs calibration data
(``fp16`` / ``fp32`` need none, MLX is weight-only), so this also
establishes the convention: a ``<name>.calib.npz`` whose keys are the six ONNX
input names, each a leading-axis batch of samples. The npz is a reproducible
producer artifact: gitignored, never committed, regenerated on demand.

Native by construction:
  * Loads with ``loading_utils.load_diffusion_cond`` (the repo's own loader:
    ``create_diffusion_cond_from_config`` + ``copy_state_dict``), wrapped in a
    ``StableAudioModel``, so it needs only the checkpoint dir
    (``model_config.json`` + ``model.safetensors`` + the bundled
    ``t5gemma-b-b-ul2/`` encoder, loaded locally; falls back to the config's HF
    model_name if absent). No ``StableAudioModel.from_pretrained`` (its
    hard-coded checkpoint registry), no ``stable-audio-tools``.
  * Prompts come from the repo's own ``interface/reprompt.py``: the Music
    few-shot examples (``SYSTEM_PROMPTS["Music"]``), pulled at runtime via the
    module's ``_extract_examples`` so there is one source of truth, not a
    copy. These are the exact post-reprompt format the model is driven with at
    deployment (genre + instruments + rhythm + mood + BPM + Length).
  * Captures from the real ``StableAudioModel.generate()`` with
    ``sampler_type="pingpong"``, ``steps=8``, ``cfg_scale=1.0`` (the settings
    the model is sampled with): the conditioner, the dist-shifted sigma
    schedule, and the pingpong renoise are all the real inference path, not a
    re-implementation.
  * Hooks the ``DiTWrapper`` forward, where conditioning arrives as the
    257-token ``cross_attn_cond`` (``[t5(256) | seconds(1)]``) and the
    257-channel ``local_add_cond`` (``[inpaint_masked_input(256) |
    inpaint_mask(1)]``, all zeros for plain text-to-music). The trailing
    seconds token is dropped to ``t5_hidden`` (256); ``seconds_total`` is the
    raw duration scalar (the engine recomputes the seconds embedding itself).

The six keys and the verified mapping:
  x              (N, 256, L)    pre-conv latent at each step
  t              (N,)           sigma at each step
  t5_hidden      (N, 256, 768)  raw T5Gemma hidden states (cross-attn token 257 dropped)
  t5_mask        (N, 256)       T5 attention mask
  seconds_total  (N,)           raw duration scalar
  local_add_cond (N, 257, L)    inpaint local conditioning (zeros for t2m)

Why these prompts, 1 seed, one duration:
  Calibration is amax (``calibration_method="max"``), so the question is "have
  we seen the tail that sets each tensor's scale", not "have we covered the
  distribution". The quantized prompt-driven GEMMs (``to_cond_embed`` and the
  cross-attn projections; attention BMMs are excluded by ``disable_mha_qdq``)
  have activation maxima dominated by T5Gemma's systematic outlier channels,
  which fire for essentially every prompt, so amax saturates after a handful of
  prompts. The reprompt example set (47 prompts, a few hundred samples
  over the 8 sigmas) sits comfortably inside the standard PTQ calibration band
  (128-512 samples), past the knee. Broader or out-of-distribution prompts can
  make calibration *worse*: under max calibration a single OOD outlier inflates
  the scale and coarsens fp8 resolution for the in-distribution inputs the
  engine actually serves, so the deployment-matched in-repo set is the right
  default. A second seed re-rolls noise the outlier channels already maxed, so
  seeds are the least useful axis. Duration is irrelevant to fp8 scales (the
  seconds-embedding front-end is inside the re-applied FP32 islands, never
  quantized), and a single duration keeps the npz rectangular for the build
  contract.

Usage:
    python make_calib.py \
        --model-config <ckpt>/model_config.json \
        --checkpoint   <ckpt>/model.safetensors \
        --out          sa3-m.calib.npz
        [--duration 54.0] [--steps 8] [--seed 1528] [--device cuda]

Requires (producer): torch, numpy, and this repo's ``stable_audio_3`` package
with the model checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

T5_TOKENS = 256
T5_HIDDEN_DIM = 768
IO_CHANNELS = 256
LOCAL_ADD_CHANNELS = 257  # [inpaint_masked_input(256) | inpaint_mask(1)]

# Default duration: with the default 6 s duration_padding_sec this adapts to a
# latent length L=646, a representative shape inside the published sa3-m profile's
# dynamic range (min 1, opt 1292, max 4096). Calibration scales are per-tensor,
# so the capture length need not equal the profile's opt point.
DEFAULT_DURATION_S = 54.0
DEFAULT_STEPS = 8
DEFAULT_SEED = 1528


def _load_prompts() -> list:
    """The Music few-shot examples from the repo's own ``reprompt.py`` (the
    exact post-reprompt format the model is driven with at deployment), via the
    module's own ``_extract_examples`` parser. Imported lazily so ``--help``
    doesn't pull in transformers (which the model load needs anyway)."""
    from stable_audio_3.interface.reprompt import (
        SYSTEM_PROMPTS, _extract_examples)
    prompts = _extract_examples(SYSTEM_PROMPTS["Music"])
    if not prompts:
        raise RuntimeError(
            "no Music examples found in reprompt.SYSTEM_PROMPTS['Music']")
    return prompts


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def _load_model(model_config_path: Path, checkpoint_path: Path, device: str):
    """Load via the repo's own ``load_diffusion_cond`` (config dict + safetensors)
    and wrap in ``StableAudioModel`` so ``generate()`` is the real entry point.
    fp32 (model_half=False) for a clean, deterministic capture; amax from fp32 is
    a safe upper bound for the fp16/fp8 engine."""
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    with open(model_config_path) as f:
        model_config = json.load(f)

    # Point the T5Gemma conditioner at the encoder bundled in the checkpoint dir
    # (snapshot_download lays it out as <ckpt>/<subfolder>/), so capture needs
    # only the checkpoint and runs offline. Drop repo_id so a stale network repo
    # can never win. Falls back to the config's HF model_name if not bundled.
    dest = model_config_path.parent
    for c in model_config.get("model", {}).get("conditioning", {}).get("configs", []):
        if c.get("type") == "t5gemma":
            cc = c.setdefault("config", {})
            subfolder = cc.get("subfolder", "t5gemma-b-b-ul2")
            if (dest / subfolder).is_dir():
                cc["model_path"] = str(dest)
                cc.pop("repo_id", None)

    print(f"  loading model: {checkpoint_path}", flush=True)
    model = load_diffusion_cond(
        model_config, str(checkpoint_path), device=device, model_half=False)
    return StableAudioModel(model, model_config, device, model_half=False)


def capture(sa3, prompts, duration, steps, base_seed):
    """Hook the DiTWrapper forward and run the pingpong generate() once per
    prompt. Each generate() emits exactly ``steps`` DiT calls, one per sigma, so
    sigma coverage is balanced without per-sigma capping."""
    rows = {"x": [], "t": [], "t5_hidden": [], "t5_mask": [], "local_add_cond": []}
    dit = sa3.dit  # DiTWrapper
    orig_forward = dit.forward

    def hook(x, t, **kw):
        ca = kw["cross_attn_cond"]
        mask = kw["cross_attn_mask"]
        if ca.shape[1] == T5_TOKENS + 1:        # drop the trailing seconds token
            ca = ca[:, :T5_TOKENS]
            mask = mask[:, :T5_TOKENS]
        lac = kw.get("local_add_cond", None)
        for i in range(x.shape[0]):             # batch is 1 at cfg_scale=1.0
            rows["x"].append(x[i].float().cpu().numpy())
            rows["t"].append(np.array([float(t[i])], dtype=np.float32))
            rows["t5_hidden"].append(ca[i].float().cpu().numpy())
            rows["t5_mask"].append(mask[i].float().cpu().numpy())
            if lac is not None:
                rows["local_add_cond"].append(lac[i].float().cpu().numpy())
            else:
                rows["local_add_cond"].append(
                    np.zeros((LOCAL_ADD_CHANNELS, x.shape[-1]), dtype=np.float32))
        return orig_forward(x, t, **kw)

    try:
        dit.forward = hook
        for p_idx, prompt in enumerate(prompts):
            sa3.generate(
                prompt=prompt, duration=duration, steps=steps,
                cfg_scale=1.0, sampler_type="pingpong",
                seed=base_seed + p_idx,
                return_latents=True,            # skip decode; we only need DiT inputs
            )
            print(f"  [{p_idx + 1}/{len(prompts)}] {len(rows['x'])} samples",
                  flush=True)
    finally:
        dit.forward = orig_forward

    return rows


def _save(rows, duration, out_path: Path):
    n = len(rows["x"])
    if n == 0:
        raise RuntimeError("captured 0 samples")
    lengths = {a.shape[-1] for a in rows["x"]}
    if len(lengths) != 1:
        raise RuntimeError(
            f"captured ragged latent lengths {sorted(lengths)} - the npz must be "
            f"rectangular; use a single --duration")
    L = lengths.pop()

    out = {
        "x":              np.stack(rows["x"]).astype(np.float32),
        "t":              np.concatenate(rows["t"]).astype(np.float32),
        "t5_hidden":      np.stack(rows["t5_hidden"]).astype(np.float32),
        "t5_mask":        np.stack(rows["t5_mask"]).astype(np.float32),
        "seconds_total":  np.full(n, float(duration), dtype=np.float32),
        "local_add_cond": np.stack(rows["local_add_cond"]).astype(np.float32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    sched = sorted({round(float(t), 6) for t in out["t"]}, reverse=True)
    print(f"\n  wrote {out_path}: {n} samples, L={L}")
    print(f"  sigma schedule ({len(sched)}): {[round(s, 4) for s in sched]}")
    lo, hi = float(out['t5_hidden'].min()), float(out['t5_hidden'].max())
    print(f"  ranges: t5_hidden [{lo:.2f}, {hi:.2f}]  "
          f"x [{float(out['x'].min()):.2f}, {float(out['x'].max()):.2f}]  "
          f"local_add_cond absmax {float(np.abs(out['local_add_cond']).max()):.3g}")
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True,
                    help="Path to model_config.json")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to model.safetensors")
    ap.add_argument("--out", default="sa3-m.calib.npz",
                    help="Output calibration npz")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                    help="Generation duration in seconds (one value: keeps the "
                         "npz rectangular; default adapts to L=646)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help="Pingpong steps (= sigmas captured per generate)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Base seed (a per-prompt offset is derived)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = _load_prompts()
    n_expected = len(prompts) * args.steps
    print(f"capturing {len(prompts)} prompts x {args.steps} sigmas = "
          f"{n_expected} samples @ {args.duration}s")

    sa3 = _load_model(Path(args.model_config), Path(args.checkpoint), args.device)
    rows = capture(sa3, prompts, args.duration, args.steps, args.seed)
    _save(rows, args.duration, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
