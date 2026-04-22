"""
Simple LoRA fine-tuning for Stable Audio 3.

Dataset layout:
  data_dir/
    clip1.wav   (or .flac, .mp3, .ogg)
    clip1.txt   ← text prompt for clip1
    clip2.wav
    clip2.txt
    ...

Saves .safetensors LoRA checkpoints compatible with the inference pipeline and run_gradio.py.

Usage:
  python train_lora.py --model medium-rf --data_dir ./my_data --output_dir ./lora_out
  python train_lora.py --model medium-rf --data_dir ./my_data --steps 500 --rank 8
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from stable_audio_3.data.dataset import (
    LatentDatasetConfig,
    LocalDatasetConfig,
    PreEncodedDataset,
    SampleDataset,
    collation_fn,
)
from stable_audio_3.loading_utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_3.model import create_diffusion_cond_from_config
from stable_audio_3.model_configs import rf_models
from stable_audio_3.models.lora import (
    LoRAParametrization,
    add_lora,
    cast_base_to_precision,
    get_lora_params,
    get_lora_state_dict,
    prepare_dora_state_dict,
    resolve_adapter_type,
    save_lora_safetensors,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(model_name: str, device: torch.device):
    if model_name not in rf_models:
        raise ValueError(
            f"LoRA training only supports RF models. Got '{model_name}', valid: {list(rf_models)}"
        )
    model_cfg = rf_models[model_name]
    with open(model_cfg.config_path) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(model_cfg.ckpt_path))
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return model, model_config


def apply_lora(
    model,
    rank: int,
    alpha: float,
    adapter_type: str = "dora",
    dropout: float = 0.0,
    include=None,
    exclude=None,
    svd_bases=None,
):
    adapter_type = resolve_adapter_type(adapter_type)
    lora_cfg = {
        torch.nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=alpha,
                lora_dropout_p=dropout,
                adapter_type=adapter_type,
            ),
        },
        torch.nn.Conv1d: {
            "weight": partial(
                LoRAParametrization.from_conv1d,
                rank=rank,
                lora_alpha=alpha,
                lora_dropout_p=dropout,
                adapter_type=adapter_type,
            ),
        },
    }
    add_lora(
        model.model, lora_cfg, include=include, exclude=exclude, svd_bases=svd_bases
    )
    add_lora(
        model.conditioner,
        lora_cfg,
        include=include,
        exclude=exclude,
        svd_bases=svd_bases,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def caption_metadata_fn(info, audio):
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = load_model(args.model, device)
    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio

    # Align to downsampling ratio
    sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank

    svd_bases = None
    if args.svd_bases_path is not None:
        print(f"Loading SVD bases from {args.svd_bases_path}")
        svd_bases = torch.load(
            args.svd_bases_path, map_location="cpu", weights_only=True
        )

    apply_lora(
        model,
        rank=args.rank,
        alpha=lora_alpha,
        adapter_type=args.adapter_type,
        dropout=args.dropout,
        include=args.include,
        exclude=args.exclude,
        svd_bases=svd_bases,
    )

    if args.lora_checkpoint is not None:
        print(f"Loading LoRA checkpoint from {args.lora_checkpoint}")
        from stable_audio_3.models.lora import load_lora_checkpoint

        lora_sd = load_lora_checkpoint(args.lora_checkpoint)
        prepare_dora_state_dict(lora_sd)
        model.model.load_state_dict(lora_sd, strict=False)
        model.conditioner.load_state_dict(lora_sd, strict=False)

    lora_params = list(get_lora_params(model.model)) + list(
        get_lora_params(model.conditioner)
    )
    # LoRA params train in fp32; base model stays in bf16
    for p in lora_params:
        p.data = p.data.float()

    if args.base_precision:
        cast_base_to_precision(model.model, args.base_precision)
        cast_base_to_precision(model.conditioner, args.base_precision)
        if model.pretransform is not None:
            model.pretransform.to(
                torch.bfloat16
                if args.base_precision in ("bf16", "bfloat16")
                else torch.float16
            )
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    optimizer = AdamW(lora_params, lr=args.lr)

    if args.encoded_dir:
        dataset = PreEncodedDataset(
            [LatentDatasetConfig(id="train", path=args.encoded_dir)],
            latent_crop_length=sample_size // ds_ratio,
            random_crop=True,
        )
    else:
        dataset = SampleDataset(
            [
                LocalDatasetConfig(
                    id="train",
                    path=args.data_dir,
                    custom_metadata_fn=caption_metadata_fn,
                )
            ],
            sample_size=sample_size,
            sample_rate=sample_rate,
            force_channels="stereo",
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=True,
        collate_fn=collation_fn,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.model.train()

    step = 0
    while step < args.steps:
        for audio_batch, metadata in loader:
            if step >= args.steps:
                break

            if args.encoded_dir:
                latents = audio_batch.to(device=device, dtype=torch.bfloat16)
            else:
                audio_batch = audio_batch.to(device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    latents = model.pretransform.encode(audio_batch)
            conditioning = model.conditioner(list(metadata), str(device))

            # rf_denoiser noise schedule: xt = (1-t)*x0 + t*noise, target = noise - x0
            B = latents.shape[0]
            t = torch.rand(B, device=device, dtype=torch.bfloat16)
            noise = torch.randn_like(latents)
            t_bc = t[:, None, None]
            noised = latents * (1 - t_bc) + noise * t_bc
            target = noise - latents

            # Inpaint model requires mask conditioning; all-zeros = pure generation
            conditioning["inpaint_mask"] = [
                torch.zeros(B, 1, latents.shape[2], device=device, dtype=torch.bfloat16)
            ]
            conditioning["inpaint_masked_input"] = [torch.zeros_like(latents)]

            model.model.train()
            pred = model(noised, t, cond=conditioning, cfg_dropout_prob=0.1)
            loss = F.mse_loss(pred.float(), target.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                print(f"Step {step}/{args.steps}  loss={loss.item():.4f}")
            if step % args.save_every == 0:
                save_checkpoint(model, args, step, lora_alpha)
    if step % args.save_every != 0:
        save_checkpoint(model, args, step, lora_alpha)
    print("Done.")


def save_checkpoint(model, args, step, lora_alpha):
    state_dict = {
        **get_lora_state_dict(model.model),
        **get_lora_state_dict(model.conditioner),
    }
    lora_config = {
        "rank": args.rank,
        "alpha": lora_alpha,
        "adapter_type": args.adapter_type,
        "include": args.include,
        "exclude": args.exclude,
    }
    out = Path(args.output_dir) / f"lora_step{step}.safetensors"
    save_lora_safetensors(state_dict, lora_config, out)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Simple LoRA fine-tuning for Stable Audio 3"
    )
    p.add_argument("--model", choices=list(rf_models), default="medium-rf")
    p.add_argument(
        "--data_dir",
        required=True,
        help="Folder with audio files and matching .txt captions",
    )
    p.add_argument(
        "--encoded_dir",
        default=None,
        help="Pre-encoded latent directory from pre_encode_dataset.py (skips per-batch encode)",
    )
    p.add_argument("--output_dir", default="lora_out")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling factor (default: same as rank)",
    )
    p.add_argument(
        "--adapter_type", choices=["lora", "dora", "lora-xs"], default="dora"
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied to LoRA inputs",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only apply LoRA to modules whose name contains one of these substrings",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Skip modules whose name contains one of these substrings",
    )
    p.add_argument(
        "--svd_bases_path",
        default=None,
        help="Path to pre-computed SVD bases (.pt) for -XS adapter types",
    )
    p.add_argument(
        "--base_precision",
        choices=["bf16", "bfloat16", "fp16", "float16"],
        default=None,
        help="Cast frozen base weights to lower precision (LoRA params stay fp32)",
    )
    p.add_argument(
        "--lora_checkpoint",
        default=None,
        help="Path to an existing LoRA .safetensors checkpoint to resume from",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Maximum clip duration in seconds (default 30)",
    )
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    train(p.parse_args())


if __name__ == "__main__":
    main()
