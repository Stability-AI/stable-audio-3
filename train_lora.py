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
  python train_lora.py --model small-rf --data_dir ./my_data --output_dir ./lora_out
  python train_lora.py --model medium-rf --data_dir ./my_data --steps 500 --rank 8
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.optim import AdamW

from stable_audio_3.inference.audio_utils import prepare_audio
from stable_audio_3.loading_utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_3.model import create_diffusion_cond_from_config
from stable_audio_3.models.minlora import (
    LoRAParametrization,
    add_lora,
    get_lora_params,
    get_lora_state_dict,
    save_lora_safetensors,
)

CONFIGS = {
    "medium-rf": ("SA3-M-RF.json", "SA3-M-RF.ckpt"),
}

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(model_name: str, device: torch.device):
    config_path, ckpt_path = CONFIGS[model_name]
    with open(config_path) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    return model, model_config


def apply_lora(model, rank: int, alpha: float, adapter_type: str = "dora"):
    """Inject LoRA into the DiT and conditioner (matches inference loading convention)."""
    lora_cfg = {
        torch.nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=alpha,
                adapter_type=adapter_type,
            ),
        },
        torch.nn.Conv1d: {
            "weight": partial(
                LoRAParametrization.from_conv1d,
                rank=rank,
                lora_alpha=alpha,
                adapter_type=adapter_type,
            ),
        },
    }
    add_lora(model.model, lora_cfg)  # DiTWrapper
    add_lora(model.conditioner, lora_cfg)  # MultiConditioner


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AudioCaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir: str, sample_rate: int, sample_size: int, io_channels: int
    ):
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.io_channels = io_channels
        self.pairs = []
        for path in sorted(Path(data_dir).iterdir()):
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            caption = path.with_suffix(".txt")
            if not caption.exists():
                print(f"Warning: no caption for {path.name}, skipping")
                continue
            self.pairs.append((path, caption))
        if not self.pairs:
            raise ValueError(f"No audio/caption pairs found in {data_dir}")
        print(f"Found {len(self.pairs)} audio/caption pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, caption_path = self.pairs[idx]
        waveform, sr = torchaudio.load(str(audio_path))
        audio = prepare_audio(
            waveform,
            in_sr=sr,
            target_sr=self.sample_rate,
            target_length=self.sample_size,
            target_channels=self.io_channels,
            device="cpu",
        ).squeeze(0)  # [C, T]
        return audio, caption_path.read_text().strip()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = load_model(args.model, device)
    sample_rate = model.sample_rate
    io_channels = model.pretransform.io_channels
    ds_ratio = model.pretransform.downsampling_ratio

    # Align to downsampling ratio
    sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio

    apply_lora(
        model, rank=args.rank, alpha=args.lora_alpha, adapter_type=args.adapter_type
    )
    lora_params = list(get_lora_params(model.model)) + list(
        get_lora_params(model.conditioner)
    )
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    optimizer = AdamW(lora_params, lr=args.lr)

    dataset = AudioCaptionDataset(args.data_dir, sample_rate, sample_size, io_channels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.model.train()

    step = 0
    while step < args.steps:
        for audio_batch, captions in loader:
            if step >= args.steps:
                break

            audio_batch = audio_batch.to(device)

            with torch.no_grad():
                latents = model.pretransform.encode(audio_batch)
                conditioning = model.conditioner(
                    [{"prompt": c, "seconds_total": args.duration} for c in captions],
                    str(device),
                )

            # rf_denoiser noise schedule: xt = (1-t)*x0 + t*noise, target = noise - x0
            B = latents.shape[0]
            t = torch.rand(B, device=device)
            noise = torch.randn_like(latents)
            t_bc = t[:, None, None]
            noised = latents * (1 - t_bc) + noise * t_bc
            target = noise - latents

            # Inpaint model requires mask conditioning; all-zeros = pure generation
            conditioning["inpaint_mask"] = [
                torch.zeros(B, 1, latents.shape[2], device=device)
            ]
            conditioning["inpaint_masked_input"] = [torch.zeros_like(latents)]

            model.model.train()
            pred = model(noised, t, cond=conditioning, cfg_scale=1.0)
            loss = F.mse_loss(pred.float(), target.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                print(f"Step {step}/{args.steps}  loss={loss.item():.4f}")
            if step % args.save_every == 0:
                save_checkpoint(model, args, step)

    save_checkpoint(model, args, step)
    print("Done.")


def save_checkpoint(model, args, step):
    state_dict = {
        **get_lora_state_dict(model.model),
        **get_lora_state_dict(model.conditioner),
    }
    lora_config = {
        "rank": args.rank,
        "alpha": args.lora_alpha,
        "adapter_type": args.adapter_type,
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
    p.add_argument("--model", choices=list(CONFIGS), default="medium-rf")
    p.add_argument(
        "--data_dir",
        required=True,
        help="Folder with audio files and matching .txt captions",
    )
    p.add_argument("--output_dir", default="lora_output")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument(
        "--adapter_type", choices=["lora", "dora", "lora-xs"], default="dora"
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Clip duration in seconds (default 30)",
    )
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    train(p.parse_args())


if __name__ == "__main__":
    main()
