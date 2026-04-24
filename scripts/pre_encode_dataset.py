"""
Pre-encode a dataset of audio clips into latents using Stable Audio 3, saving the latents and metadata to disk.

Dataset layout:
  data_dir/
    clip1.wav   (or .flac, .mp3, .ogg)
    clip1.txt   ← text prompt for clip1
    clip2.wav
    clip2.txt
    ...

Saves .npy files for latents and .json files for metadata, compatible with train_lora.py --encoded_dir.

Usage:
  uv run python pre_encode_dataset.py --model small --data_dir ./my_data --output_path ./latents_out
  uv run python pre_encode_dataset.py --model medium --data_dir ./my_data --output_path ./latents_out --batch_size 4
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from stable_audio_3.loading_utils import load_autoencoder
from stable_audio_3.model_configs import all_models
from stable_audio_3.data.dataset import (
    LocalDatasetConfig,
    SampleDataset,
    collation_fn,
)


def caption_metadata_fn(info, _audio):
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = all_models[args.model]
    local_config, local_ckpt = cfg.resolve()
    autoencoder = load_autoencoder(local_config, local_ckpt, device=str(device))
    autoencoder.eval().requires_grad_(False)
    if args.model_half:
        autoencoder = autoencoder.half()

    with open(local_config) as f:
        sample_rate = json.load(f)["sample_rate"]

    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train", path=args.data_dir, custom_metadata_fn=caption_metadata_fn
            )
        ],
        sample_size=args.sample_size,
        sample_rate=sample_rate,
        force_channels="stereo",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=False,
        collate_fn=collation_fn,
    )

    os.makedirs(args.output_path, exist_ok=True)

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

        with torch.no_grad():
            latents = autoencoder.encode(audio)

        for i, latent in enumerate(latents):
            latent_np = latent.cpu().numpy()
            latent_id = f"{nb:06d}{i:04d}"

            np.save(os.path.join(args.output_path, f"{latent_id}.npy"), latent_np)

            md = dict(metadata[i])
            padding_mask = resize_padding_mask(
                md["padding_mask"], latent_np.shape[-1]
            ).int()
            md["padding_mask"] = padding_mask.cpu().numpy().tolist()
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(os.path.join(args.output_path, f"{latent_id}.json"), "w") as f:
                json.dump(md, f)

    print("Done")


def resize_padding_mask(padding_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize a padding mask to target_length using ceiling-based length scaling.

    Unlike F.interpolate(mode="nearest"), this ensures any target position
    that partially overlaps valid audio is marked valid (rounds up).
    """
    if padding_mask.ndim == 1:
        valid_length = padding_mask.sum()
        source_length = padding_mask.shape[0]
        valid_target_length = (
            torch.ceil(valid_length.float() * target_length / source_length)
            .long()
            .clamp(max=target_length)
        )
        positions = torch.arange(target_length, device=padding_mask.device)
        return positions < valid_target_length
    else:
        valid_lengths = padding_mask.sum(dim=-1)  # (B,)
        source_length = padding_mask.shape[-1]
        valid_target_lengths = (
            torch.ceil(valid_lengths.float() * target_length / source_length)
            .long()
            .clamp(max=target_length)
        )
        positions = torch.arange(target_length, device=padding_mask.device).unsqueeze(0)
        return positions < valid_target_lengths.unsqueeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode audio dataset to latents")
    parser.add_argument("--model", choices=list(all_models), default="medium-rf")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Folder with audio files and matching .txt captions",
    )
    parser.add_argument(
        "--output_path", required=True, help="Folder to write .npy/.json latent pairs"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=12582912,  # 380s at 44.1kHz, 2 channels
        help="Audio samples to pad/crop to (default ~380s at 44.1kHz)",
    )
    parser.add_argument(
        "--model_half", action="store_true", help="Run autoencoder in fp16"
    )
    args = parser.parse_args()
    main(args)
