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
  uv run python pre_encode_dataset.py --model same-s --data_dir ./my_data --output_path ./latents_out
  uv run python pre_encode_dataset.py --model same-l --data_dir ./my_data --output_path ./latents_out --batch_size 4
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stable_audio_3 import AutoencoderPipeline
from stable_audio_3.model_configs import ae_models
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

    ae = AutoencoderPipeline.from_pretrained(args.model, device=str(device))
    if args.model_half:
        ae.autoencoder = ae.autoencoder.half()

    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train", path=args.data_dir, custom_metadata_fn=caption_metadata_fn
            )
        ],
        sample_size=args.sample_size,
        sample_rate=ae.sample_rate,
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

    silence_path = os.path.join(args.output_path, "silence.npy")
    if not os.path.exists(silence_path):
        print("Saving silence latent")
        silence_audio = torch.zeros(
            1, ae.autoencoder.io_channels, args.sample_size, device=device
        )
        if args.model_half:
            silence_audio = silence_audio.half()
        with torch.no_grad():
            silence_latent = ae.encode(silence_audio, ae.sample_rate)
        np.save(silence_path, silence_latent.cpu().numpy())

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

        latents = ae.encode(audio, ae.sample_rate)

        for i, latent in enumerate(latents):
            latent_np = latent.cpu().numpy()
            latent_id = f"{nb:06d}{i:04d}"

            md = dict(metadata[i])
            padding_mask = (
                F.interpolate(
                    md["padding_mask"].unsqueeze(0).unsqueeze(1).float(),
                    size=latent_np.shape[-1],
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .int()
            )
            if args.no_pad:
                padding_np = padding_mask.cpu().numpy()
                valid_indices = np.where(padding_np == 1)[0]
                if len(valid_indices) > 0:
                    valid_length = valid_indices[-1] + 1
                    latent_np = latent_np[:, :valid_length]
                    padding_mask = padding_mask[:valid_length]

            np.save(os.path.join(args.output_path, f"{latent_id}.npy"), latent_np)

            md["padding_mask"] = padding_mask.cpu().numpy().tolist()
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(os.path.join(args.output_path, f"{latent_id}.json"), "w") as f:
                json.dump(md, f)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode audio dataset to latents")
    parser.add_argument("--model", choices=list(ae_models), default="same-l")
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
    parser.add_argument(
        "--no_pad",
        action="store_true",
        help="Trim latents to valid audio length (variable-length mode); requires --batch_size 1",
    )
    args = parser.parse_args()

    if args.no_pad and args.batch_size > 1:
        parser.error(
            "--no_pad requires --batch_size 1 (variable-length samples cannot be batched)"
        )

    main(args)
