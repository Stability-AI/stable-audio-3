"""
Generate audio for every prompt in song_describer-nosinging.csv.

Usage:
    python scripts/generate_song_describer.py [--model small] [--steps 8] \
        [--output-dir outputs/song_describer]

Outputs one stereo WAV per row, named by caption_id.
Skips already-generated files so the run is resumable.
"""

import argparse
import csv
import os
import sys

import torch
import torchaudio

from stable_audio_3 import StableAudioPipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="small", help="Model name (default: small)")
    p.add_argument("--steps", type=int, default=8, help="Diffusion steps (default: 8)")
    p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Duration in seconds (default: 120)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/song_describer",
        help="Directory to write WAV files (default: outputs/song_describer)",
    )
    p.add_argument(
        "--csv",
        default="song_describer-nosinging.csv",
        help="Path to the CSV file (default: song_describer-nosinging.csv)",
    )
    p.add_argument("--device", default="mps", help="Device to run on (default: mps)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts from CSV
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} prompts from {args.csv}")
    print(
        f"Model: {args.model}  |  Device: {args.device}  |  Steps: {args.steps}  |  Duration: {args.duration}s"
    )
    print(f"Output dir: {args.output_dir}")

    # Load model
    print("Loading model...")
    pipe = StableAudioPipeline.from_pretrained(args.model, args.device)
    sample_rate = pipe.model.sample_rate
    print(f"Model loaded. Sample rate: {sample_rate} Hz")

    skipped = 0
    generated = 0
    errors = 0

    for i, row in enumerate(rows):
        caption_id = row["caption_id"]
        prompt = row["caption"]
        out_path = os.path.join(args.output_dir, f"{caption_id}.wav")

        if os.path.exists(out_path):
            skipped += 1
            continue

        print(
            f"[{i + 1}/{len(rows)}] caption_id={caption_id}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}"
        )

        try:
            audio = pipe.generate(
                prompt=prompt,
                duration=args.duration,
                steps=args.steps,
            )
            # audio shape: [batch, channels, samples] — take first item
            generated += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors += 1

    print(
        f"\nDone. Generated: {generated}  |  Skipped (already exist): {skipped}  |  Errors: {errors}"
    )


if __name__ == "__main__":
    main()
