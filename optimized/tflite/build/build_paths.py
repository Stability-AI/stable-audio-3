"""Shared build config — resolves the work dir, source checkpoints, and makes torch_defs importable.
Every build script starts with:  from build_paths import WORK, ckpt
No machine-specific paths are baked in — you set two things via the environment:

    SA3_BUILD_WORK   output/scratch dir for all intermediates + the 8 final rungs   (default: build/_work)
    SA3_CKPT_MEDIUM  path to the sa3-medium checkpoint  (.safetensors, has pretransform.model.{encoder,decoder})
    SA3_CKPT_SMMUSIC path to the sa3-sm-music checkpoint (the SAME-S autoencoder)

Download the checkpoints from HuggingFace first (see build/README.md).
"""
import os
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD / "torch_defs"))          # torch model defs importable by extract/export scripts

WORK = Path(os.environ.get("SA3_BUILD_WORK", BUILD / "_work"))
WORK.mkdir(parents=True, exist_ok=True)
for _sub in ("same-l", "same-s"):
    (WORK / _sub).mkdir(exist_ok=True)

_CKPT_ENV = {"medium": "SA3_CKPT_MEDIUM", "sm-music": "SA3_CKPT_SMMUSIC"}
_CKPT_HF = {"medium": "stabilityai/stable-audio-3-medium (ARC .safetensors)",
            "sm-music": "stabilityai/stable-audio-3-sm-music (ckpt)"}


def ckpt(name: str) -> str:
    """Absolute path to a source checkpoint (from its env var). Raises with guidance if unset."""
    env = _CKPT_ENV[name]
    p = os.environ.get(env)
    if not p or not Path(p).exists():
        raise SystemExit(f"Set ${env} to the {name} checkpoint file — download "
                         f"{_CKPT_HF[name]} from HuggingFace. See build/README.md.")
    return p
