"""Compatibility imports for the repository's shared WAV helpers."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
FULL_REPO_HELPER_DIR = Path(__file__).resolve().parents[4] / "stable_audio_3"
HELPER_DIR = (
    FULL_REPO_HELPER_DIR
    if (FULL_REPO_HELPER_DIR / "audio_output.py").is_file()
    else THIS_DIR
)
if not (HELPER_DIR / "audio_output.py").is_file():
    raise ModuleNotFoundError(
        "shared audio_output.py is missing; reinstall the TFLite bundle or use a full checkout"
    )
sys.path.insert(0, str(HELPER_DIR))

from audio_output import (  # noqa: E402, F401
    audio_to_pcm16,
    dbfs_to_amplitude,
    protect_audio_peak,
    save_wav,
)
