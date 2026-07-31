"""Compatibility import for the repository's shared WAV helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "stable_audio_3"))

from audio_output import protect_audio_peak, save_wav  # noqa: E402, F401
