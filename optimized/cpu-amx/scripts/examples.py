"""Shared, colored "Try these commands" block for the cpu-amx release.

Appended to `./sa3 --help`. The cpu-amx runtime ships the MEDIUM DiT only (the
int8 C++ AMX core), so every example uses `--dit medium`; the decoder toggles
between the native `same-l` and the faster distilled `same-s`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""

BOLD, CYAN, GREEN = _c("\033[1m"), _c("\033[1;36m"), _c("\033[1;32m")
YELLOW, DIM, RESET = _c("\033[1;33m"), _c("\033[2m"), _c("\033[0m")


def _prefix() -> str:
    wrapper = SCRIPT_DIR / "sa3"
    if wrapper.exists() and os.access(wrapper, os.X_OK):
        return "./sa3"
    return "python scripts/sa3_cpu_amx.py"


def print_example_commands(header: str | None = None) -> None:
    prefix = _prefix()

    def hdr(text):
        print(f"\n  {CYAN}{text}{RESET}")

    def cmd(args, comment=""):
        line = f"{prefix} {args}"
        if comment:
            print(f"    {GREEN}$ {line}{RESET}  {DIM}# {comment}{RESET}")
        else:
            print(f"    {GREEN}$ {line}{RESET}")

    if header is None:
        header = f"{BOLD}Examples:{RESET}"
    print(f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"  {header}")
    print(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

    hdr("🎵 Generate audio from a prompt")
    cmd('--prompt "A beautiful piano arpeggio grows into a cinematic climax" \\\n'
        '        --dit medium --decoder same-l --seconds 30 --out piano.wav',
        "native codec, best fidelity")
    cmd('--prompt "lofi house loop, 120 BPM" \\\n'
        '        --dit medium --decoder same-s --seconds 15 --out lofi.wav',
        "distilled same-s decoder — faster")

    hdr("▶  Play immediately after generation")
    cmd('--prompt "ambient drone" --dit medium --decoder same-l \\\n'
        '        --seconds 10 --out drone.wav --play',
        "writes WAV + plays (ffplay/aplay/paplay/afplay)")

    hdr("🎚️  Audio-to-audio & inpainting (needs an input WAV; C++ AMX SAME-{S,L} encoder)")
    cmd('--prompt "jazz fusion with electric piano" --dit medium --decoder same-l \\\n'
        '        --init-audio funk.wav --init-noise-level 0.7 --out funk_jazz.wav',
        "variation: 0.4-0.8 typical")
    cmd('--prompt "explosive drum break" --dit medium --decoder same-l \\\n'
        '        --init-audio funk.wav --inpaint-range "4,7" --out funk_drums.wav',
        "regenerate seconds 4-7, keep the rest")

    hdr("🎯 Steer with CFG + negative prompts")
    cmd('--prompt "ambient drone" --cfg 3.0 \\\n'
        '        --negative-prompt "drums, vocals, distortion" \\\n'
        '        --dit medium --decoder same-l --out clean_drone.wav',
        "cfg > 1.0 toward prompt, neg pushes away")

    hdr("⚙️  Precision / speed dials")
    cmd('--prompt "techno beat" --dit medium --decoder same-s \\\n'
        '        --decoder-precision int8 --threads 32 --out techno.wav',
        "int8 fused decoder (smaller/faster); more threads")

    print(f"\n  {YELLOW}note:{RESET} cpu-amx ships the MEDIUM DiT only. For sm-music / sm-sfx use "
          f"optimized/tflite or optimized/mlx.")
    print(f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")
