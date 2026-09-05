"""Full-stack self-test for the cpu-amx release.

Phase 1 verifies the vendored assets + the C++ backend .so/weights are present.
Phase 2 runs every CLI configuration end-to-end (medium DiT only) and gates each
output WAV on: correct duration (±0.1s), finite samples, and not-silent
(peak ≥ 0.005, rms ≥ 0.0005).

    python scripts/test_all_configs.py
"""
from __future__ import annotations

import subprocess
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

_HOME = os.environ.get("SA3_CPUAMX_HOME",
                       os.path.expanduser("~/.cache/stable-audio-3/cpu-amx"))

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
PY = sys.executable
CLI = str(SCRIPTS / "sa3_cpu_amx.py")

TIMEOUT = 300   # CPU is slower than MLX-GPU; a2a/inpaint pay a one-time torch AE load


# ─── phase 1: assets + backend engines present ─────────────────────────────
EXPECTED_ASSETS = [
    ROOT / "assets" / "t5gemma_f16.npz",     # SentencePiece tokenizer
    ROOT / "assets" / "cond_medium.npz",     # conditioner weights
]
EXPECTED_SO = [
    os.path.join(_HOME, "t5gemma_cpu_amx", "t5gemma_cpu_amx.so"),
    os.path.join(_HOME, "dit_medium_cpu_amx", "dit_cpu_amx.so"),
    os.path.join(_HOME, "same_s_cpu_amx", "same_s_cpu_amx.so"),
    os.path.join(_HOME, "same_l_cpu_amx", "same_l_cpu_amx.so"),
    os.path.join(_HOME, "same_s_int8fused_cpu_amx", "same_s_int8fused_cpu_amx.so"),
    os.path.join(_HOME, "same_l_int8fused_cpu_amx", "same_l_int8fused_cpu_amx.so"),
    os.path.join(_HOME, "same_s_encoder_cpu_amx", "same_s_encoder_cpu_amx.so"),   # a2a/inpaint init-encode
    os.path.join(_HOME, "same_l_encoder_cpu_amx", "same_l_encoder_cpu_amx.so"),
]


def test_assets():
    print("\n[ phase 1 ] vendored assets + C++ AMX engines present\n")
    fails = []
    for p in EXPECTED_ASSETS + [Path(x) for x in EXPECTED_SO]:
        ok = p.exists()
        sz = f"{p.stat().st_size/1e6:.1f} MB" if ok else "MISSING"
        print(f"  {'✓' if ok else '✗'} {str(p):68s}  {sz}")
        if not ok:
            fails.append(str(p))
    return fails


# ─── phase 2: CLI matrix ────────────────────────────────────────────────────
def read_wav(path):
    with wave.open(str(path), "rb") as w:
        nch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), np.int16).reshape(-1, nch).T.astype(np.float32) / 32767.0
    return pcm, sr


def run_cli(name, extra_args, expected_seconds):
    out = Path(tempfile.gettempdir()) / f"sa3cpuamx_test_{name}.wav"
    cmd = [PY, CLI, "--seed", "42", "--out", str(out)] + extra_args
    if "--steps" not in extra_args:
        cmd += ["--steps", "4"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {TIMEOUT}s"
    dt = time.time() - t0
    if proc.returncode != 0:
        last = proc.stderr.strip().split("\n")[-1] if proc.stderr.strip() else "(no stderr)"
        return False, f"exit {proc.returncode}: {last[:140]}"
    if not out.exists():
        return False, "no output WAV produced"
    pcm, sr = read_wav(out)
    actual = pcm.shape[-1] / sr
    if abs(actual - expected_seconds) > 0.1:
        return False, f"duration {actual:.2f}s ≠ {expected_seconds:.2f}s"
    if not np.isfinite(pcm).all():
        return False, "audio contains NaN/Inf"
    peak, rms = float(np.abs(pcm).max()), float(np.sqrt((pcm ** 2).mean()))
    if peak < 0.005 or rms < 0.0005:
        return False, f"effectively silent (peak={peak:.4f} rms={rms:.4f})"
    out.unlink(missing_ok=True)
    return True, f"{dt:.1f}s  peak={peak:.3f} rms={rms:.3f}"


def test_configs():
    print("\n[ phase 2 ] end-to-end CLI configurations (medium DiT, steps=4, 3s clips)\n")
    SEC, EXP = "3", 3.0

    init_wav = Path(tempfile.gettempdir()) / "sa3cpuamx_test_init.wav"
    print("  [setup] generating a small init clip for a2a/inpaint …")
    r = subprocess.run([PY, CLI, "--prompt", "drums", "--dit", "medium", "--decoder", "same-s",
                        "--seconds", SEC, "--seed", "1", "--steps", "4", "--out", str(init_wav)],
                       capture_output=True, text=True, timeout=TIMEOUT, cwd=str(ROOT))
    if r.returncode != 0 or not init_wav.exists():
        print(f"  [setup] FAILED to make init clip: {r.stderr.strip().splitlines()[-1:]}")
        return [("setup-init", "init clip generation failed")]
    print(f"  [setup] init clip at {init_wav}\n")

    D = ["--dit", "medium"]
    matrix = [
        # text-to-audio: medium × each decoder
        ("t2a-same-s",  ["--prompt", "lofi house", *D, "--decoder", "same-s", "--seconds", SEC]),
        ("t2a-same-l",  ["--prompt", "piano solo", *D, "--decoder", "same-l", "--seconds", SEC]),
        # decoder int8 precision (both codecs)
        ("t2a-s-int8",  ["--prompt", "lofi house", *D, "--decoder", "same-s", "--seconds", SEC,
                         "--decoder-precision", "int8"]),
        ("t2a-l-int8",  ["--prompt", "piano solo", *D, "--decoder", "same-l", "--seconds", SEC,
                         "--decoder-precision", "int8"]),
        # audio-to-audio
        ("a2a",         ["--prompt", "ambient", *D, "--decoder", "same-l", "--seconds", SEC,
                         "--init-audio", str(init_wav), "--init-noise-level", "0.6"]),
        # inpainting (paste-back)
        ("inpaint",     ["--prompt", "guitar solo", *D, "--decoder", "same-s", "--seconds", SEC,
                         "--init-audio", str(init_wav), "--inpaint-range", "1,2"]),
        # CFG variants
        ("cfg3",        ["--prompt", "techno beat", *D, "--decoder", "same-l", "--seconds", SEC,
                         "--cfg", "3.0"]),
        ("cfg3-neg",    ["--prompt", "techno beat", *D, "--decoder", "same-l", "--seconds", SEC,
                         "--cfg", "3.0", "--negative-prompt", "vocals, drums"]),
        ("cfg0.5",      ["--prompt", "techno", *D, "--decoder", "same-l", "--seconds", SEC,
                         "--cfg", "0.5"]),
        ("cfg-apg0",    ["--prompt", "techno", *D, "--decoder", "same-l", "--seconds", SEC,
                         "--cfg", "3.0", "--apg", "0.0"]),
        # step counts
        ("steps1",      ["--prompt", "lofi", *D, "--decoder", "same-l", "--seconds", SEC, "--steps", "1"]),
        ("steps16",     ["--prompt", "lofi", *D, "--decoder", "same-l", "--seconds", SEC, "--steps", "16"]),
        # empty prompt (unconditional)
        ("empty-prompt", ["--prompt", "", *D, "--decoder", "same-l", "--seconds", SEC]),
        # no-free-models
        ("no-free",     ["--prompt", "lofi", *D, "--decoder", "same-l", "--seconds", SEC, "--no-free-models"]),
    ]

    fails = []
    for name, args in matrix:
        ok, info = run_cli(name, args, EXP)
        print(f"  {'✓' if ok else '✗'} {name:14s}  {info}")
        if not ok:
            fails.append((name, info))
    init_wav.unlink(missing_ok=True)
    return fails


def main():
    print("=" * 72)
    print("sa3 cpu-amx full-stack self-test")
    print("=" * 72)
    asset_fails = test_assets()
    config_fails = test_configs()
    print("\n" + "=" * 72 + "\nSUMMARY\n" + "=" * 72)
    if not asset_fails and not config_fails:
        print("✓ ALL PASS")
        return 0
    print(f"asset failures : {len(asset_fails)}")
    for f in asset_fails:
        print(f"    - {f}")
    print(f"config failures: {len(config_fails)}")
    for name, info in config_fails:
        print(f"    - {name}: {info}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
