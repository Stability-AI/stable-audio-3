#!/usr/bin/env python3
"""Interactive build menu for the SA3 TRT engines.

Detects the current GPU's architecture, lists which engines are already built
under ../models/<arch>/ and which are still missing, then loops asking you what
to build next. There's also a "Build all missing" option at the bottom.

Usage:
    python build.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Make _arch.py importable when build.py is invoked from any cwd.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _arch import detect_arch, arch_dir as _arch_dir, repo_root  # noqa: E402


# ─── Colors (auto-detect TTY) ───────────────────────────────────────────────
if sys.stdout.isatty():
    BOLD, DIM    = "\033[1m", "\033[2m"
    RED, GREEN   = "\033[31m", "\033[32m"
    YELLOW, CYAN = "\033[33m", "\033[36m"
    MAGENTA      = "\033[35m"
    RESET        = "\033[0m"
else:
    BOLD = DIM = RED = GREEN = YELLOW = CYAN = MAGENTA = RESET = ""


def fmt_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    s = float(n_bytes)
    for u in units:
        if s < 1024 or u == "GB":
            return f"{s:.0f} {u}" if u in ("B", "KB") else f"{s:.1f} {u}"
        s /= 1024
    return f"{s:.1f} GB"


# Each menu item dispatches `build_from_onnx.py <name>` which pulls the canonical
# ONNX from HuggingFace and compiles it for the local GPU arch. `outputs` are
# paths relative to ../models/<arch>/ — used to mark a target as built.
def _from_onnx(name):
    return [sys.executable, "build_from_onnx.py", name]


TARGETS = [
    {"label": "t5gemma  (text encoder + tokenizer)",
     "command": _from_onnx("t5gemma"),
     "outputs": ["t5gemma/t5gemma_fp16mixed.trt", "t5gemma/tokenizer.json"]},
    {"label": "same-s encoder",
     "command": _from_onnx("same-s-encoder"),
     "outputs": ["same-s/enc_dynamic_bf16.trt"]},
    {"label": "same-s decoder",
     "command": _from_onnx("same-s-decoder"),
     "outputs": ["same-s/dec_dynamic_bf16.trt"]},
    {"label": "same-l encoder (Triton SWA plugin, FP16-mixed)",
     "command": _from_onnx("same-l-encoder"),
     "outputs": ["same-l/enc_dynamic_triton_swa.trt"]},
    {"label": "same-l decoder (Triton SWA plugin, FP16-mixed)",
     "command": _from_onnx("same-l-decoder"),
     "outputs": ["same-l/dec_dynamic_triton_swa.trt"]},
    # DiT engines now build from pre-processed FP16-mixed ONNX on HF;
    # build_from_onnx.py does the simple STRONGLY_TYPED compile.
    # Medium ships TWO engines: fp16-mixed (the DEFAULT since 2026-07 — its
    # attention core now fuses, see below) and bf16 (built from the raw fp32
    # dit.onnx with BuilderFlag.BF16). sm-music/sm-sfx are fp16-mixed only.
    #
    # fp16-mixed became the medium default when the RoPE island was bounded so
    # QK^T + Softmax run FP16: the engine went 0 -> 96 fused MHA nodes and 4.3x
    # faster at L=4096, which removed the speed reason to prefer bf16. It is also
    # the accurate one — teacher-forced velocity cos 1.0000 vs the FP32 engine at
    # every length, where bf16 evaluates RoPE's rotation angle too coarsely and
    # drifts at long sequence (the angle reaches ~4155 rad at L=4092, where bf16's
    # spacing is 32 rad — more than a full 2*pi rotation).
    {"label": "DiT medium  (SA3-M, FP16-mixed — medium DEFAULT, attention-fused)",
     "command": _from_onnx("sa3-m"),
     "outputs": ["sa3-m/dit_fp16mixed.trt"]},
    {"label": "DiT medium  (SA3-M, bf16 — selectable; drifts at long sequence)",
     "command": _from_onnx("sa3-m-bf16"),
     "outputs": ["sa3-m/dit_bf16.trt"]},
    {"label": "DiT medium  (SA3-M, fp8 — selectable; MAX-SPEED clean tier, ~1.3x over fp16mixed)",
     "command": _from_onnx("sa3-m-fp8"),
     "outputs": ["sa3-m/dit_fp8.trt"]},
    {"label": "DiT sm-music (FP16-mixed)",
     "command": _from_onnx("sa3-sm-music"),
     "outputs": ["sa3-sm-music/dit_fp16mixed.trt"]},
    {"label": "DiT sm-sfx (FP16-mixed)",
     "command": _from_onnx("sa3-sm-sfx"),
     "outputs": ["sa3-sm-sfx/dit_fp16mixed.trt"]},
    # FP32 variants — opt-in. ~2× engine size, ~2× slower, but bit-equivalent
    # to PyTorch eager. Useful for precision-debug or reference comparisons.
    {"label": "[opt-in] same-l decoder FP32",
     "command": _from_onnx("same-l-decoder-fp32"),
     "outputs": ["same-l/dec_dynamic_fp32.trt"]},
    {"label": "[opt-in] same-s decoder FP32",
     "command": _from_onnx("same-s-decoder-fp32"),
     "outputs": ["same-s/dec_dynamic_fp32.trt"]},
    {"label": "[opt-in] DiT medium FP32",
     "command": _from_onnx("sa3-m-fp32"),
     "outputs": ["sa3-m/dit_fp32.trt"]},
    {"label": "[opt-in] DiT sm-music FP32",
     "command": _from_onnx("sa3-sm-music-fp32"),
     "outputs": ["sa3-sm-music/dit_fp32.trt"]},
    {"label": "[opt-in] DiT sm-sfx FP32",
     "command": _from_onnx("sa3-sm-sfx-fp32"),
     "outputs": ["sa3-sm-sfx/dit_fp32.trt"]},
]


def target_status(target: dict, arch_dir: Path) -> tuple[bool, list[tuple[str, int]]]:
    """Return (all_built, [(rel_path, size_bytes_or_-1), ...])."""
    rows = []
    all_built = True
    for rel in target["outputs"]:
        p = arch_dir / rel
        if p.exists() and p.stat().st_size > 0:
            rows.append((rel, p.stat().st_size))
        else:
            rows.append((rel, -1))
            all_built = False
    return all_built, rows


def render_menu(arch: str, arch_dir: Path) -> list[bool]:
    """Print the menu. Returns a list parallel to TARGETS of bool 'is_built'."""
    print()
    print(f"  {BOLD}GPU arch:{RESET}   {CYAN}{arch}{RESET}")
    try:
        rel_dir = arch_dir.relative_to(repo_root())
        rel_dir_str = str(rel_dir)
    except ValueError:
        rel_dir_str = str(arch_dir)
    print(f"  {BOLD}Output dir:{RESET} {DIM}{rel_dir_str}/{RESET}")
    print()

    label_w = max(len(t["label"]) for t in TARGETS)
    built_flags = []
    for i, t in enumerate(TARGETS, start=1):
        all_built, rows = target_status(t, arch_dir)
        built_flags.append(all_built)
        head_mark = f"{GREEN}✓{RESET}" if all_built else f"{RED}✗{RESET}"
        print(f"  {BOLD}[{i}]{RESET} {head_mark}  {t['label']:<{label_w}}")
        for rel, sz in rows:
            tick = f"{GREEN}✓{RESET}" if sz >= 0 else f"{RED}✗{RESET}"
            size_s = fmt_size(sz) if sz >= 0 else f"{DIM}(missing){RESET}"
            print(f"        {tick}  {DIM}{rel}{RESET}  {size_s}")

    n_missing = built_flags.count(False)
    print()
    if n_missing == 0:
        print(f"  {BOLD}{GREEN}[A]{RESET} Build all missing  {DIM}(nothing missing — all engines built){RESET}")
    else:
        print(f"  {BOLD}{YELLOW}[A]{RESET} Build all missing  {DIM}({n_missing} target(s)){RESET}")
    print(f"  {BOLD}{DIM}[Q]{RESET} Quit")
    return built_flags


def run_build(target: dict) -> bool:
    """Invoke a build script in build/. Streams output. Returns True on success."""
    print()
    print(f"  {CYAN}→{RESET} {BOLD}{target['label']}{RESET}")
    print(f"    {DIM}cmd: {' '.join(target['command'])}{RESET}")
    print()
    # Always run from build/ so relative imports + ../models/ resolve correctly.
    rc = subprocess.call(target["command"], cwd=str(SCRIPTS_DIR))
    if rc == 0:
        print()
        print(f"  {GREEN}✓ built {target['label']}{RESET}")
        return True
    print()
    print(f"  {RED}✗ build failed (exit {rc}): {target['label']}{RESET}")
    return False


def main() -> int:
    arch = detect_arch()
    arch_dir = Path(_arch_dir(arch))
    print()
    print(f"{BOLD}{CYAN}━━━ SA3 TRT engine build menu ━━━{RESET}")

    while True:
        built_flags = render_menu(arch, arch_dir)
        print()
        try:
            choice = input(f"  Choose [1-{len(TARGETS)} / A / Q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice in ("q", "quit", "exit", ""):
            return 0

        if choice in ("a", "all"):
            missing = [t for t, ok in zip(TARGETS, built_flags) if not ok]
            if not missing:
                print(f"  {DIM}Nothing to build.{RESET}")
                continue
            print()
            print(f"  {BOLD}Building {len(missing)} missing target(s) in order...{RESET}")
            for t in missing:
                if not run_build(t):
                    print(f"  {YELLOW}Stopping due to error. Re-run to retry.{RESET}")
                    break
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(TARGETS):
                run_build(TARGETS[idx - 1])
                continue

        print(f"  {RED}invalid choice: {choice}{RESET}")


if __name__ == "__main__":
    sys.exit(main())
