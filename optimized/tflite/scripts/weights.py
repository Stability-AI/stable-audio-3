"""Shared weights manifest + downloader.

Maps every TFLite / LiteRT model file the runtime needs to its position in the
`stabilityai/stable-audio-3-optimized` HuggingFace repo (under `tflite/…`).

`install.py` calls `ensure_local` upfront for the bundles the user picks.
`sa3_tflite.py` calls `ensure_local` lazily, just before each model loads — so
a fresh checkout with no weights still works if the user is willing to wait for
the first run to download them.

The SentencePiece tokenizer is BUNDLED at models/tokenizer.model (the .tflite
T5Gemma is encoder-only), so it's not in this manifest.
"""

from __future__ import annotations

from pathlib import Path

REPO_ID = "stabilityai/stable-audio-3-optimized"
# weights.py lives in <project>/scripts/; SCRIPT_DIR points at the project
# root so the local rel paths in the manifest ("models/tflite/foo.tflite")
# resolve against the actual project layout.
SCRIPT_DIR = Path(__file__).resolve().parent.parent


# Bundles the install script offers to the user. Each maps to a list of
# model files (local relative path on the left, HF repo path on the right).
# T5Gemma is in SHARED because every bundle needs it. The two small DiTs
# share the SAME-S codec; medium uses the SAME-L codec.

DIT_BUNDLES: dict[str, list[tuple[str, str]]] = {
    "sm-music": [
        ("models/tflite/sa3-sm-music/dit_fp32.tflite", "tflite/sa3-sm-music/dit_fp32.tflite"),
        ("models/tflite/same-s/enc_fp32.tflite",       "tflite/same-s/enc_fp32.tflite"),
        ("models/tflite/same-s/dec_fp32.tflite",       "tflite/same-s/dec_fp32.tflite"),
    ],
    "sm-sfx": [
        ("models/tflite/sa3-sm-sfx/dit_fp32.tflite",   "tflite/sa3-sm-sfx/dit_fp32.tflite"),
        ("models/tflite/same-s/enc_fp32.tflite",       "tflite/same-s/enc_fp32.tflite"),
        ("models/tflite/same-s/dec_fp32.tflite",       "tflite/same-s/dec_fp32.tflite"),
    ],
    "medium": [
        ("models/tflite/sa3-m/dit_fp32.tflite",        "tflite/sa3-m/dit_fp32.tflite"),
        ("models/tflite/same-l/enc_fp32.tflite",       "tflite/same-l/enc_fp32.tflite"),
        ("models/tflite/same-l/dec_fp32.tflite",       "tflite/same-l/dec_fp32.tflite"),
    ],
}

SHARED: list[tuple[str, str]] = [
    ("models/tflite/t5gemma/encoder_fp16.tflite", "tflite/t5gemma/encoder_fp16.tflite"),
]

# Human-friendly bundle sizes (for the install prompt). Exact, from HF metadata.
BUNDLE_SIZES = {
    "sm-music": "2.3 GB  (small music DiT + SAME-S codec, all fp32)",
    "sm-sfx":   "2.3 GB  (small sfx DiT + SAME-S codec, all fp32)",
    "medium":   "9.5 GB  (medium DiT + SAME-L codec, all fp32)",
}
# T5Gemma (shared, fp16) adds ~0.6 GB the first time any bundle is fetched.

# Quantized DiT + decoder variants (selected via sa3_tflite.py --precision; wXaY =
# weight/activation bit-widths, "16" = fp16). Not part of the install bundles — they
# lazy-download on first use. w16a32 = fp16 weights (half size, ≈lossless, slower on
# CPU); w8a32 / w8a8-dyn = GPTQ-calibrated int8 weights.
QUANT_PRECISIONS = ("w16a32", "w8a32", "w8a8-dyn")
PRECISIONS = ("fp32",) + QUANT_PRECISIONS   # everything --precision accepts
DIT_SUBDIR = {"sm-music": "sa3-sm-music", "sm-sfx": "sa3-sm-sfx", "medium": "sa3-m"}


def dit_rel(dit: str, precision: str = "fp32") -> str:
    """Local rel path of a DiT model file for (family, precision)."""
    return f"models/tflite/{DIT_SUBDIR[dit]}/dit_{precision}.tflite"


def dec_rel(dec: str, precision: str = "fp32") -> str:
    """Local rel path of a SAME codec decoder file for (codec, precision)."""
    return f"models/tflite/{dec}/dec_{precision}.tflite"


def enc_rel(dec: str, precision: str = "fp32") -> str:
    """Local rel path of a SAME codec encoder file for (codec, precision).
    Encoder int8 is GPTQ-calibrated on real audio (torch-free, in-place on the
    tflite graph) — w8a32 measures 36/46 dB latent PSNR (same-s/same-l)."""
    return f"models/tflite/{dec}/enc_{precision}.tflite"

# Flat (local_rel_path → hf_path) lookup — used by sa3_tflite.py for lazy
# auto-download at load time.
FLAT_MANIFEST: dict[str, str] = {}
for _items in DIT_BUNDLES.values():
    for _rel, _hf in _items:
        FLAT_MANIFEST[_rel] = _hf
for _rel, _hf in SHARED:
    FLAT_MANIFEST[_rel] = _hf
for _prec in QUANT_PRECISIONS:
    for _fam in DIT_SUBDIR:
        _rel = dit_rel(_fam, _prec)
        FLAT_MANIFEST[_rel] = _rel.replace("models/tflite/", "tflite/", 1)
    for _dec in ("same-s", "same-l"):
        for _rel in (dec_rel(_dec, _prec), enc_rel(_dec, _prec)):
            FLAT_MANIFEST[_rel] = _rel.replace("models/tflite/", "tflite/", 1)


def _hf_token_configured() -> bool:
    """True if any HF token is set — env var or cached login on disk."""
    import os
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token  # huggingface_hub ≥ 0.19
        return bool(get_token())
    except ImportError:
        try:
            from huggingface_hub import HfFolder
            return bool(HfFolder.get_token())
        except Exception:
            return False
    except Exception:
        return False


_LOGIN_TIP_SHOWN = False

def _show_hf_login_tip_once() -> None:
    """Print a one-time login suggestion if no HF token is configured.

    Anonymous downloads work but have a ~50 GB/day soft cap on HF's LFS CDN
    and lower aggregate bandwidth — a free token effectively removes both.
    Stays silent if a token is already in place.
    """
    global _LOGIN_TIP_SHOWN
    if _LOGIN_TIP_SHOWN:
        return
    _LOGIN_TIP_SHOWN = True
    if _hf_token_configured():
        return
    import sys
    YEL  = "\033[1;33m" if sys.stdout.isatty() else ""
    BOLD = "\033[1m"    if sys.stdout.isatty() else ""
    DIM  = "\033[2m"    if sys.stdout.isatty() else ""
    RST  = "\033[0m"    if sys.stdout.isatty() else ""
    print()
    print(f"  {YEL}⚠  not logged in to HuggingFace{RST} — anonymous downloads work but are")
    print(f"     rate-limited (~50 GB/day cap on the LFS CDN). For faster, higher-limit")
    print(f"     downloads, log in once with a free read-only token:")
    print()
    print(f"       1. create an account at {BOLD}https://huggingface.co/join{RST}")
    print(f"       2. generate a token at {BOLD}https://huggingface.co/settings/tokens{RST}")
    print(f"          {DIM}('Read' scope is enough){RST}")
    print(f"       3. save it on this machine — pick one:")
    print(f"            {BOLD}hf auth login{RST}              {DIM}# modern (huggingface_hub ≥ 1.0){RST}")
    print(f"            {BOLD}huggingface-cli login{RST}      {DIM}# classic; still works{RST}")
    print(f"            {BOLD}export HF_TOKEN=hf_xxx{RST}     {DIM}# one-off / scripts{RST}")
    print()


def ensure_local(local_rel_path: str, verbose: bool = True) -> Path:
    """Resolve a model file to an absolute local path, downloading if missing.

    Files are streamed into the HuggingFace cache (~/.cache/huggingface/hub/)
    and symlinked into the project at `local_rel_path` so the on-disk layout
    looks the same whether the file was bundled or downloaded.
    """
    target = SCRIPT_DIR / local_rel_path
    if target.exists() or target.is_symlink():
        return target

    if local_rel_path not in FLAT_MANIFEST:
        raise FileNotFoundError(
            f"{local_rel_path} is not in the weights manifest — can't auto-download."
        )

    # First-download tip: nudge users toward logging in to HF for better limits.
    # No-op if a token is already configured.
    _show_hf_login_tip_once()

    hf_filename = FLAT_MANIFEST[local_rel_path]
    if verbose:
        print(f"  ↓ downloading {hf_filename}  (from {REPO_ID})")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download weights.\n"
            "Run:  pip install huggingface_hub\n"
            "Or run the install.py script in this directory."
        ) from e

    cached = hf_hub_download(repo_id=REPO_ID, filename=hf_filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Symlink keeps the HF cache canonical (one copy on disk) while exposing
    # the file at the project-relative path the runtime expects.
    if target.is_symlink():
        target.unlink()
    target.symlink_to(cached)
    return target


def is_present(local_rel_path: str) -> bool:
    """True if the file exists locally (does not trigger a download)."""
    p = SCRIPT_DIR / local_rel_path
    return p.exists() or p.is_symlink()


def bundle_status(bundle: str) -> tuple[int, int]:
    """Returns (present_count, total_count) for the bundle (including SHARED)."""
    items = DIT_BUNDLES[bundle] + SHARED
    present = sum(1 for rel, _ in items if is_present(rel))
    return present, len(items)
