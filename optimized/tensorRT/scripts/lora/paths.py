"""Where this install keeps its model files.

Three things need locating and none of them belong in source: the TensorRT engine and its
branch map (built locally -- TensorRT bakes the GPU architecture in, so they cannot ship),
the base checkpoint (only the verification harnesses need it), and the frozen SVD bases
(only `-xs` adapters need those).

Everything resolves relative to this checkout, overridable by environment variable, and a
missing file raises where it is asked for rather than resolving to the wrong model. That
last part is the point: an earlier version let one constant silently resolve to a different
model's SVD bases, which folds an -xs adapter into a rotated basis with nothing raising.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The LoRA merge math (all 8 adapter variants) already ships in this repo for the TFLite
# path. It is the product's ground truth for what an adapter means, so it is imported, not
# forked -- two copies of that file would be two definitions of "merged weight".
_LORA_CORE = Path(__file__).resolve().parents[3] / "tflite" / "scripts"
if str(_LORA_CORE) not in sys.path:
    sys.path.insert(0, str(_LORA_CORE))

# Built locally by lora/build_branch.py.
ENGINE_DIR = Path(os.environ.get("SA3_ENGINE_DIR", ROOT / "engines"))
BRANCH_ENGINE = ENGINE_DIR / "dit_fp16_lora.trt"
BRANCH_MAP = Path(os.environ.get("SA3_BRANCH_MAP", ROOT / "lora" / "branch_map_medium_lora.json"))

# Downloaded: stabilityai/stable-audio-3-optimized, onnx/sa3-m/dit_fp16.onnx (+ .data).
ONNX_DIR = Path(os.environ.get("SA3_ONNX_DIR", ROOT / "onnx" / "sa3-m"))
ONNX = ONNX_DIR / "dit_fp16.onnx"

# Only the verify_* harnesses need these; normal adapter loading never touches them.
CKPT_DIR = Path(os.environ.get("SA3_CKPT_DIR", ROOT / "models" / "sa3-medium"))
SVD_BASES = Path(os.environ.get("SA3_SVD_BASES", CKPT_DIR / "svd_bases.pt"))

# SA3-medium is ONE model everywhere: an adapter trained on any variant applies to the
# others, so the variant is a naming detail, not a compatibility gate.
# The published onnx/sa3-m/dit_fp16.onnx is built from ARC. Folding an adapter's row
# norms against the other variant reads as a 27% fold error, so this is not a
# preference: it is a property of the engine.
ENGINE_VARIANT = "arc"
VARIANTS = ("arc", "base")
STEM = {"arc": "stable-audio-3-medium-ARC", "base": "stable-audio-3-medium-RF"}


def checkpoint(variant: str = "arc") -> Path:
    """The base weights. Needed to bake an adapter's row norms and by the verifiers."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    p = CKPT_DIR / f"{STEM[variant]}.safetensors"
    if not p.exists():
        raise FileNotFoundError(
            f"no {variant} checkpoint at {p}\n"
            f"    set $SA3_CKPT_DIR to the directory holding {STEM[variant]}.safetensors")
    return p


def config(variant: str = "arc") -> Path:
    p = CKPT_DIR / f"{STEM[variant]}.json"
    if not p.exists():
        raise FileNotFoundError(f"no {variant} config at {p}; set $SA3_CKPT_DIR")
    return p


def svd_bases() -> Path:
    """Frozen SVD bases -- required ONLY by `-xs` adapters, which are trained against them."""
    if not SVD_BASES.exists():
        raise FileNotFoundError(
            f"no SVD bases at {SVD_BASES}\n"
            f"    -xs adapters are trained against a FROZEN basis and cannot be folded "
            f"without it; set $SA3_SVD_BASES. Plain lora / dora-rows adapters do not need it.")
    return SVD_BASES
