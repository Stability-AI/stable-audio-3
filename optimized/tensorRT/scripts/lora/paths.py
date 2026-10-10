"""Where this install keeps its model files, per DiT.

Three things need locating and none of them belong in source: the TensorRT engine and its
branch map (built locally -- TensorRT bakes the GPU architecture in, so they cannot ship),
the base checkpoint (only the verification harnesses need it), and the frozen SVD bases
(only `-xs` adapters need those).

Everything resolves relative to this checkout, overridable by environment variable, and a
missing file raises where it is asked for rather than resolving to the wrong model. That
last part is the point: an earlier version let one constant silently resolve to a different
model's SVD bases, which folds an -xs adapter into a rotated basis with nothing raising.

Three DiTs are served, and they are DIFFERENT models -- 24 layers x 1536 for medium, 20 x
1024 for the two small ones. An adapter trained on one does not apply to another and an
engine built for one cannot be driven with another's map, so every path here is keyed by
model. `for_model(name)` is the accessor; the module-level constants are sa3-medium, kept
because they are the published names this package's own scripts already import.
"""
import os
import sys
from pathlib import Path

# optimized/tensorRT/ -- this file sits at scripts/lora/paths.py
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# The LoRA merge math (all 8 adapter variants) already ships in this repo for the TFLite
# path. It is the product's ground truth for what an adapter means, so it is imported, not
# forked -- two copies of that file would be two definitions of "merged weight".
_LORA_CORE = Path(__file__).resolve().parents[3] / "tflite" / "scripts"
if str(_LORA_CORE) not in sys.path:
    sys.path.insert(0, str(_LORA_CORE))


# Built locally by build_branch.py, and placed where every other engine in this install
# lives: models/<arch>/<model>/. TensorRT bakes the GPU architecture into the plan, so the
# arch is part of the path -- one install can hold several side by side.
def _arch() -> str:
    """sm_<major><minor> of the current device, matching sa3_trt_core.ARCH."""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"
    except Exception:
        return os.environ.get("SA3_ARCH", "sm_90")


ARCH = _arch()
MODELS_DIR = Path(os.environ.get("SA3_MODELS_DIR", ROOT / "models"))

DEFAULT_MODEL = "sa3-medium"

# One row per DiT. `slug` is the directory name used throughout this repo and on the HF
# model repo (models/<arch>/<slug>/, onnx/<slug>/); `env` suffixes the override variables
# for the non-default models, so SA3_CKPT_DIR still means medium and never silently
# retargets. `stems` are the published checkpoint basenames where there are any -- the
# small models publish as model.safetensors, which the resolver finds by shape instead.
MODELS = {
    "sa3-medium": dict(
        slug="sa3-m", dit="medium", env=None, layers=24, embed_dim=1536, n_targets=229,
        map_name="branch_map_medium_lora.json", ckpt_dir="sa3-medium",
        stems={"arc": "stable-audio-3-medium-ARC", "base": "stable-audio-3-medium-RF"},
        hf_repo="stabilityai/stable-audio-3-medium",
    ),
    "sa3-sm-music": dict(
        slug="sa3-sm-music", dit="sm-music", env="SM_MUSIC", layers=20, embed_dim=1024,
        n_targets=193,
        map_name="branch_map_sm_music_lora.json", ckpt_dir="sa3-sm-music",
        stems={}, hf_repo="stabilityai/stable-audio-3-small-music",
    ),
    "sa3-sm-sfx": dict(
        slug="sa3-sm-sfx", dit="sm-sfx", env="SM_SFX", layers=20, embed_dim=1024,
        n_targets=193,
        map_name="branch_map_sm_sfx_lora.json", ckpt_dir="sa3-sm-sfx",
        stems={}, hf_repo="stabilityai/stable-audio-3-small-sfx",
    ),
}
# Every model here is ARC: the published ONNX in onnx/<slug>/ is traced from the distilled
# variant. Folding an adapter's row norms against the other variant reads as a 27% fold
# error on medium, so this is not a preference -- it is a property of the engine.
ENGINE_VARIANT = "arc"
VARIANTS = ("arc", "base")

# Aliases, so --model sa3-m, --dit medium and model="sa3-medium" all land on the same row.
# The slug is what the directories are called, `dit` is what sa3_trt.py's --dit calls it,
# and the long name is what the runtime calls the model. Three spellings already existed in
# this package; resolving them in one place is cheaper than three lookup tables.
ALIASES = {spec["slug"]: name for name, spec in MODELS.items()}
ALIASES.update({spec["dit"]: name for name, spec in MODELS.items()})
ALIASES.update({name: name for name in MODELS})


def canonical(model: str) -> str:
    """'sa3-m' / 'sa3-medium' -> 'sa3-medium'. Raises on anything not served."""
    try:
        return ALIASES[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; this build serves "
            f"{', '.join(sorted(MODELS))} (slugs {', '.join(sorted(m['slug'] for m in MODELS.values()))})"
        ) from None


def _env(spec, base, default):
    """SA3_<base> for the default model, SA3_<base>_<SUFFIX> for the others."""
    key = f"SA3_{base}" if spec["env"] is None else f"SA3_{base}_{spec['env']}"
    v = os.environ.get(key)
    return Path(v) if v else default


# Candidate basenames for a checkpoint and its config, in order. The medium tree names its
# files after the model; the published small repos ship model.safetensors + model_config.json;
# a hand-assembled tree may just symlink `ckpt`/`config`. All three are real layouts in use,
# and guessing wrong here is the one failure that produces plausible numbers from the wrong
# weights, so the resolver tries them explicitly instead of formatting a name.
_CKPT_NAMES = ("{stem}.safetensors", "{stem}.ckpt", "model.safetensors", "ckpt")
_CONFIG_NAMES = ("{stem}.json", "model_config.json", "config")


class ModelPaths:
    """Every path that depends on which DiT you mean."""

    def __init__(self, name):
        self.name = canonical(name)
        spec = MODELS[self.name]
        self.spec = spec
        self.slug = spec["slug"]
        self.dit = spec["dit"]
        self.layers = spec["layers"]
        self.n_targets = spec["n_targets"]
        self.engine_variant = ENGINE_VARIANT
        self.engine_dir = _env(spec, "ENGINE_DIR", MODELS_DIR / ARCH / self.slug)
        self.branch_engine = self.engine_dir / "dit_fp16_lora.trt"
        # Two maps, and the distinction matters. SHIPPED is tracked: it is the target-set
        # definition, a property of the ONNX graph, the same on every machine. BUILT is
        # written beside the engine by build_branch and carries build provenance --
        # absolute ONNX path, GPU, hostname. models/ is gitignored, so that stays out of
        # the repo; writing it over the tracked file would dirty every builder's tree and
        # invite them to commit their own paths.
        self.shipped_map = HERE / spec["map_name"]
        self.built_map = self.engine_dir / spec["map_name"]
        self.branch_map = _env(
            spec, "BRANCH_MAP",
            self.built_map if self.built_map.exists() else self.shipped_map)
        # Downloaded: stabilityai/stable-audio-3-optimized, onnx/<slug>/dit_fp16.onnx.
        self.onnx_dir = _env(spec, "ONNX_DIR", ROOT / "onnx" / self.slug)
        self.onnx = self.onnx_dir / "dit_fp16.onnx"
        # Only the verify_* harnesses need these; normal adapter loading never touches them.
        self.ckpt_dir = _env(spec, "CKPT_DIR", MODELS_DIR / spec["ckpt_dir"])
        self.svd_bases = _env(spec, "SVD_BASES", self.ckpt_dir / "svd_bases.pt")

    def __repr__(self):
        return f"<ModelPaths {self.name} engine={self.branch_engine}>"

    def _variant_dirs(self, variant):
        """<ckpt_dir>/<variant>/ first, then <ckpt_dir>/ -- both are layouts in use."""
        return (self.ckpt_dir / variant, self.ckpt_dir)

    def env_var(self, base: str) -> str:
        """The environment variable that overrides `base` for THIS model."""
        return f"SA3_{base}" if self.spec["env"] is None else f"SA3_{base}_{self.spec['env']}"

    def _find(self, variant, names, what):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        stem = self.spec["stems"].get(variant)
        tried = []
        for d in self._variant_dirs(variant):
            for pat in names:
                if "{stem}" in pat and not stem:
                    continue
                p = d / pat.format(stem=stem)
                tried.append(p)
                if p.exists():
                    return p
        looked = ", ".join(dict.fromkeys(p.name for p in tried))
        raise FileNotFoundError(
            f"no {variant} {what} for {self.name}\n"
            f"    looked for [{looked}] under {self.ckpt_dir} and {self.ckpt_dir / variant}\n"
            f"    set ${self.env_var('CKPT_DIR')} to a directory holding it\n"
            f"    the published weights are at {self.spec['hf_repo']}")

    def checkpoint(self, variant: str = ENGINE_VARIANT) -> Path:
        """The base weights. Needed to bake an adapter's row norms and by the verifiers."""
        return self._find(variant, _CKPT_NAMES, "checkpoint")

    def config(self, variant: str = ENGINE_VARIANT) -> Path:
        return self._find(variant, _CONFIG_NAMES, "config")

    def bases(self) -> Path:
        """Frozen SVD bases -- required ONLY by `-xs` adapters, trained against them."""
        if not self.svd_bases.exists():
            raise FileNotFoundError(
                f"no SVD bases at {self.svd_bases}\n"
                f"    -xs adapters are trained against a FROZEN basis and cannot be folded "
                f"without it. Plain lora / dora-rows adapters do not need it.")
        return self.svd_bases


def branch_map_candidates():
    """Every branch map this install could hold, most specific first.

    The tracked maps live beside this file; the maps `build_branch.py` writes live beside
    their engine under models/<arch>/<slug>/, and for a model with no tracked map that
    built one is the ONLY copy. A finder that globs just one of the two places reports
    "no map matches" for an engine sitting right there.
    """
    out = []
    for name in MODELS:
        mp = for_model(name)
        out += [mp.built_map, mp.shipped_map]
    out += sorted(HERE.glob("branch_map*.json"))
    seen, uniq = set(), []
    for q in out:
        if q not in seen and q.exists():
            seen.add(q); uniq.append(q)
    return uniq


_CACHE = {}


def for_model(model: str = DEFAULT_MODEL) -> ModelPaths:
    """The paths for one DiT. Cached, so repeated lookups are free."""
    name = canonical(model)
    if name not in _CACHE:
        _CACHE[name] = ModelPaths(name)
    return _CACHE[name]


# ---------------------------------------------------------------------------
# sa3-medium at module level. These are the names the rest of this package imports;
# they were here before the other two models were served and they still mean medium.
# ---------------------------------------------------------------------------
_M = for_model(DEFAULT_MODEL)
ENGINE_DIR = _M.engine_dir
BRANCH_ENGINE = _M.branch_engine
SHIPPED_MAP = _M.shipped_map
BUILT_MAP = _M.built_map
BRANCH_MAP = _M.branch_map
ONNX_DIR = _M.onnx_dir
ONNX = _M.onnx
CKPT_DIR = _M.ckpt_dir
SVD_BASES = _M.svd_bases
STEM = dict(MODELS[DEFAULT_MODEL]["stems"])


def checkpoint(variant: str = "arc") -> Path:
    """sa3-medium base weights. Other models: for_model(m).checkpoint(variant)."""
    return _M.checkpoint(variant)


def config(variant: str = "arc") -> Path:
    return _M.config(variant)


def svd_bases() -> Path:
    """sa3-medium frozen SVD bases. Other models: for_model(m).bases()."""
    return _M.bases()
