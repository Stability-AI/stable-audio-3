"""Locating, downloading and preparing the DirectML-ready ONNX graphs."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import onnx

from . import surgery
from .specs import (
    AUDIO_CHANNELS,
    CODECS,
    MODELS,
    ONNX_REPO,
    T5_FILES,
    TOKENIZER_FILE,
    CodecSpec,
    ModelSpec,
    checkpoint_key,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DIT_VERSION = 2
DECODER_VERSION = 4
ENCODER_VERSION = 1
GRAPH_NAME = "model.onnx"
STAMP_NAME = "prepared.json"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


ENCODER_INPUT = "audio"
ENCODER_SAMPLES = "N"


class AssetError(RuntimeError):
    pass


def default_cache_dir() -> Path:
    return Path(os.environ.get("SA3_DML_CACHE") or PACKAGE_ROOT / "models")


@dataclass
class AssetConfig:
    cache_dir: Path
    onnx_dir: Path | None = None
    checkpoint: Path | None = None
    offline: bool = False
    revision: str | None = None
    use_checkpoint: bool = True


class Assets:
    def __init__(self, config: AssetConfig, log: Callable[[str], None] = print):
        self.config = config
        self.log = log
        self._checkpoints: dict[str, Path | None] = {}

    def fetch(self, repo_file: str, repo_id: str = ONNX_REPO) -> Path:
        """Path to a file of an official repo, from --onnx-dir or the Hugging Face cache."""
        if repo_id == ONNX_REPO and self.config.onnx_dir is not None:
            relative = repo_file[len("onnx/"):] if repo_file.startswith("onnx/") else repo_file
            path = Path(self.config.onnx_dir) / relative
            if path.is_file():
                return path
            if self.config.offline:
                raise AssetError(f"{path} not found")
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        try:
            return Path(hf_hub_download(repo_id, repo_file, revision=self.config.revision,
                                        local_files_only=True))
        except LocalEntryNotFoundError as exc:
            if self.config.offline:
                raise AssetError(f"{repo_id}/{repo_file} is not cached locally and offline mode is on") from exc
        self.log(f"downloading {repo_id}/{repo_file}")
        try:
            return Path(hf_hub_download(repo_id, repo_file, revision=self.config.revision))
        except HfHubHTTPError as exc:
            raise AssetError(f"cannot download {repo_id}/{repo_file}: {exc}. Gated Stability AI "
                             "repositories require accepting the license and `hf auth login`.") from exc

    def checkpoint(self, model: ModelSpec) -> Path | None:
        """Local safetensors checkpoint for a model, if one is configured or cached.

        A configured folder that holds only other models' checkpoints falls back, with a notice, to
        the Hugging Face cache and then to the weights of the official ONNX graph. Any other
        configured path without a matching checkpoint is an error.
        """
        if not self.config.use_checkpoint:
            return None
        if model.name not in self._checkpoints:
            self._checkpoints[model.name] = self._resolve_checkpoint(model)
        return self._checkpoints[model.name]

    def _resolve_checkpoint(self, model: ModelSpec) -> Path | None:
        root = self.config.checkpoint or os.environ.get("SA3_CHECKPOINT_DIR")
        if root:
            root = Path(root)
            if not root.exists():
                raise AssetError(f"checkpoint path {root} does not exist")
            found = _find_checkpoint(root, model)
            if found is not None:
                _check_identity(found, model)
                return found
            if not _holds_other_models(root, model):
                raise AssetError(f"no {model.name} checkpoint (model.safetensors) under {root}")
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model.repo_id, "model.safetensors")
        found = Path(cached) if isinstance(cached, str) else None
        if root:
            source = found or "the weights of the official ONNX graph"
            self.log(f"no {model.name} checkpoint under {root}: using {source}")
        return found

    def tokenizer(self, model: ModelSpec) -> Path:
        ckpt = self.checkpoint(model)
        if ckpt is not None:
            local = ckpt.parent / TOKENIZER_FILE
            if local.is_file():
                return local
        return self.fetch(TOKENIZER_FILE, model.repo_id)

    def text_encoder(self) -> Path:
        return self.fetch(T5_FILES[0])

    def dit(self, model: ModelSpec) -> Path:
        ckpt = self.checkpoint(model)
        files = [model.dit_files[0]] if ckpt is not None else list(model.dit_files)
        return self._prepared(f"dit-{model.name}", DIT_VERSION, files, ckpt, surgery.add_cross_attn_scale)

    def decoder(self, codec: CodecSpec | str) -> Path:
        codec = CODECS[codec] if isinstance(codec, str) else codec

        def transform(m: onnx.ModelProto) -> None:
            surgery.replace_swa_attention(m)
            surgery.remove_random_normal(m)
            surgery.zero_padded_frames(m)
            if not surgery.add_token_noise(m):
                raise AssetError(f"{codec.name} decoder graph has no learned upsampling tokens")
            surgery.expose_float_audio(m)

        return self._prepared(f"decoder-{codec.name}", DECODER_VERSION, list(codec.decoder_files), None, transform)

    def encoder(self, codec: CodecSpec | str) -> Path:
        codec = CODECS[codec] if isinstance(codec, str) else codec

        def transform(m: onnx.ModelProto) -> None:
            surgery.replace_swa_attention(m)
            surgery.name_input_dims(m, ENCODER_INPUT, [1, AUDIO_CHANNELS, ENCODER_SAMPLES])
            if codec.encoder_token_noise and not surgery.add_token_noise(m, tokens_per_frame=1):
                raise AssetError(f"{codec.name} encoder graph has no learned downsampling tokens")

        return self._prepared(f"encoder-{codec.name}", ENCODER_VERSION, list(codec.encoder_files), None, transform)

    def _prepared(self, key: str, version: int, files: list[str], ckpt: Path | None,
                  transform: Callable[[onnx.ModelProto], None]) -> Path:
        target = Path(self.config.cache_dir) / key
        stamp = {
            "version": version,
            "sources": files,
            "revision": self.config.revision,
            "weights": _file_identity(ckpt) if ckpt is not None else "onnx",
        }
        graph = target / GRAPH_NAME
        if graph.is_file() and _read_stamp(target) == stamp:
            return graph

        paths = [self.fetch(f) for f in files]
        self.log(f"preparing {key} for DirectML" + (f" (weights: {ckpt})" if ckpt else ""))
        bound = ckpt is not None
        proto = onnx.load(str(paths[0]), load_external_data=not bound)
        if bound:
            _bind_checkpoint(proto, ckpt)
        transform(proto)

        tmp = target.with_name(target.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        onnx.save_model(proto, str(tmp / GRAPH_NAME), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=GRAPH_NAME + ".data", size_threshold=1024)
        (tmp / STAMP_NAME).write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        return graph


def _folder_names(model: ModelSpec) -> list[str]:
    return [model.name, model.repo_id.split("/")[-1]]


def _find_checkpoint(root: Path, model: ModelSpec) -> Path | None:
    if root.is_file():
        return root
    for candidate in [root, *(root / n for n in _folder_names(model))]:
        path = candidate / "model.safetensors"
        if path.is_file():
            return path
    return None


def _holds_other_models(root: Path, model: ModelSpec) -> bool:
    """True for a folder of per-model checkpoints that simply has none for `model`."""
    if not root.is_dir() or any((root / n).exists() for n in _folder_names(model)):
        return False
    return any((root / n / "model.safetensors").is_file()
               for other in MODELS.values() if other.name != model.name for n in _folder_names(other))


def _check_identity(ckpt: Path, model: ModelSpec) -> None:
    """Reject a checkpoint whose model_config.json names a different model."""
    config = ckpt.parent / "model_config.json"
    if not config.is_file():
        return
    data = json.loads(config.read_text(encoding="utf-8"))
    repos = {c.get("config", {}).get("repo_id")
             for c in data.get("model", {}).get("conditioning", {}).get("configs", [])}
    repos.discard(None)
    if repos and model.repo_id not in repos:
        raise AssetError(f"{ckpt} belongs to {sorted(repos)}, not {model.repo_id}")


def _bind_checkpoint(proto: onnx.ModelProto, ckpt: Path) -> None:
    from safetensors import safe_open

    with safe_open(str(ckpt), framework="np") as f:
        keys = set(f.keys())
        pairs = [(init, checkpoint_key(init.name)) for init in proto.graph.initializer]
        missing = [key for _, key in pairs if key is not None and key not in keys]
        if missing:
            raise AssetError(f"{ckpt} lacks {len(missing)} DiT tensors, e.g. {missing[:3]}")
        for init, key in pairs:
            if key is not None:
                surgery.set_initializer(init, f.get_tensor(key))
    unbound = [i.name for i in proto.graph.initializer if i.data_location == onnx.TensorProto.EXTERNAL]
    if unbound:
        raise AssetError(f"graph tensors without a checkpoint source: {unbound[:3]}")


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime": int(stat.st_mtime)}


def _read_stamp(target: Path) -> dict | None:
    try:
        return json.loads((target / STAMP_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
