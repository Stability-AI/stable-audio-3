"""Model registry and pipeline constants, mirroring the released Stable Audio 3 configs."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
LATENT_DIM = 256
DOWNSAMPLING_RATIO = 4096
TEXT_TOKENS = 256
DURATION_PADDING_SEC = 6.0
LATENT_ALIGN = 2

ONNX_REPO = "stabilityai/stable-audio-3-optimized"
TOKENIZER_FILE = "t5gemma-b-b-ul2/tokenizer.json"


@dataclass(frozen=True)
class CodecSpec:
    """SAME autoencoder. `token_noise` / `encoder_token_noise` are the reference `mask_noise` of
    the decoder and encoder; `frame_align` is the latent granularity of its attention chunks."""

    name: str
    decoder_files: tuple[str, ...]
    encoder_files: tuple[str, ...]
    token_noise: float
    encoder_token_noise: float
    frame_align: int = 1
    chunk: int = 128
    overlap: int = 32


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    dit_files: tuple[str, ...]
    codec: str
    sample_size: int

    @property
    def max_seconds(self) -> float:
        return self.sample_size / SAMPLE_RATE


CODECS: dict[str, CodecSpec] = {
    "same-l": CodecSpec("same-l", ("onnx/same-l/dec_fp16.onnx",), ("onnx/same-l/enc_fp16.onnx",),
                        token_noise=0.1, encoder_token_noise=0.001),
    "same-s": CodecSpec("same-s", ("onnx/same-s/dec_bf16.onnx",), ("onnx/same-s/enc_bf16.onnx",),
                        token_noise=0.01, encoder_token_noise=0.0, frame_align=2),
}

MODELS: dict[str, ModelSpec] = {
    "medium": ModelSpec(
        "medium", "stabilityai/stable-audio-3-medium",
        ("onnx/sa3-m/dit_fp16.onnx", "onnx/sa3-m/dit_fp16.onnx.data"), "same-l", 16777216),
    "small-sfx": ModelSpec(
        "small-sfx", "stabilityai/stable-audio-3-small-sfx",
        ("onnx/sa3-sm-sfx/dit_fp16.onnx",), "same-s", 5292032),
    "small-music": ModelSpec(
        "small-music", "stabilityai/stable-audio-3-small-music",
        ("onnx/sa3-sm-music/dit_fp16.onnx",), "same-s", 5292032),
}

T5_FILES = ("onnx/t5gemma/encoder.onnx",)

CHECKPOINT_ALIASES = {
    "dit.": "model.model.",
    "padding_embedding": "conditioner.conditioners.prompt.padding_embedding",
    "seconds_weight": "conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
    "seconds_bias": "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
}


def checkpoint_key(onnx_name: str) -> str | None:
    """Checkpoint tensor name for a DiT graph initializer, or None if it is graph-only."""
    if onnx_name in CHECKPOINT_ALIASES:
        return CHECKPOINT_ALIASES[onnx_name]
    if onnx_name.startswith("dit."):
        return CHECKPOINT_ALIASES["dit."] + onnx_name[len("dit."):]
    return None
