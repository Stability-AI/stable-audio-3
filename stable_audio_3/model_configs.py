from typing import NamedTuple

from huggingface_hub import hf_hub_download


class ModelConfig(NamedTuple):
    repo_id: str
    config_path: str
    ckpt_path: str

    def resolve(self):
        """Download files from HuggingFace Hub and return local cached paths."""
        local_config = hf_hub_download(repo_id=self.repo_id, filename=self.config_path)
        local_ckpt = hf_hub_download(repo_id=self.repo_id, filename=self.ckpt_path)
        return local_config, local_ckpt


rf_models: dict[str, ModelConfig] = {
    "small-rf": ModelConfig(
        "stabilityai/stable-audio-3-small",
        "stable-audio-3-small-RF.json",
        "stable-audio-3-small-RF.safetensors",
    ),
    "medium-rf": ModelConfig(
        "stabilityai/stable-audio-3-medium",
        "stable-audio-3-medium-RF.json",
        "stable-audio-3-medium-RF.safetensors",
    ),
}

arc_models: dict[str, ModelConfig] = {
    "small": ModelConfig(
        "stabilityai/stable-audio-3-small",
        "stable-audio-3-small-ARC.json",
        "stable-audio-3-small-ARC.safetensors",
    ),
    "medium": ModelConfig(
        "stabilityai/stable-audio-3-medium",
        "stable-audio-3-medium-ARC.json",
        "stable-audio-3-medium-ARC.safetensors",
    ),
}

all_models: dict[str, ModelConfig] = {**rf_models, **arc_models}
