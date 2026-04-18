from typing import NamedTuple


class ModelConfig(NamedTuple):
    config_path: str
    ckpt_path: str


rf_models: dict[str, ModelConfig] = {
    "small-rf": ModelConfig("SA3_s_rf_inpaint.json", "sa3-longer-rf.ckpt"),
    "medium-rf": ModelConfig("SA3-M-RF.json", "SA3-M-RF.ckpt"),
}

arc_models: dict[str, ModelConfig] = {
    "small": ModelConfig("SA3-S-ARC-CLAP.json", "SA3-S-ARC-CLAP-5k.ckpt"),
    "medium": ModelConfig("SA3-M-ARC-CLAP.json", "SA3-M-ARC-50k.ckpt"),
}

all_models: dict[str, ModelConfig] = {**rf_models, **arc_models}
