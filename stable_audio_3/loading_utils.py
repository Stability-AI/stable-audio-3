import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from stable_audio_3.factory import (
    create_autoencoder_from_config,
    create_diffusion_cond_from_config,
)


def copy_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    state_dict = remap_state_dict_keys(state_dict, model_state_dict)
    for key in state_dict:
        if (
            key in model_state_dict
            and state_dict[key].shape == model_state_dict[key].shape
        ):
            model_state_dict[key] = state_dict[key]
        else:
            print(
                f"Key {key} not found in target state_dict or shape mismatch. Skipping."
            )

    model.load_state_dict(model_state_dict, strict=False)


def load_autoencoder(config_path: str, ckpt_path: str, device: str = "cpu"):
    """Load only the autoencoder from a combined DiT+autoencoder checkpoint.

    Only pretransform tensors are read from disk, directly onto the target device.
    Standalone AE-only checkpoints (e.g. stabilityai/SAME-L / SAME-S) have no prefix.
    """

    with open(config_path) as f:
        config = json.load(f)

    autoencoder = create_autoencoder_from_config(config["model"], config["sample_rate"])

    # Full DiT checkpoints nest the AE under pretransform.model.*;
    # standalone AE-only checkpoints have no prefix.
    nested_prefix = "pretransform.model."
    with safe_open(ckpt_path, framework="pt", device=device) as f:
        all_keys = list(f.keys())
    if any(k.startswith(nested_prefix) for k in all_keys):
        effective_prefix = nested_prefix
    else:
        effective_prefix = ""  # standalone AE — keys are already bare
    with safe_open(ckpt_path, framework="pt", device=device) as f:
        state_dict = {
            k[len(effective_prefix) :]: f.get_tensor(k)
            for k in all_keys
            if k.startswith(effective_prefix)
        }

    copy_state_dict(autoencoder, state_dict)
    return autoencoder.to(device)


def load_diffusion_cond(
    model_config,
    ckpt_path: str,
    device: str = "cuda",
    model_half: bool = False,
):
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    if model_half:
        model.to(torch.float16)
    return model


def remap_state_dict_keys(state_dict, model_state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key not in model_state_dict:
            parts = key.split(".")
            for i in range(1, len(parts)):
                candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1 :])
                if candidate in model_state_dict:
                    key = candidate
                    break
        remapped[key] = value
    return remapped
