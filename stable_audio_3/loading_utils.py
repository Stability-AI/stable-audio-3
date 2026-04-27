import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from stable_audio_3.model import (
    create_autoencoder_from_config,
    create_diffusion_cond_from_config,
)


def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    state_dict = remap_state_dict_keys(state_dict, model_state_dict)
    ignored_params = []
    for key in state_dict:
        if (
            key in model_state_dict
            and state_dict[key].shape == model_state_dict[key].shape
            and not any(ignored_key in key for ignored_key in ignored_params)
        ):
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        else:
            print(
                f"Key {key} not found in target state_dict or shape mismatch. Skipping."
            )

    model.load_state_dict(model_state_dict, strict=False)


def load_autoencoder(config_path: str, ckpt_path: str, device: str = "cpu"):
    """Load only the autoencoder from a combined DiT+autoencoder checkpoint.

    For .safetensors checkpoints, only pretransform tensors are read from disk,
    directly onto the target device.
    For .ckpt checkpoints, the full state dict is loaded but the DiT is never instantiated.
    """

    with open(config_path) as f:
        config = json.load(f)

    autoencoder = create_autoencoder_from_config(config["model"], config["sample_rate"])

    # ARC checkpoints store autoencoder weights under pretransform.model.*,
    # RF checkpoints use pretransform.* directly, and standalone AE-only checkpoints
    # (e.g. from stabilityai/SAME-L / SAME-S) have no prefix at all.
    prefix = "pretransform."
    arc_prefix = "pretransform.model."
    if ckpt_path.endswith(".safetensors"):
        with safe_open(ckpt_path, framework="pt", device=device) as f:
            all_keys = list(f.keys())
        if any(k.startswith(arc_prefix) for k in all_keys):
            effective_prefix = arc_prefix
        elif any(k.startswith(prefix) for k in all_keys):
            effective_prefix = prefix
        else:
            effective_prefix = (
                ""  # standalone AE checkpoint — all keys belong to the AE
            )
        with safe_open(ckpt_path, framework="pt", device=device) as f:
            state_dict = {
                k[len(effective_prefix) :]: f.get_tensor(k)
                for k in all_keys
                if k.startswith(effective_prefix)
            }
    else:
        full = torch.load(ckpt_path, map_location=device, weights_only=True)[
            "state_dict"
        ]
        if any(k.startswith(arc_prefix) for k in full):
            effective_prefix = arc_prefix
        elif any(k.startswith(prefix) for k in full):
            effective_prefix = prefix
        else:
            effective_prefix = ""
        state_dict = {
            k[len(effective_prefix) :]: v
            for k, v in full.items()
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
    state_dict = load_ckpt_state_dict(ckpt_path)
    copy_state_dict(model, state_dict)
    model.to(device).eval().requires_grad_(False)
    if model_half:
        model.to(torch.float16)
    return model


def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]

    return state_dict


def remap_state_dict_keys(state_dict, model_state_dict):
    """Remap state_dict keys to match model_state_dict keys.

    Handles cases where checkpoint keys have extra nesting (e.g. pretransform.model.* -> pretransform.*).
    """
    remapped = {}
    for key, value in state_dict.items():
        if key not in model_state_dict:
            # Try stripping one level of nesting from each prefix segment
            parts = key.split(".")
            for i in range(1, len(parts)):
                candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1 :])
                if candidate in model_state_dict:
                    key = candidate
                    break
        remapped[key] = value
    return remapped
