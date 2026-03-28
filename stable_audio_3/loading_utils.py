import torch
from safetensors.torch import load_file

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
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape and not any(ignored_key in key for ignored_key in ignored_params):
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        else:
            print(f"Key {key} not found in target state_dict or shape mismatch. Skipping.")

    model.load_state_dict(model_state_dict, strict=False)

def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    
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
                candidate = ".".join(parts[:i]) + "." + ".".join(parts[i+1:])
                if candidate in model_state_dict:
                    key = candidate
                    break
        remapped[key] = value
    return remapped