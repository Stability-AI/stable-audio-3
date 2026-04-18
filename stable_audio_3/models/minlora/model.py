"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""

import math
import re
from functools import partial
from itertools import product

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class LoRAParametrization(nn.Module):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=1,
                adapter_type="lora", W0=None, lora_index=0
                ):
        super().__init__()
        dtype = W0.dtype if W0 is not None else torch.get_default_dtype()
        device = W0.device if W0 is not None else None
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=dtype, device=device))
        self.register_buffer("lora_strength", torch.tensor(1.0, dtype=dtype, device=device), persistent=False)
        self.adapter_type = adapter_type
        self.lora_index = lora_index

        if self.adapter_type == "lora":
            self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in)), dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank)), dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.forward_fn = self.lora_forward
            self._adapter_forward_fn = self.forward_fn

        elif self.adapter_type == "dora":
            # low-rank factors (same as LoRA)
            self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in)), dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank)), dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # magnitude: one value per column, initialized from pretrained weight norms
            if W0 is not None:
                col_norms = W0.view(W0.shape[0], -1).norm(dim=0, keepdim=True)
                self.magnitude = nn.Parameter(col_norms.to(dtype))
            else:
                self.magnitude = nn.Parameter(torch.ones(self.swap((1, fan_in)), dtype=dtype, device=device))
            self.forward_fn = self.dora_forward
            self._adapter_forward_fn = self.forward_fn

        elif self.adapter_type == "lora-xs":
            # frozen bases U:(fan_out,r), V:(fan_in,r); tiny trainable core M:(r,r)
            self.register_buffer("U", torch.empty(self.swap((fan_out, self.rank)), device=device), persistent=True)
            self.register_buffer("V", torch.empty(self.swap((fan_in,  self.rank)), device=device), persistent=True)
            self.M_xs = nn.Parameter(torch.zeros(self.rank, self.rank, dtype=dtype, device=device))
            if W0 is not None:
                W0_2d = W0.view(W0.shape[0], -1).float()
                U_full, S, Vh_full = torch.linalg.svd(W0_2d, full_matrices=False)
                self.U.copy_(U_full[:, :self.rank].to(dtype))
                self.V.copy_(Vh_full[:self.rank, :].T.to(dtype))
            self.forward_fn = self.lora_xs_forward
            self._adapter_forward_fn = self.forward_fn


    def lora_forward(self, W):
        delta = torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(W.shape)
        return W + (self.scaling * self.lora_strength * delta).to(W.dtype)

    def dora_forward(self, W):
        # work in 2D so norm/magnitude ops don't broadcast badly for Conv1d/Conv2d
        orig_shape = W.shape
        W_2d = W.view(W.shape[0], -1)
        # low-rank update on the *direction*
        delta = torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A))))
        V = W_2d + (self.scaling * self.lora_strength * delta).to(W_2d.dtype)
        # normalize columns (unit direction), then scale by per-column magnitude
        V_hat = V / (V.norm(dim=0, keepdim=True) + 1e-12)
        return (V_hat * self.magnitude.to(V_hat.dtype)).view(orig_shape)

    def lora_xs_forward(self, W):
        delta = (self.U.float() @ self.M_xs @ self.V.T.float()).view(W.shape)
        return W + (self.scaling * self.lora_strength.float() * delta).to(W.dtype)


    def forward(self, X):
        return self.forward_fn(X)

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self._adapter_forward_fn

    @classmethod
    def _get_original_weight(cls, layer, attr_name="weight"):
        """Get the true original weight, even when the layer is already parametrized."""
        if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, attr_name):
            return getattr(layer.parametrizations, attr_name).original.detach()
        return getattr(layer, attr_name).detach()

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, **kwargs):
        fan_out, fan_in = layer.weight.shape
        W0 = cls._get_original_weight(layer)
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha,
            W0=W0, **kwargs
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, **kwargs):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        W0 = cls._get_original_weight(layer).view(layer.weight.shape[0], -1)
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha,
            W0=W0, **kwargs
        )

    @classmethod
    def from_conv1d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, **kwargs):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        W0 = cls._get_original_weight(layer).view(layer.weight.shape[0], -1)
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha,
            W0=W0, **kwargs
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, **kwargs):
        fan_in, fan_out = layer.weight.shape
        W0 = cls._get_original_weight(layer).view(layer.weight.shape[0], -1)
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha,
            W0=W0, **kwargs
        )


default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4),
    },
}


# --- layer name filtering helpers ---

def _expand(s):
    """Expand bracket notation in layer filter strings, e.g. 'layers[0-5]' -> ['layers0', ..., 'layers5']."""
    t = re.split(r'\[(\d+)-(\d+)\]', s)
    if len(t) == 1:
        return [s]
    lits, starts, ends = t[0::3], t[1::3], t[2::3]
    pools = []
    for a, b in zip(starts, ends):
        ai, bi = int(a), int(b)
        step = 1 if bi >= ai else -1
        pools.append([str(n) for n in range(ai, bi + step, step)])
    out = []
    for combo in product(*pools):
        pieces = []
        for i, lit in enumerate(lits):
            pieces.append(lit)
            if i < len(combo):
                pieces.append(combo[i])
        out.append(''.join(pieces))
    return out


def _matches_any(name, patterns):
    """Check if name contains any pattern substring (with bracket expansion)."""
    if not patterns:
        return False
    for pattern in patterns:
        for expanded in _expand(pattern.strip()):
            if expanded in name:
                return True
    return False


# --- core LoRA application ---

def _match_layer_type(layer, lora_config):
    """Find the matching lora_config key for a layer, using isinstance to handle ParametrizedLinear etc."""
    for layer_type in lora_config:
        if isinstance(layer, layer_type):
            return layer_type
    return None


def apply_lora(layer, register=True, merge=False, lora_config=default_lora_config):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        matched_type = _match_layer_type(layer, lora_config)
        if matched_type is not None:
            for attr_name, parametrization in lora_config[matched_type].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_lora(model, lora_config=default_lora_config, include=None, exclude=None):
    """Add LoRA parametrization to layers in a model.

    Args:
        model: The model to add LoRA to.
        lora_config: Dict mapping nn.Module types to parametrization configs.
        include: Optional list of substrings. If provided, only modules whose
                 name contains at least one pattern get LoRA. Supports bracket
                 expansion (e.g. "layers[0-11]").
        exclude: Optional list of substrings. Modules matching any pattern are
                 skipped even if they match an include pattern.
    """
    if include is None and exclude is None:
        # Fast path: original behavior, no name inspection needed
        model.apply(partial(apply_lora, lora_config=lora_config))
    else:
        applied = []
        skipped = []
        for name, module in model.named_modules():
            matched_type = _match_layer_type(module, lora_config)
            if matched_type is None:
                continue

            if include is not None and not _matches_any(name, include):
                skipped.append(name)
                continue

            if exclude is not None and _matches_any(name, exclude):
                skipped.append(name)
                continue
            # Apply LoRA to this module
            for attr_name, parametrization_fn in lora_config[matched_type].items():
                parametrize.register_parametrization(module, attr_name, parametrization_fn(module))
            applied.append(name)

def add_lora_by_name(model, target_module_names, lora_config=default_lora_config):
    """Add LoRA to specific layers by name. Convenience wrapper around add_lora()."""
    add_lora(model, lora_config=lora_config, include=target_module_names)


def merge_lora(model):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=True))

def remove_lora(model):
    """remove lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=False))

def set_lora_strength(model, strength: float, lora_index=None):
    """Set lora strength. If lora_index is None, sets all LoRAs. If specified, sets only that index."""
    strength = float(strength)
    if lora_index is None:
        for name, buf in model.named_buffers(recurse=True):
            if name.endswith("lora_strength"):
                buf.fill_(strength)
    else:
        for p in _iter_lora_params(model):
            if p.lora_index == lora_index:
                p.lora_strength.fill_(strength)

def _iter_lora_params(model):
    for _, mod in model.named_modules():
        plist = getattr(getattr(mod, "parametrizations", None), "weight", None)
        if plist is None:
            continue
        for p in plist:
            if isinstance(p, LoRAParametrization):
                yield p
