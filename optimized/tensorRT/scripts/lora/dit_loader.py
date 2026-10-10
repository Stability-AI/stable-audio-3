"""Load the SA3-medium DiT with its conditioning baked in, and the pingpong sampler.

The same cond-baked wrapper the shipped ONNX was traced from, so a torch forward through it
is structurally identical to an engine forward -- which is what makes the verify_* harnesses
a real comparison rather than an approximate one. Nothing here is dimension-specific: the
shapes come from the config file named by `paths`.
"""
import os, json, math, time
import torch, torch.nn as nn

import paths

# One place decides which weights these are. The variant matters: the published ONNX is
# built from ARC, and folding the other variant's weights into an ARC-built engine is wrong
# with nothing raising -- it just reads as a large fold error.
MODEL_DIR = str(paths.CKPT_DIR)
CKPT = str(paths.CKPT_DIR / (paths.STEM[paths.ENGINE_VARIANT] + ".ckpt"))
CONFIG = str(paths.config(paths.ENGINE_VARIANT))

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
IO_CHANNELS = 256
T5_TOKENS = 256
T5_HIDDEN_DIM = 768
LOCAL_ADD_COND_DIM = 257

NUM_COND_MIN_VAL = 0
NUM_COND_MAX_VAL = 384
NUM_COND_FOURIER_DIM = 256
NUM_COND_MIN_FREQ = 0.5
NUM_COND_MAX_FREQ = 10000.0

COND_KEYS = {
    "padding_embedding": "conditioner.conditioners.prompt.padding_embedding",
    "seconds_weight": "conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
    "seconds_bias": "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
}


def patch_for_onnx():
    """Disable flash-attn (no ONNX symbolic) and make RMSNorm export-friendly.

    Same patch the medium producer uses. Only for tracing / TRT-reference runs —
    the true eager path keeps flash_attn.
    """
    import stable_audio_tools.models.transformer as satt
    satt.flash_attn_func = None
    satt.flash_attn_kvpacked_func = None
    satt.flash_attn_varlen_func = None

    def patched_rms(self, x):
        d = x.dtype
        if self.force_fp32:
            x = x.float()
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        g = self.gamma.float() if self.force_fp32 else self.gamma
        return (g * x).to(d) if self.force_fp32 else g * x

    satt.RMSNorm.forward = patched_rms


def load_state_dict():
    # A checkpoint may arrive as .ckpt or .safetensors -- same flat state dict either way.
    if str(CKPT).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(CKPT))
    else:
        sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        for prefix in ("diffusion_ema.", "diffusion."):
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
                break
        else:
            out[k] = v
    return out


def load_model(device="cuda", dtype=torch.float32, verbose=True):
    """Full model: DiT + autoencoder pretransform + T5Gemma conditioner."""
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict

    with open(CONFIG) as f:
        cfg = json.load(f)
    t0 = time.time()
    model = create_model_from_config(cfg)
    sd = load_state_dict()
    copy_state_dict(model, sd)
    model = model.to(dtype).to(device).eval()
    if verbose:
        n_dit = sum(p.numel() for p in model.model.model.parameters())
        print(f"[load] {time.time()-t0:.1f}s  DiT={n_dit/1e6:.0f}M params", flush=True)
    return model, cfg, sd


class DiTWithCond(nn.Module):
    """DiT with the conditioner's cheap tail baked in (padding-embedding substitution +
    seconds_total Fourier→Linear), so the exported engine takes raw T5 hidden states.

    Inputs: x[1,256,L], t[1], t5_hidden[1,256,768], t5_mask[1,256], seconds_total[1],
            local_add_cond[1,257,L]  ->  velocity[1,256,L]
    """

    def __init__(self, dit, ct):
        super().__init__()
        self.dit = dit
        self.register_buffer("padding_embedding", ct["padding_embedding"].float().contiguous())
        half = NUM_COND_FOURIER_DIM // 2
        ramp = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
        freqs = torch.exp(ramp * (math.log(NUM_COND_MAX_FREQ) - math.log(NUM_COND_MIN_FREQ))
                          + math.log(NUM_COND_MIN_FREQ))
        freqs = freqs * (2.0 * math.pi)
        self.register_buffer("seconds_freqs", freqs.contiguous())
        self.register_buffer("seconds_weight", ct["seconds_weight"].float().contiguous())
        self.register_buffer("seconds_bias", ct["seconds_bias"].float().contiguous())

    def _apply_padding(self, t5_hidden, t5_mask):
        pe = self.padding_embedding.view(1, 1, -1)
        m = t5_mask.unsqueeze(-1).to(torch.bool)
        pe = pe.to(t5_hidden.dtype)
        return torch.where(m, t5_hidden, pe)

    def _seconds_embed(self, seconds_total, dtype):
        s = seconds_total.clamp(NUM_COND_MIN_VAL, NUM_COND_MAX_VAL).float()
        s = (s - NUM_COND_MIN_VAL) / (NUM_COND_MAX_VAL - NUM_COND_MIN_VAL)
        angles = s.unsqueeze(-1) * self.seconds_freqs
        ff = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)
        y = torch.nn.functional.linear(ff, self.seconds_weight.to(dtype), self.seconds_bias.to(dtype))
        return y.unsqueeze(1)

    def forward(self, x, t, t5_hidden, t5_mask, seconds_total, local_add_cond):
        t5_padded = self._apply_padding(t5_hidden, t5_mask)
        sec_emb = self._seconds_embed(seconds_total, t5_padded.dtype)
        cross_attn_cond = torch.cat([t5_padded, sec_emb], dim=1)
        global_cond = sec_emb.squeeze(1)
        return self.dit._forward(x, t, cross_attn_cond=cross_attn_cond,
                                 global_embed=global_cond, local_add_cond=local_add_cond)


def make_wrapper(model, sd, device="cuda"):
    ct = {name: sd[key].clone() for name, key in COND_KEYS.items()}
    return DiTWithCond(model.model.model, ct).to(device).eval()


def seconds_to_latent_len(seconds):
    return max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))


def build_sigmas(steps, latent_len, device="cuda", sigma_max=1.0):
    """Pingpong schedule with the SA3 FluxDistributionShift warp (runtime.DistributionShift)."""
    base_shift, max_shift, min_length, max_length = 0.5, 1.15, 256, 4096
    sl = min(max(int(latent_len), min_length), max_length)
    mu = -(base_shift + (max_shift - base_shift) * (sl - min_length) / (max_length - min_length))
    t = torch.linspace(sigma_max, 0.0, steps + 1, device=device)
    sig = 1 - math.exp(mu) / (math.exp(mu) + (1.0 / (1.0 - t) - 1.0))
    sig[0] = sigma_max
    sig[-1] = 0.0
    return sig


def sample_pingpong(fwd, x, sigmas, seed=0, on_step=None):
    """rf_denoiser pingpong sampler (identical to scripts/pt_inference.py)."""
    steps = len(sigmas) - 1
    g_loop = torch.Generator(device=x.device)
    g_loop.manual_seed(int(seed) + 1)
    with torch.no_grad():
        for i in range(steps):
            t_curr, t_next = sigmas[i], sigmas[i + 1]
            v = fwd(x, t_curr.reshape(1).contiguous())
            denoised = x - t_curr * v
            if i < steps - 1:
                noise = torch.randn(*x.shape, device=x.device, dtype=x.dtype, generator=g_loop)
                x = (1.0 - t_next) * denoised + t_next * noise
            else:
                x = denoised
            if on_step is not None:
                on_step(i, x)
    return x


def save_wav(path, audio, sample_rate=SAMPLE_RATE):
    """audio: torch [1,2,T] float or numpy (T,2) int16."""
    import numpy as np, wave
    if isinstance(audio, torch.Tensor):
        a = audio.detach().float().clamp(-1, 1).mul(32767.0).to(torch.int16)
        a = a.squeeze(0).T.contiguous().cpu().numpy()
    else:
        a = audio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(a.astype(np.int16).tobytes())
    return path
