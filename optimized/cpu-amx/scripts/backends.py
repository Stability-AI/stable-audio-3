"""Backend loaders for the cpu-amx release.

Every heavy component is a torch-free C++ AMX engine, loaded by inserting its
source directory on sys.path and importing the ctypes shim (the .so + weights
live in those directories; we do not copy them). The a2a/inpaint init-encoder is
now ALSO a torch-free C++ AMX engine (SAME-S / SAME-L, matched to the decoder),
so the whole release is C++/numpy — no torch anywhere. (The legacy fp32 torch AE
remains as load_ae_encoder_torch() for reference/fallback only.)

Components
    T5Gemma    : t5gemma_cpu_amx.so  (bf16 AMX)          -> [1,256,768]
    DiT        : dit_cpu_amx.so       (int8 AMX, MEDIUM)  cond_backend(x,t,cross,gcond)->v
    decoders   : same_{s,l}_cpu_amx.so (bf16, default) or *_int8fused (int8, --decoder-precision int8)
    encoder    : same_{s,l}_encoder_cpu_amx.so (bf16 AMX) audio[1,2,N] -> latent[1,256,T]  (a2a/inpaint)

DiT stability note: the dit .so heap-corrupts if called at more than one
sequence length in a process, and double-frees at teardown. The CLI runs one
generation (one T_lat) per process and os._exit(0)s before teardown; the gradio
isolates each generation in its own subprocess. DiT threads default to 1 (the
verified-stable setting; it is fast enough for short clips).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ── C++ backend source directories (import .so via sys.path; do not copy) ────
DIR_T5 = "/weka2/cj/clod/t5gemma_cpu_amx"
DIR_DIT = "/weka2/cj/clod/q4/sa3-w4-cluster"          # cpu_amx_backend.py (DiTCppAmx)
DIR_SAMES = "/weka2/cj/clod/same_s_cpu_amx"
DIR_SAMEL = "/weka2/cj/clod/same_l_cpu_amx"
DIR_SAMES_INT8 = "/weka2/cj/clod/same_s_int8fused_cpu_amx"
DIR_SAMEL_INT8 = "/weka2/cj/clod/same_l_int8fused_cpu_amx"
DIR_SAMES_ENC = "/weka2/cj/clod/same_s_encoder_cpu_amx"   # C++ AMX SAME-S encoder (bf16)
DIR_SAMEL_ENC = "/weka2/cj/clod/same_l_encoder_cpu_amx"   # C++ AMX SAME-L encoder (bf16)
DIR_SAMES_ENC_INT8 = "/weka2/cj/clod/same_s_encoder_int8fused_cpu_amx"   # C++ AMX SAME-S encoder (int8)
DIR_SAMEL_ENC_INT8 = "/weka2/cj/clod/same_l_encoder_int8fused_cpu_amx"   # C++ AMX SAME-L encoder (int8)
DIR_AE = "/weka2/cj/clod/sa3s/fast_load"              # samel_loader.load_model (torch AE, fallback only)

DIT_THREADS = 1   # verified-stable; the .so heap-races at higher thread counts


def _add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


try:
    import weights as _hfw            # pulls each engine's .so + weight blob from HF on first use
except Exception:
    _hfw = None

def _ensure(group):
    """Download an engine's binaries from HF if not present locally (no-op on a local build)."""
    if _hfw is not None:
        try:
            _hfw.ensure(group)
        except Exception as e:
            print(f"[cpu-amx] HF fetch '{group}' failed ({e}); assuming local build", flush=True)


# ── T5Gemma text encoder ────────────────────────────────────────────────────
def load_t5gemma(threads=16):
    """Returns a callable enc(ids[1,256] int, mask[1,256] int) -> [1,256,768] fp32."""
    _ensure("t5gemma")
    _add_path(DIR_T5)
    from t5gemma_cpu_backend import T5GemmaCPU
    return T5GemmaCPU(threads=int(threads))


# ── DiT (medium int8) — the cond_backend for sample()/sample_cfg ────────────
def load_dit(threads=DIT_THREADS):
    """Returns a DiTCppAmx: __call__(x[1,256,T], t, cross[1,257,768], gcond[1,768]) -> v[1,256,T]."""
    _ensure("dit")
    _add_path(DIR_DIT)
    from cpu_amx_backend import DiTCppAmx
    return DiTCppAmx(threads=int(threads))


# ── Decoders (bf16 default, int8 optional) ──────────────────────────────────
class Decoder:
    """Wraps a C++ SAME-S / SAME-L engine as decode(latent[1,256,T]) -> audio[2,N].

    Even-length requirement (PAD_MODULO=2): SAME-S (both precisions) AND the
    SAME-L *int8-fused* engine require an even latent length — an odd T is
    edge-padded by one column, decoded, and the extra 4096 samples trimmed.
    SAME-L *bf16* takes any length (its band attention is linear; forward_pcm
    chunks C=64/overlap=8)."""

    def __init__(self, name: str, precision: str = "bf16", threads: int = 16):
        assert name in ("same-s", "same-l"), name
        assert precision in ("bf16", "int8"), precision
        self.name, self.precision = name, precision
        self.is_sames = name == "same-s"
        # SAME-S (any prec) and SAME-L int8 assert even T; SAME-L bf16 does not.
        self.needs_even = self.is_sames or (name == "same-l" and precision == "int8")
        if name == "same-s" and precision == "bf16":
            _ensure("same_s_decoder_bf16"); _add_path(DIR_SAMES)
            from same_s_cpu_backend import SamesCPU
            self.m = SamesCPU(threads=threads)
        elif name == "same-s" and precision == "int8":
            _ensure("same_s_decoder_int8"); _add_path(DIR_SAMES_INT8)
            from same_s_int8fused_backend import SamesInt8FusedCPU
            self.m = SamesInt8FusedCPU(threads=threads)
        elif name == "same-l" and precision == "bf16":
            _ensure("same_l_decoder_bf16"); _add_path(DIR_SAMEL)
            from same_l_cpu_backend import SamelCPU
            self.m = SamelCPU(threads=threads)
        else:  # same-l int8
            _ensure("same_l_decoder_int8"); _add_path(DIR_SAMEL_INT8)
            from same_l_int8fused_backend import SamelInt8FusedCPU
            self.m = SamelInt8FusedCPU(threads=threads)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """latent (1,256,T) fp32 -> audio (2, T*4096) fp32."""
        latent = np.ascontiguousarray(latent, np.float32)
        T = latent.shape[-1]
        if self.needs_even and (T % 2 == 1):
            padded = np.concatenate([latent, latent[..., -1:]], axis=-1)   # -> even
            pcm = self.m.forward_pcm(padded)[0]                            # (2,(T+1)*4096)
            return np.ascontiguousarray(pcm[:, : T * SAMPLES_PER_LATENT])
        return self.m.forward_pcm(latent)[0]                              # (2, T*4096)


from pipeline import SAMPLES_PER_LATENT  # noqa: E402  (after Decoder so import is cheap)


def load_decoder(name: str, precision: str = "bf16", threads: int = 16) -> Decoder:
    return Decoder(name, precision, threads)


# ── Audio-to-audio / inpaint init-encoder (torch-free C++ AMX SAME-{S,L} encoder) ──
class CppEncoder:
    """Torch-free C++ AMX-BF16 autoencoder ENCODER for a2a / inpaint. Matches the
    chosen decoder (same-s decoder -> SAME-S encoder, same-l -> SAME-L encoder), so
    encode/decode share the same autoencoder. numpy + ctypes only — no torch, no 2 GB
    checkpoint load.

    encode(audio (2,N) fp32, T_lat) -> latent (1,256,T_lat) fp32.
    The encoder downsamples 4096 audio samples per latent token, so the audio is
    trimmed/zero-padded to exactly T_lat*4096 samples before encoding."""

    def __init__(self, name: str, precision: str = "bf16", threads: int = 16):
        assert name in ("same-s", "same-l"), name
        assert precision in ("bf16", "int8"), precision
        self.name = name; self.precision = precision
        self.device = "cpu-amx"
        if name == "same-s" and precision == "bf16":
            _ensure("same_s_encoder_bf16"); _add_path(DIR_SAMES_ENC)
            from same_s_encoder_backend import SamesEncoderCPU
            self.m = SamesEncoderCPU(threads=threads)
        elif name == "same-s":
            _ensure("same_s_encoder_int8"); _add_path(DIR_SAMES_ENC_INT8)
            from same_s_encoder_int8fused_backend import SameSEncoderInt8FusedCPU
            self.m = SameSEncoderInt8FusedCPU(threads=threads)
        elif name == "same-l" and precision == "bf16":
            _ensure("same_l_encoder_bf16"); _add_path(DIR_SAMEL_ENC)
            from same_l_encoder_backend import SamelEncoderCPU
            self.m = SamelEncoderCPU(threads=threads)
        else:
            _ensure("same_l_encoder_int8"); _add_path(DIR_SAMEL_ENC_INT8)
            from same_l_encoder_int8fused_backend import SameLEncoderInt8FusedCPU
            self.m = SameLEncoderInt8FusedCPU(threads=threads)

    def encode(self, audio: np.ndarray, T_lat: int) -> np.ndarray:
        return np.ascontiguousarray(self.m.encode(audio, T_lat), np.float32)


def load_encoder(name: str = "same-l", precision: str = "bf16", threads: int = 16) -> CppEncoder:
    """C++ AMX encoder matching the decoder; precision follows --decoder-precision. name in {'same-s','same-l'}."""
    return CppEncoder(name, precision=precision, threads=threads)


# ── (fallback) fp32 torch AE encoder — retained for reference; not used by the release ──
def _pick_free_gpu() -> str | None:
    """Return the index (as str) of the CUDA device with the most free memory,
    or None if CUDA/nvidia-smi is unavailable (then the AE runs on CPU)."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        free = [int(x) for x in out.stdout.split()]
        if not free:
            return None
        return str(int(np.argmax(free)))
    except Exception:
        return None


class AEEncoder:
    """fp32 SAME-L autoencoder ENCODER for a2a / inpaint. Torch — imported lazily.

    encode(audio (2,N) fp32, T_lat) -> latent (1,256,T_lat) fp32.
    The AE downsamples 4096 audio samples per latent token, so the audio is
    trimmed/zero-padded to exactly T_lat*4096 samples before encoding."""

    def __init__(self, device: str | None = None):
        gpu = _pick_free_gpu()
        if device is None and gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu   # must precede torch import
        _add_path(DIR_AE)
        import torch
        from samel_loader import load_model
        import stable_audio_tools.models.autoencoders as ae
        self.torch = torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        model = load_model(dev)
        # inference-clean bottleneck: no stochastic noise regularisation / masking
        model.bottleneck.noise_regularize = False
        for m in model.modules():
            if isinstance(m, ae.TransformerResamplingBlock):
                m.mask_noise = 0.0
        self.model = model.eval()

    def encode(self, audio: np.ndarray, T_lat: int) -> np.ndarray:
        torch = self.torch
        target = T_lat * SAMPLES_PER_LATENT
        a = np.ascontiguousarray(audio, np.float32)
        if a.shape[-1] >= target:
            a = a[:, :target]
        else:
            a = np.pad(a, ((0, 0), (0, target - a.shape[-1])))
        with torch.no_grad():
            x = torch.from_numpy(a[None]).to(self.device)                 # (1,2,N)
            lat = self.model.encode(x)                                    # (1,256,T_lat)
        return np.ascontiguousarray(lat.float().cpu().numpy(), np.float32)


def load_ae_encoder_torch(device: str | None = None) -> AEEncoder:
    """Legacy fp32 torch AE encoder (fallback/reference; the release uses load_encoder)."""
    return AEEncoder(device)
