from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from torchaudio.transforms import Resample, MelSpectrogram
from torch.nn.utils import weight_norm
import scipy.signal

from .wavelets import WaveletEncode1d, WaveletDecode1d
from .blocks import ResidualUnit, WNConv1d

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
    
    def tokenize(self, x):
        raise NotImplementedError
    
    def decode_tokens(self, tokens):
        raise NotImplementedError

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False, enable_grad = False):
        super().__init__(enable_grad=enable_grad, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        if not enable_grad:
            self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()
    
    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]
    
    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pack_complex_to_channels(U: torch.Tensor) -> torch.Tensor:
    """Pack complex U[B,C,F,M] → real z[B,2·C·F,M] as [Re, Im] along channels."""
    B, C, F, M = U.shape
    Z = torch.empty((B, 2 * C * F, M), dtype=U.real.dtype, device=U.device)
    Z[:, 0::2, :] = U.real.reshape(B, C * F, M)
    Z[:, 1::2, :] = U.imag.reshape(B, C * F, M)
    return Z


def _unpack_channels_to_complex(Z: torch.Tensor, C: int, F: int) -> torch.Tensor:
    """Inverse of _pack_complex_to_channels. Z[B,2·C·F,M] → U[B,C,F,M] (complex)."""
    B, CF2, M = Z.shape
    assert CF2 == 2 * C * F, f"Z has {CF2} channels but expected {2*C*F}"
    Zre = Z[:, 0::2, :]
    Zim = Z[:, 1::2, :]
    return torch.complex(Zre, Zim).reshape(B, C, F, M)


def _sine_window(N: int, device, dtype) -> torch.Tensor:
    n = torch.arange(N, device=device, dtype=dtype)
    return torch.sin(math.pi * (n + 0.5) / N)


def _onesided_tight_weights(n_fft: int, device, dtype) -> torch.Tensor:
    """Return [1,1,F,1] weights: interior bins √2, DC/Nyquist 1 (for one-sided energy)."""
    F = n_fft // 2 + 1
    w = torch.ones(F, dtype=dtype, device=device)
    if (n_fft % 2) == 0:
        if F > 2:
            w[1:F-1] = math.sqrt(2.0)
    else:
        if F > 1:
            w[1:] = math.sqrt(2.0)
    return w.view(1, 1, F, 1)

def _demod_sign(F: int, M: int, device, dtype, expand_bc: bool = True) -> torch.Tensor:
    """Parity demod for hop=N/2.
    Returns ±1 with shape [1,1,F,M] (if expand_bc) for unambiguous broadcast
    against X[...,F,M]. Numerically exact for hop=N/2.
    """
    k_odd = (torch.arange(F, device=device, dtype=torch.int8) & 1).view(F, 1)
    m_odd = (torch.arange(M, device=device, dtype=torch.int8) & 1).view(1, M)
    parity = (k_odd & m_odd).to(torch.float32)
    sign = (1.0 - 2.0 * parity).to(dtype) # {0,1} -> {+1,-1}
    return sign.view(1, 1, F, M) if expand_bc else sign

def _to_mid_side(x: torch.Tensor) -> torch.Tensor:
    """x[B,2,T] → [B,2,T] with orthonormal mid/side."""
    s2 = math.sqrt(0.5)
    L, R = x[:, 0:1, :], x[:, 1:2, :]
    M = (L + R) * s2
    S = (L - R) * s2
    return torch.cat([M, S], dim=1)


def _from_mid_side(x: torch.Tensor) -> torch.Tensor:
    s2 = math.sqrt(0.5)
    M, S = x[:, 0:1, :], x[:, 1:2, :]
    L = (M + S) * s2
    R = (M - S) * s2
    return torch.cat([L, R], dim=1)


# -----------------------------------------------------------------------------
# Pretransform
# -----------------------------------------------------------------------------
class ComplexSTFTPretransform(Pretransform):  
    def __init__(
        self,
        channels: int,
        n_fft: int = 1024,
        demodulate: bool = True,
        center: bool = False,
        value_norm: str = "tight",   # 'tight' | 'none'
        use_mid_side: bool = False,
        ema_flatten: bool = True,
        flatten_alpha: float = 0.5,
        ema_beta: float = 1e-4,
        w_min: float = 0.25,
        w_max: float = 4.0,
        use_compander: bool = True,
        comp_alpha: float = 0.9,
        beta_min: float = 0.25,
        beta_max: float = 4.0,
        eps: float = 1e-12,
        enable_grad: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gl_correction_steps: int = 0,  # NEW
        freeze_stats: bool = False,  # NEW
    ):
        super().__init__(enable_grad=enable_grad, io_channels=channels, is_discrete=False)
        self.C = int(channels)
        self.n_fft = int(n_fft)
        self.win_length = int(n_fft)
        self.hop_length = self.win_length // 2
        self.demodulate = bool(demodulate)
        self.center = bool(center)
        self.value_norm = value_norm
        self.use_mid_side = bool(use_mid_side)
        self.ema_flatten = bool(ema_flatten)
        self.flatten_alpha = float(flatten_alpha)
        self.ema_beta = float(ema_beta)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.use_compander = bool(use_compander)
        self.comp_alpha = float(comp_alpha)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.gl_correction_steps = int(max(0, gl_correction_steps))  # NEW

        _device = device if device is not None else torch.device("cpu")
        _dtype = dtype if dtype is not None else torch.float32

        # Fixed tight configuration: sine window, hop = n_fft//2
        win = _sine_window(self.win_length, _device, _dtype)
        self.register_buffer("window", win, persistent=False)

        if value_norm not in ("tight", "none"):
            raise ValueError("value_norm must be 'tight' or 'none'")

        # Bin weights for one-sided tightness
        w_one = _onesided_tight_weights(self.n_fft, _device, _dtype)
        self.register_buffer("_onesided_w", w_one, persistent=False)  # [1,1,F,1]

        # Derived sizes
        self.F = self.n_fft // 2 + 1
        self.downsampling_ratio = self.hop_length
        self.encoded_channels = self.C * (2 * self.F)

        # EMA stats buffers (shared across channels): shapes [1,1,F,1]
        self.register_buffer("psd", torch.ones(1,1,self.F,1, dtype=torch.float32, device=_device))        # E[|U|^2]
        self.register_buffer("s2a", torch.ones(1,1,self.F,1, dtype=torch.float32, device=_device))        # E[|Û|^{2α}] after flatten

        # Cache last decode length
        self._last_length: Optional[int] = None
        self.freeze_stats = bool(freeze_stats)

    # ---------------- STFT wrappers (unitary FFT) ----------------
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = fold_channels_into_batch(x)  # [B·C,T]
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=True,       # unitary FFT
            return_complex=True,
        )  # [B,F,M]
        X = unfold_channels_from_batch(X, C)
        return X  # [B,C,F,M]

    def _istft(self, X: torch.Tensor, length: Optional[int]) -> torch.Tensor:
        B, C, F, M = X.shape
        X = fold_channels_into_batch(X)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=True,       # unitary FFT
            length=length,
            return_complex=False,
        )  # [B,T]
        x = unfold_channels_from_batch(x, C)
        return x  # [B,C,T]

    # ---------------- EMA helpers ----------------
    def _compute_W(self) -> Optional[torch.Tensor]:
        if not self.ema_flatten or self.flatten_alpha == 0.0:
            return None
        W = (self.psd + self.eps) ** (-0.5 * self.flatten_alpha)
        return torch.clamp(W, self.w_min, self.w_max)

    def _compute_Beta(self) -> Optional[torch.Tensor]:
        if not self.use_compander:
            return None
        Beta = (self.s2a + self.eps) ** (-0.5)
        return torch.clamp(Beta, self.beta_min, self.beta_max)

    # ---------------- NEW: GL post-correction helper ----------------
    def _gl_k_steps_time(self, X0: torch.Tensor, length: int, steps: int) -> torch.Tensor:
        """
        Run `steps` iterations of fixed-magnitude Griffin–Lim in RAW STFT.
        Keeps the STFT frame grid identical to X0 on all intermediate projections.
        Returns the final time-domain signal of length `length`.

        X0: [B,C,F,M] complex STFT (raw, i.e., after demod inverse etc.)
        steps: int >= 1
        """
        assert steps >= 1
        B, C, F, M = X0.shape
        hop = self.hop_length

        # Choose ISTFT length that guarantees STFT(...).shape[-1] == M
        if self.center:
            T_grid = (M - 1) * hop
        else:
            T_grid = (M - 1) * hop + self.win_length

        V = X0.abs()   # target magnitudes [B,C,F,M]
        X = X0         # current complex STFT on the same grid [B,C,F,M]

        for _ in range(steps):
            # Unit phasor (robust at zeros)
            ph = X / X.abs().clamp_min(self.eps)
            Z = V * ph
            # Project Z to consistency on the SAME frame grid
            x_mid = self._istft(Z, length=T_grid)
            X = self._stft(x_mid)  # guaranteed [B,C,F,M] by construction

        # Final synthesis to requested output length
        x_hat = self._istft(X, length=length)
        return x_hat

    # ---------------- API ----------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16:
            x = x.float()  # ensure float for fft when using bf16
        assert x.dim() == 3 and x.size(1) == self.C
        self._last_length = x.size(-1)

        # Optional mid–side (time domain)
        if self.use_mid_side and self.C == 2:
            x = _to_mid_side(x)

        X = self._stft(x)  # [B,C,F,M]
        if self.demodulate:
            rot = _demod_sign(self.F, X.size(-1), X.device, X.real.dtype, expand_bc=True)
            U = X * rot  # demod (unitary)
        else:
            U = X

        if self.value_norm == "tight":
            U = U * self._onesided_w  # one-sided weighting

        # --- EMA flattening (linear) ---
        if self.ema_flatten:
            if self.training and not self.freeze_stats:
                # FP32 stats to avoid fp16 under/overflow in pow
                with torch.cuda.amp.autocast(enabled=False):
                    P_hat = U.detach().abs().float().pow(2).mean(dim=(0,1,3), keepdim=True)  # [1,1,F,1]
                self.psd = (1.0 - self.ema_beta) * self.psd + self.ema_beta * P_hat
            W = self._compute_W()
            if W is not None:
                U = U * W
        else:
            W = None

        # --- Compander (radial power-law) ---
        if self.use_compander:
            # Update stats on **flattened** coefficients (FP32)
            if self.training and not self.freeze_stats:
                with torch.cuda.amp.autocast(enabled=False):
                    S_hat = U.detach().abs().float().pow(2 * self.comp_alpha).mean(dim=(0,1,3), keepdim=True)
                self.s2a = (1.0 - self.ema_beta) * self.s2a + self.ema_beta * S_hat

            Beta = self._compute_Beta()  # stays FP32 via buffers

            r = U.abs()
            # r^alpha in FP32, then cast back to real dtype for multiply
            with torch.cuda.amp.autocast(enabled=False):
                r_alpha = r.float().pow(self.comp_alpha)
            r_alpha = r_alpha.to(r.dtype)

            p = torch.where(r > 0, U / r.clamp_min(self.eps), torch.zeros_like(U))

            U = (Beta * r_alpha) * p

        return _pack_complex_to_channels(U)  # [B,2·C·F,M]


    def decode(self, z: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        B, EncC, M = z.shape
        T_len = length if length is not None else self._last_length
        if T_len is None:
            raise ValueError("Decode length unknown. Pass length= or call encode() first.")

        if z.dtype == torch.bfloat16:
            z = z.float()  # ensure float for fft when using bf16

        U = _unpack_channels_to_complex(z, C=self.C, F=self.F)

        # Inverse compander
        if self.use_compander:
            Beta = self._compute_Beta()  # FP32 via buffers
            r_prime = U.abs()

            p = torch.where(r_prime > 0, U / r_prime.clamp_min(self.eps), torch.zeros_like(U))

            # (r'/β)^(1/α) in FP32 to avoid fp16 blow-ups
            with torch.cuda.amp.autocast(enabled=False):
                rrec = torch.clamp(r_prime.float() / Beta, min=0.0).pow(1.0 / self.comp_alpha)
            rrec = rrec.to(r_prime.dtype)

            U = rrec * p

        # Inverse flattening
        if self.ema_flatten:
            W = self._compute_W()
            if W is not None:
                U = U / W

        if self.value_norm == "tight":
            U = U / self._onesided_w

        if self.demodulate:
            rot = _demod_sign(self.F, M, U.device, U.real.dtype, expand_bc=True)
            X = U * rot  # RAW STFT (re-modulate)
        else:
            X = U

        # -------- Optional GL post-correction (fixed magnitudes, k steps) --------
        steps = self.gl_correction_steps
        if steps > 0:
            x_hat = self._gl_k_steps_time(X, length=T_len, steps=steps)
        else:
            x_hat = self._istft(X, length=T_len)

        # Inverse mid–side
        if self.use_mid_side and self.C == 2:
            x_hat = _from_mid_side(x_hat)
        return x_hat


class WaveletPretransform(Pretransform):
    def __init__(self, channels, levels, wavelet = "bior4.4", enable_grad = False, **kwargs):
        super().__init__(enable_grad=False, io_channels=channels, is_discrete=False)

        self.encoder = WaveletEncode1d(channels, levels, wavelet)
        self.decoder = WaveletDecode1d(channels, levels, wavelet)

        self.downsampling_ratio = 2 ** levels
        self.io_channels = channels
        self.encoded_channels = channels * self.downsampling_ratio

    def encode(self, x):
        x = self.encoder(x) 
        return x
    
    def decode(self, z):
        return self.decoder(z)

class Ortho1x1Conv1d(nn.Module):
    def __init__(self, C, n_reflect=4, learn_scale=False, eps=0.1):
        super().__init__()
        self.C = C
        self.v = nn.ParameterList([nn.Parameter(torch.randn(C)) for _ in range(n_reflect)])
        self.learn_scale = learn_scale
        if learn_scale:
            self.d = nn.Parameter(torch.zeros(C))  # small diagonal scale
            self.eps = eps

    def _apply_householders(self, X):  # X: [B,C,T], returns Q @ X
        for v in self.v:
            v_ = v / (v.norm(p=2) + 1e-8)               # [C]
            proj = (v_[None,:,None] * X).sum(1, keepdim=True)  # [B,1,T]
            X = X - 2.0 * v_[None,:,None] * proj         # (I - 2 vv^T) X
        return X

    def _scale(self, Y):
        if not self.learn_scale:
            return Y
        s = torch.exp(self.eps * torch.tanh(self.d)).view(1, -1, 1)
        return s * Y

    def forward(self, x):
        qx = self._apply_householders(x)
        y = self._scale(qx)
        return y

    def inverse(self, y):
        x = y
        if self.learn_scale:
            s = torch.exp(self.eps * torch.tanh(self.d)).view(1, -1, 1)
            x = x / (s + 1e-8)
        # Householder inverse = apply same reflections in reverse
        for v in reversed(self.v):
            v_ = v / (v.norm(p=2) + 1e-8)
            proj = (v_[None,:,None] * x).sum(1, keepdim=True)
            x = x - 2.0 * v_[None,:,None] * proj
        return x


class HouseholderPretransform(Pretransform):
    """
    Encoder: Stack of [pad → S2D(s) → invertible 1x1] per stage.
    Decoder: Exact inverse stack [inv 1x1 → D2S(s) → crop].
    The product of strides determines the patch rate.
    """
    def __init__(self, channels, patch_size, strides, postfilter_channels = 0,enable_grad = True,**kwargs):
        super().__init__(enable_grad=enable_grad, io_channels=channels, is_discrete=False)

        self.channels = channels
        self.patch_size = patch_size
        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size

        if postfilter_channels > 0:
            self.postfilter = nn.Sequential(
            WNConv1d(in_channels=channels, out_channels=postfilter_channels, kernel_size=7, padding=3, bias=False),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=1, use_snake=True, bias = False),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=3, use_snake=True, bias = False),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=9, use_snake=True, bias = False),
            WNConv1d(in_channels=postfilter_channels, out_channels=channels, kernel_size=7, padding=3, bias=False))

        if strides is None:
            self.strides = [2] * int(math.log2(patch_size))
        else:
            self.strides = strides

        # Channel tracking
        C = self.channels
        self.mixers = nn.ModuleList()
        self.stage_channels = []  # channels after each S2D
        self.wavelet_encoders = []
        self.wavelet_decoders = []

        for s in self.strides:
            self.wavelet_encoders.append(WaveletEncode1d(C,int(math.log2(s))))
            self.wavelet_decoders.append(WaveletDecode1d(C, int(math.log2(s))))
            C *= s                    # S2D multiplies channels by s
            r = min(C, 16)
            self.stage_channels.append(C)
            self.mixers.append(Ortho1x1Conv1d(C, n_reflect=r, learn_scale=False))
    
        self._T_last = None  # filled on encode
        
        if not enable_grad:
            self.mixers.requires_grad_(False).eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T]
        returns z: [B, C_out, T / total_downsample], with C_out = C_in * Π s_i
        """
        B, C, T0 = x.shape
        assert C == self.channels

        pad = (-T0) % self.downsampling_ratio
        if pad:
            x = torch.nn.functional.pad(x, (0, pad), mode="reflect")  # right-pad once

        z = x
        for s, mix, wavelet_enc in zip(self.strides, self.mixers, self.wavelet_encoders):
            z = wavelet_enc(z)
            #z = space_to_depth_1d(z, s)  # divisible due to global pad
            z = mix(z)

        # remember original length for future decode() calls
        self._T_last = T0
        return z

    def decode(self, z: torch.Tensor, T_target: int = None) -> torch.Tensor:
        """
        z: [B, C_out, T / total_downsample]
        returns x: [B, C_in, T_target] if provided, else reconstructs to last seen length.
        """
        x = z
        for s, mix, wavelet_dec in zip(reversed(self.strides), reversed(self.mixers), reversed(self.wavelet_decoders)):
            x = mix.inverse(x)
            x = wavelet_dec(x)
            #x = depth_to_space_1d(x, s)

        # crop once to the requested length (or last encoded length)
        Tt = self._T_last if T_target is None else T_target
        if Tt is not None and x.shape[-1] != Tt:
            x = x[..., :Tt]

        if hasattr(self, 'postfilter'):
            x = self.postfilter(x)
        return x


# ---------- small utils ----------

def _same_pad_1d(x, k: int, dilation: int = 1, mode: str = "reflect"):
    assert k % 2 == 1, "Use odd kernels for symmetric 'same' padding."
    pad = dilation * (k - 1) // 2
    return torch.nn.functional.pad(x, (pad, pad), mode=mode)

class _ConvSame1d(nn.Module):
    """C->C conv with 'same' padding (odd kernels), optional groups."""
    def __init__(self, C, kernel_size=5, groups=1, bias=False, dilation=1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.ks = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(C, C, kernel_size, groups=groups, bias=bias, dilation=dilation)
        # near-identity init for stable start
        nn.init.zeros_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = _same_pad_1d(x, self.ks, self.dilation)
        return self.conv(x)

# ---------- single lifting stage (linear, PR) ----------

class _LiftingStage1D(nn.Module):
    """
    One dyadic lifting stage over C channels.
    Analysis: split -> predict -> update -> rescale, returns (L, H) both [B,C,T/2].
    Synthesis: exact inverse.
    """
    def __init__(self, C, k_predict=5, k_update=5, groups=1):
        super().__init__()
        self.C = C
        self.P = _ConvSame1d(C, k_predict, groups=groups, bias=False)  # predict
        self.U = _ConvSame1d(C, k_update, groups=groups, bias=False)   # update
        # channelwise rescale; alpha=exp(s), beta=exp(-s) ensures det=1
        self.log_scale = nn.Parameter(torch.zeros(1, C, 1))

    @staticmethod
    def _split_even_odd(x):
        e = x[..., ::2]
        o = x[..., 1::2]
        if o.size(-1) < e.size(-1):  # pad odd to match even length
            o = torch.nn.functional.pad(o, (0, 1))
        return e, o

    @staticmethod
    def _merge_even_odd(e, o):
        B, C, L = e.shape
        y = torch.empty(B, C, 2 * L, device=e.device, dtype=e.dtype)
        y[..., ::2] = e
        y[..., 1::2] = o
        return y

    def analysis(self, x):
        """
        x: [B,C,T] -> L,H: [B,C,T/2 rounded up]
        """
        e, o = self._split_even_odd(x)
        d = o - self.P(e)          # predict
        s = e + self.U(d)          # update
        a = torch.exp(self.log_scale)        # alpha
        b = torch.exp(-self.log_scale)       # beta
        L = a * s
        H = b * d
        return L, H

    def synthesis(self, L, H):
        """
        Exact inverse: (L,H) -> x: [B,C,~2T]
        """
        a = torch.exp(self.log_scale)
        b = torch.exp(-self.log_scale)
        s = L / (a + 1e-12)
        d = H / (b + 1e-12)
        e = s - self.U(d)
        o = d + self.P(e)
        x = self._merge_even_odd(e, o)
        return x

# ---------- packetized J-level lifting encoder/decoder ----------

class LiftingEncode1d(nn.Module):
    """
    J-level lifting 'wavelet packet' encoder.
    Each level halves time and doubles channels via concatenation [L,H] along channels.
    Output: [B, C*2^J, T/2^J].
    """
    def __init__(self, C, J, k_predict=5, k_update=5, groups=1):
        super().__init__()
        assert J >= 1 and int(J) == J
        self.J = int(J)
        stages = []
        C_cur = C
        for _ in range(self.J):
            stages.append(_LiftingStage1D(C_cur, k_predict=k_predict, k_update=k_update, groups=groups))
            C_cur *= 2
        self.stages = nn.ModuleList(stages)  # note: stage i maps C_i -> 2*C_i

    def forward(self, x):
        """
        x: [B,C,T] -> y: [B, C*2^J, T/2^J]
        """
        z = x
        for stage in self.stages:
            L, H = stage.analysis(z)        # both [B, C_i, T/2]
            z = torch.cat([L, H], dim=1)    # [B, 2*C_i, T/2]
        return z

class LiftingDecode1d(nn.Module):
    """
    Exact inverse of LiftingEncode1d sharing the same stages (no param duplication).
    """
    def __init__(self, encode_module: LiftingEncode1d):
        super().__init__()
        # Do NOT register stages again; just keep a reference.
        self.encode = encode_module

    def forward(self, y):
        """
        y: [B, C*2^J, T/2^J] -> x: [B,C,T]
        """
        z = y
        # Walk stages in reverse; at stage i there were C_i channels before doubling.
        for stage in reversed(self.encode.stages):
            C_i = stage.C
            L, H = z[:, :C_i, :], z[:, C_i:2*C_i, :]
            z = stage.synthesis(L, H)       # [B, C_i, 2*T]
        return z

# ---------- Drop-in Pretransform with lifting ----------

class LiftingPretransform(Pretransform):
    """
    Encoder: [pad] -> repeat { LiftingEncode1d(stage) -> Ortho1x1 } per stride stage.
    Decoder: exact inverse { Ortho^{-1} -> LiftingDecode1d(stage) } -> [crop] (then optional postfilter).
    Matches HouseholderPretransform interface and I/O contracts.
    """
    def __init__(self, channels, patch_size, strides, postfilter_channels=0,
                 k_predict=5, k_update=5, groups=1, enable_grad = True, **kwargs):
        super().__init__(enable_grad=enable_grad, io_channels=channels, is_discrete=False)
        self.channels = channels
        self.patch_size = patch_size
        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size

        # Optional postfilter (kept identical to your Householder version)
        if postfilter_channels > 0:
            self.postfilter = nn.Sequential(
                WNConv1d(in_channels=channels, out_channels=postfilter_channels, kernel_size=7, padding=3, bias=False),
                ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels, dilation=1, use_snake=True, bias=False),
                ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels, dilation=3, use_snake=True, bias=False),
                ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels, dilation=9, use_snake=True, bias=False),
                WNConv1d(in_channels=postfilter_channels, out_channels=channels, kernel_size=7, padding=3, bias=False),
            )

        # Stride schedule (same convention)
        if strides is None:
            self.strides = [2] * int(math.log2(patch_size))
        else:
            self.strides = strides

        # Per-stage modules (mirrors your attributes)
        C = self.channels
        self.stage_channels = []       # channels after each encoder stage
        self.mixers = nn.ModuleList()  # Ortho1x1 per stage (kept for parity)
        self.wavelet_encoders = nn.ModuleList()
        self.wavelet_decoders = []     # NOT ModuleList: decoders share encoder params

        for s in self.strides:
            J = int(math.log2(s))
            enc = LiftingEncode1d(C, J=J, k_predict=k_predict, k_update=k_update, groups=groups)
            dec = LiftingDecode1d(enc)   # shares weights; no param duplication
            self.wavelet_encoders.append(enc)
            self.wavelet_decoders.append(dec)
            C *= s
            self.stage_channels.append(C)
            r = min(C, 16)
            self.mixers.append(Ortho1x1Conv1d(C, n_reflect=r, learn_scale=False))

        self._T_last = None  # filled on encode
        if not enable_grad:
            self.mixers.requires_grad_(False).eval()
            self.wavelet_encoders.requires_grad_(False).eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T] -> z: [B, C_in * Π s_i, T / Π s_i]
        """
        B, C, T0 = x.shape
        assert C == self.channels

        pad = (-T0) % self.downsampling_ratio
        if pad:
            x = torch.nn.functional.pad(x, (0, pad), mode="reflect")  # right-pad once

        z = x
        for enc, mix in zip(self.wavelet_encoders, self.mixers):
            z = enc(z)   # lifting packet: halves T, doubles C per level; totals s-fold
            z = mix(z)

        self._T_last = T0
        return z

    def decode(self, z: torch.Tensor, T_target: int = None) -> torch.Tensor:
        """
        z: [B, C_out, T / total_downsample] -> x: [B, C_in, T_target or last T]
        """
        x = z
        for mix, dec in zip(reversed(self.mixers), reversed(self.wavelet_decoders)):
            x = mix.inverse(x)
            x = dec(x)

        Tt = self._T_last if T_target is None else T_target
        if Tt is not None and x.shape[-1] != Tt:
            x = x[..., :Tt]

        if hasattr(self, 'postfilter'):
            x = self.postfilter(x)
        return x

class PatchedPretransform(Pretransform):
    def __init__(self, channels, patch_size, oversampling = 1, postfilter_channels = 0, **kwargs):
        super().__init__(enable_grad=True, io_channels=channels, is_discrete=False)
        self.channels = channels
        self.patch_size = patch_size
        self.oversampling = oversampling

        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size

        if self.oversampling > 1:
            self.input_upsampler = Resample(1, self.oversampling)
            self.output_downsampler = Resample(self.oversampling, 1)

        if postfilter_channels > 0:
            self.postfilter = nn.Sequential(
            WNConv1d(in_channels=channels, out_channels=postfilter_channels, kernel_size=7, padding=3, bias=True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=1, use_snake=True, bias = True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=3, use_snake=True, bias = True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=9, use_snake=True, bias = True),
            WNConv1d(in_channels=postfilter_channels, out_channels=channels, kernel_size=7, padding=3, bias=False))

    def _pad(self, x):
        seq_len = x.shape[-1]
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x
        
    def encode(self, x):
        if self.oversampling > 1:
            x = self.input_upsampler(x)
        x = self._pad(x)
        x = rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)
        return x
    def decode(self, x):
        x = rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)
        if hasattr(self, 'postfilter'):
            x = self.postfilter(x)
        if self.oversampling > 1:
            x = self.output_downsampler(x)
        return x

class HILPQMFPretransform(Pretransform):
    def __init__(self, channels, subbands, taps, beta, cutoff_freq):
        # TODO: Fix PQMF to take in in-channels
        super().__init__(enable_grad=False, io_channels=channels, is_discrete=False)
        from .transforms import PQMF
        self.channels = channels
        self.encoded_channels = channels * subbands
        self.pqmf = PQMF(subbands = subbands, taps = taps, beta = beta, cutoff_freq = cutoff_freq)

    def encode(self, x):
        # x is (Batch x Channels x Time)
        x = fold_channels_into_batch(x)
        pqmf = self.pqmf.analysis(x)
        pqmf = unfold_channels_from_batch(pqmf, self.channels)
        pqmf = rearrange(pqmf, 'b c f t -> b (c f) t')
        return pqmf

    def decode(self, z):
        z = rearrange(z, "b (c f) t -> b c f t", c=self.channels)
        z = fold_channels_into_batch(z)
        x = self.pqmf.synthesis(z)
        x = unfold_channels_from_batch(x, self.channels)
        return x


class PQMFPretransform(Pretransform):
    def __init__(self, attenuation=100, num_bands=16, channels = 1):
        # TODO: Fix PQMF to take in in-channels
        super().__init__(enable_grad=True, io_channels=channels, is_discrete=False)
        from .pqmf import PQMF
        self.pqmf = PQMF(attenuation, num_bands)


    def encode(self, x):
        # x is (Batch x Channels x Time)
        x = self.pqmf.forward(x)
        # pqmf.forward returns (Batch x Channels x Bands x Time)
        # but Pretransform needs Batch x Channels x Time
        # so concatenate channels and bands into one axis
        return rearrange(x, "b c n t -> b (c n) t")

    def decode(self, x):
        # x is (Batch x (Channels Bands) x Time), convert back to (Batch x Channels x Bands x Time) 
        x = rearrange(x, "b (c n) t -> b c n t", n=self.pqmf.num_bands)
        # returns (Batch x Channels x Time) 
        return self.pqmf.inverse(x)
        
class PretrainedDACPretransform(Pretransform):
    def __init__(self, model_type="44khz", model_bitrate="8kbps", scale=1.0, quantize_on_decode: bool = True, chunked=True):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)
        
        import dac
        
        model_path = dac.utils.download(model_type=model_type, model_bitrate=model_bitrate)
        
        self.model = dac.DAC.load(model_path)

        self.quantize_on_decode = quantize_on_decode

        if model_type == "44khz":
            self.downsampling_ratio = 512
        else:
            self.downsampling_ratio = 320

        self.io_channels = 1

        self.scale = scale

        self.chunked = chunked

        self.encoded_channels = self.model.latent_dim

        self.num_quantizers = self.model.n_codebooks

        self.codebook_size = self.model.codebook_size

    def encode(self, x):

        latents = self.model.encoder(x)

        if self.quantize_on_decode:
            output = latents
        else:
            z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.model.n_codebooks)
            output = z
        
        if self.scale != 1.0:
            output = output / self.scale
        
        return output

    def decode(self, z):
        
        if self.scale != 1.0:
            z = z * self.scale

        if self.quantize_on_decode:
            z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        return self.model.decode(z)

    def tokenize(self, x):
        return self.model.encode(x)[1]
    
    def decode_tokens(self, tokens):
        latents = self.model.quantizer.from_codes(tokens)
        return self.model.decode(latents)
    
class AudiocraftCompressionPretransform(Pretransform):
    def __init__(self, model_type="facebook/encodec_32khz", scale=1.0, quantize_on_decode: bool = True):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)
        
        try:
            from audiocraft.models import CompressionModel
        except ImportError:
            raise ImportError("Audiocraft is not installed. Please install audiocraft to use Audiocraft models.")
               
        self.model = CompressionModel.get_pretrained(model_type)

        self.quantize_on_decode = quantize_on_decode

        self.downsampling_ratio = round(self.model.sample_rate / self.model.frame_rate)

        self.sample_rate = self.model.sample_rate

        self.io_channels = self.model.channels

        self.scale = scale

        #self.encoded_channels = self.model.latent_dim

        self.num_quantizers = self.model.num_codebooks

        self.codebook_size = self.model.cardinality

        self.model.to(torch.float16).eval().requires_grad_(False)

    def encode(self, x):

        assert False, "Audiocraft compression models do not support continuous encoding"

        # latents = self.model.encoder(x)

        # if self.quantize_on_decode:
        #     output = latents
        # else:
        #     z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.model.n_codebooks)
        #     output = z
        
        # if self.scale != 1.0:
        #     output = output / self.scale
        
        # return output

    def decode(self, z):
        
        assert False, "Audiocraft compression models do not support continuous decoding"

        # if self.scale != 1.0:
        #     z = z * self.scale

        # if self.quantize_on_decode:
        #     z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        # return self.model.decode(z)

    def tokenize(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            return self.model.encode(x.to(torch.float16))[0]
    
    def decode_tokens(self, tokens):
        with torch.amp.autocast("cuda", enabled=False):
            return self.model.decode(tokens)
