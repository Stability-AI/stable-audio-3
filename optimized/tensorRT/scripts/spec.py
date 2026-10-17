"""Mel-spectrogram renderer ported from Underfit's dashboard.

Source: /path/to/underfit/dashboard/server.py — the 3-band tinted stereo
mel spectrogram used in the Underfit run dashboards. Same algorithm,
same constants, same Slaney mel scale + librosa-compatible mel filterbank.

Differences vs the Underfit original:
  - Input is an int16 (T, 2) numpy array (in-memory PCM) rather than a file
    path; no soundfile dependency. SAMPLE_RATE assumed 44.1 kHz, resampled
    to 32 kHz internally for the spec (matches Underfit's target_sr).
  - Output is PNG bytes returned in-memory (no file write).
  - Exposes render_spectrogram_png(pcm_int16, sample_rate, width, height)
    so callers can choose a render resolution.
"""
from __future__ import annotations
import io
import math
from functools import lru_cache

import numpy as np
import torch
from PIL import Image


# Underfit's 3-band tint
SPEC_BANDS = [
    (0, 200, (1.0, 0.0, 0.0)),       # Bass -> Red
    (200, 1500, (0.0, 1.0, 0.0)),    # Mid  -> Green
    (1500, 16000, (0.0, 0.0, 1.0)),  # High -> Blue
]
_BAND_COLORS = np.array([c for _, _, c in SPEC_BANDS], dtype=np.float32)

# Slaney mel-scale constants (linear below 1 kHz, log above), matches librosa default.
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    log_term = np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(hz >= _MIN_LOG_HZ, _MIN_LOG_MEL + log_term, hz / _F_SP)


def _mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    return np.where(mels >= _MIN_LOG_MEL,
                    _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels - _MIN_LOG_MEL)),
                    _F_SP * mels)


def _mel_frequencies(n_mels, fmax, fmin=0.0):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels))


@lru_cache(maxsize=8)
def _mel_filterbank(n_mels, n_fft, sr, fmax):
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    filt = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (fft_f - lo) / max(ce - lo, 1e-10)
        right = (hi - fft_f) / max(hi - ce, 1e-10)
        filt[i] = np.maximum(0, np.minimum(left, right))
    enorm = (2.0 / (pts[2:n_mels + 2] - pts[0:n_mels])).astype(np.float32)
    filt *= enorm[:, None]
    return torch.from_numpy(filt)


def _melspectrogram(y_ch, sr, n_mels=30, fmax=16000, hop_length=2048, n_fft=2048):
    y_t = torch.from_numpy(np.ascontiguousarray(y_ch)).float()
    win = torch.hann_window(n_fft)
    spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, window=win,
                       center=True, return_complex=True, pad_mode='reflect')
    return (_mel_filterbank(n_mels, n_fft, sr, fmax) @ spec.abs().square()).numpy()


def _power_to_db(S, top_db=80.0):
    log_spec = 10.0 * np.log10(np.maximum(S, 1e-10))
    return np.maximum(log_spec - log_spec.max(), -top_db)


def _mel_channel(y_ch, sr, n_mels=30):
    """dB-scaled mel + band-tinted RGB for one mono channel."""
    S = _melspectrogram(y_ch, sr, n_mels=n_mels, fmax=16000, hop_length=2048)

    S_db = _power_to_db(S)
    np.clip(S_db, -60, 0, out=S_db)
    S_db += 60.0
    S_db /= 60.0
    np.power(S_db, 0.6, out=S_db)

    mel_f = _mel_frequencies(n_mels, fmax=16000)
    n_frames = S.shape[1]
    band_norms = np.empty((3, n_frames), dtype=np.float32)
    for i, (flo, fhi, _) in enumerate(SPEC_BANDS):
        mask = (mel_f >= flo) & (mel_f < fhi)
        if mask.any():
            power = np.sum(S[mask], axis=0)
            db = 10.0 * np.log10(power + 1e-10)
            np.clip(db, -20, None, out=db)
            db -= -20
            mx = db.max()
            if mx > 0:
                db /= mx
            band_norms[i] = db
        else:
            band_norms[i] = 0.0

    rgb = band_norms.T @ _BAND_COLORS
    for c in range(3):
        mx = rgb[:, c].max()
        if mx > 0:
            rgb[:, c] /= mx

    return S_db, rgb


def _resample_to_32k(pcm_int16, sr_in):
    """Resample (T, 2) int16 PCM to (2, T') float32 at 32 kHz."""
    target_sr = 32000
    if sr_in == target_sr:
        y = pcm_int16.astype(np.float32) / 32768.0
        return y.T, target_sr
    # Convert int16 → float32 → (2, T), then torch interp.
    y = pcm_int16.astype(np.float32).T / 32768.0      # (2, T)
    y_t = torch.from_numpy(np.ascontiguousarray(y))
    new_len = int(round(y_t.shape[-1] * target_sr / sr_in))
    y_t = torch.nn.functional.interpolate(
        y_t.unsqueeze(0), size=new_len, mode='linear', align_corners=False
    ).squeeze(0)
    return y_t.numpy(), target_sr


def render_spectrogram_png(pcm_int16, sample_rate=44100, width=1200, height=240,
                            n_mels=30) -> bytes:
    """Render Underfit-style 3-band tinted stereo mel spectrogram to PNG bytes.

    Args:
        pcm_int16: (T, 2) int16 numpy array — stereo PCM.
        sample_rate: input PCM sample rate (default 44100).
        width, height: final image size (default 1200x240; Underfit ships 300x60).
        n_mels: mel filterbank size (30 matches Underfit).

    Returns: PNG bytes ready to base64 / save / etc.
    """
    if pcm_int16.ndim != 2 or pcm_int16.shape[1] != 2:
        raise ValueError(f"expected (T, 2) PCM, got {pcm_int16.shape}")
    y, sr = _resample_to_32k(pcm_int16, sample_rate)
    # y is (2, T') float32
    S_L, rgb_L = _mel_channel(y[0], sr, n_mels=n_mels)
    S_R, rgb_R = _mel_channel(y[1], sr, n_mels=n_mels)

    nf = min(S_L.shape[1], S_R.shape[1])
    S_L, S_R = S_L[:, :nf], S_R[:, :nf]
    rgb_L, rgb_R = rgb_L[:nf], rgb_R[:nf]
    nm = S_L.shape[0]

    # L channel: flip vertically so bass-band is at bottom.
    S_L = S_L[::-1]

    img = np.empty((nm * 2, nf, 3), dtype=np.float32)
    img[:nm] = S_L[:, :, np.newaxis] * rgb_L[np.newaxis, :, :]
    img[nm:] = S_R[:, :, np.newaxis] * rgb_R[np.newaxis, :, :]

    np.clip(img, 0, 1, out=img)
    img *= 255
    pil = Image.fromarray(img.astype(np.uint8)).resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
