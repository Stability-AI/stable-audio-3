"""Device-neutral helpers for CUDA / MPS / CPU.

Single-device training and inference should route device selection and AMP
(autocast / GradScaler) construction through this module instead of hardcoding
"cuda". Multi-GPU / DeepSpeed / Lightning-strategy code is out of scope and
intentionally untouched.

Empirically verified on torch 2.13.0 + macOS 15 (Apple Silicon):
  - torch.amp.autocast("mps", dtype=torch.float16) works
  - torch.amp.autocast("mps", dtype=torch.bfloat16) works (macOS 14+ required)
  - torch.amp.GradScaler("mps") works (scale/unscale_/step/update)
  - @autocast("cuda", enabled=False) does NOT disable an active MPS autocast,
    so fp32 islands need the device-aware `disable_autocast` below.
"""
import functools

import torch


def resolve_device(preference=None) -> str:
    """Best available device string: explicit preference > cuda > mps > cpu."""
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=None)
def mps_bf16_supported() -> bool:
    """bf16 on MPS needs macOS 14+; probe once with a tiny op."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.ones(2, 2, device="mps", dtype=torch.bfloat16)
        (x @ x).sum().item()
        return True
    except Exception:
        return False


def autocast_context(device_type=None, dtype=None, enabled=True):
    """torch.amp.autocast with per-backend fixups.

    - device_type may be a device string ("cuda:0", "mps") or None (auto-resolve).
    - bf16 on MPS is downgraded to fp16 (with a printed note) when the OS
      doesn't support it.
    """
    dt = torch.device(device_type or resolve_device()).type
    if dt == "mps" and dtype is torch.bfloat16 and not mps_bf16_supported():
        print(
            "[device] bf16 autocast unsupported on this macOS/MPS build; "
            "falling back to fp16 autocast",
            flush=True,
        )
        dtype = torch.float16
    if dt == "cpu":
        # Training code historically used autocast("cuda"), a silent no-op on
        # CPU-only hosts. Keep CPU runs in fp32 rather than newly enabling CPU
        # fp16/bf16 autocast (slow and numerically different).
        enabled = False
    return torch.amp.autocast(dt, dtype=dtype, enabled=enabled)


class NoOpGradScaler:
    """Transparent stand-in when torch.amp.GradScaler doesn't support a backend."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def make_grad_scaler(device_type=None, enabled=True):
    """GradScaler for the given backend.

    torch 2.13 supports GradScaler("mps") natively (verified). On CPU, or if
    construction fails (older torch), returns a disabled/no-op scaler so
    callers can use the scaler API unconditionally.
    """
    dt = torch.device(device_type or resolve_device()).type
    if dt == "cpu":
        # fp16 grad scaling is pointless on CPU; keep the API but disabled
        # (matches the old GradScaler("cuda")-on-cpu auto-disable behavior).
        enabled = False
    try:
        return torch.amp.GradScaler(dt, enabled=enabled)
    except Exception as e:
        if enabled:
            print(
                f"[device] torch.amp.GradScaler({dt!r}) unsupported "
                f"({type(e).__name__}: {e}); using no-op scaler — fp16 loss "
                "scaling disabled, watch for gradient underflow",
                flush=True,
            )
        return NoOpGradScaler()


def _first_tensor_device_type(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device.type
    for a in kwargs.values():
        if isinstance(a, torch.Tensor):
            return a.device.type
    return None


def disable_autocast(fn):
    """Device-aware replacement for @autocast("cuda", enabled=False).

    The cuda-pinned decorator does NOT disable an active MPS (or CPU) autocast
    context, silently defeating fp32 islands (RoPE, Fourier timestep features)
    on non-CUDA backends. This wrapper disables autocast for the device the
    inputs actually live on.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        dt = _first_tensor_device_type(args, kwargs) or "cuda"
        with torch.amp.autocast(dt, enabled=False):
            return fn(*args, **kwargs)

    return wrapper


def empty_device_cache(device_type=None):
    """Release cached allocator memory on the active accelerator, if any."""
    dt = torch.device(device_type or resolve_device()).type
    if dt == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dt == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
