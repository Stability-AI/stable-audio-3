"""Current-GPU SM arch as a plain 'sm_XX' string.

Used by swa_mma_aot to retarget the AOT SWA kernel's PTX `.target`. The plain form (no
architecture-specific 'a' suffix, e.g. sm_90a / sm_120a) is required because TRT's PTX
loader rejects the arch-specific variants; the m16n8k8 TF32 MMA this kernel uses is
available from sm_80 up, so the plain target compiles and loads on every arch.
"""


def detect_arch() -> str:
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"
