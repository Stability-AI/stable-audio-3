"""
TRT plugin for diff SWA attention — NO dtype cast, minimal overhead.
Triton SWA kernel auto-compiles for whatever dtype TRT passes (FP32 or BF16).
Eliminates ~18ms of cast overhead per decode at L=1292.
"""
import torch
import tensorrt.plugin as trtp
import numpy as np
import numpy.typing as npt
from typing import Tuple

_stream_cache = {}
_triton_fn = None


@trtp.register("samel::diff_attn_swa")
def diff_attn_swa_desc(q_bat: trtp.TensorDesc, k_bat: trtp.TensorDesc,
                        v_bat: trtp.TensorDesc,
                        num_heads: npt.NDArray[np.int64]) -> trtp.TensorDesc:
    out = q_bat.like()
    out.shape_expr[-2] = q_bat.shape_expr[-2] // 2
    return out


@trtp.impl("samel::diff_attn_swa")
def diff_attn_swa_impl(q_bat: trtp.Tensor, k_bat: trtp.Tensor, v_bat: trtp.Tensor,
                         num_heads: npt.NDArray[np.int64],
                         outputs: Tuple[trtp.Tensor], stream: int):
    global _triton_fn
    if stream not in _stream_cache:
        _stream_cache[stream] = torch.cuda.ExternalStream(stream)
    if _triton_fn is None:
        from triton_swa_v2 import triton_swa_attn_v2
        _triton_fn = triton_swa_attn_v2

    with torch.cuda.stream(_stream_cache[stream]):
        q = torch.as_tensor(q_bat, device="cuda")
        k = torch.as_tensor(k_bat, device="cuda")
        v = torch.as_tensor(v_bat, device="cuda")
        out_t = torch.as_tensor(outputs[0], device="cuda")

        # NO dtype cast — Triton auto-compiles for the input dtype
        o = _triton_fn(q, k, v, window=17)
        H = int(np.asarray(num_heads).reshape(-1)[0])
        out_t.copy_(o[:, :, :H, :] - o[:, :, H:, :])
