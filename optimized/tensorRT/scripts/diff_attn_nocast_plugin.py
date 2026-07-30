"""
TRT plugin for diff SWA attention — NO dtype cast, minimal overhead.

Two implementations of the SAME op are registered:

  aot_impl  (preferred)  An ahead-of-time PTX kernel compiled INTO the engine at
                         build time. No Python, no Triton, nothing host-side in the
                         enqueue path — so the engine is CUDA-graph-capturable.
  impl      (fallback)   The original Triton path, which re-enters Python on every
                         enqueue. TRT uses this only if the AOT path is unavailable.

Why the AOT path matters: with `impl` alone the built engine is NOT stream-capturable
on sm_120. `enqueueV3` returns False under capture, TRT reports "this TRT engine is
not stream capturable" from myelin/runner.cpp, and the stage is silently omitted from
the graph — the SAME-L decode vanished from the mega-graph and every render returned
the pre-capture warmup decode of zero latents (a constant wash, identical across
seeds and prompts, exit code 0). It happened to capture on sm_90, but re-entering
Python inside a captured region is fragile by construction rather than supported.

Capturability is baked into the engine at BUILD time, so engines must be rebuilt to
pick this up. The ONNX is unchanged — same op name, same signature.

The AOT kernel's contract, verified against what TRT actually passes this op in the
SAME-L decoder: FP32 in/out, q/k/v each (B, N, 2H, D) = (1, 4352, 48, 64), D=64,
2H=48 so H=24, window=17. It performs the differential subtraction in FP32 inside
the kernel, which is also what the Triton path does via o[:, :, :H] - o[:, :, H:].
"""
import torch
import tensorrt.plugin as trtp
from typing import Tuple

from diff_swa_ptx_kernel import generate_diff_swa_ptx

WINDOW = 17
HEAD_DIM = 64
WARPS_PER_BLOCK = 4
_ptx_cache: dict = {}


def _ptx_for(num_heads: int):
    """(kernel_name, ptx) for this head count, generated once and cached."""
    if num_heads not in _ptx_cache:
        _ptx_cache[num_heads] = generate_diff_swa_ptx(
            window=WINDOW, D=HEAD_DIM, H=num_heads,
            warps_per_block=WARPS_PER_BLOCK)
    return _ptx_cache[num_heads]

_stream_cache = {}
_triton_fn = None


@trtp.register("samel::diff_attn_swa")
def diff_attn_swa_desc(q_bat: trtp.TensorDesc, k_bat: trtp.TensorDesc,
                        v_bat: trtp.TensorDesc, num_heads: int) -> trtp.TensorDesc:
    out = q_bat.like()
    out.shape_expr[-2] = q_bat.shape_expr[-2] // 2
    return out


@trtp.impl("samel::diff_attn_swa")
def diff_attn_swa_impl(q_bat: trtp.Tensor, k_bat: trtp.Tensor, v_bat: trtp.Tensor,
                         num_heads: int, outputs: Tuple[trtp.Tensor], stream: int):
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
        H = num_heads
        out_t.copy_(o[:, :, :H, :] - o[:, :, H:, :])


@trtp.aot_impl("samel::diff_attn_swa")
def diff_attn_swa_aot(q_bat: trtp.TensorDesc, k_bat: trtp.TensorDesc,
                       v_bat: trtp.TensorDesc, num_heads: int,
                       outputs: Tuple[trtp.TensorDesc], tactic: int = None
                       ) -> Tuple[str, str, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    """AOT PTX variant — makes the engine graph-capturable. See module docstring.

    Grid is (ceil(N / warps_per_block), H_out, B); each warp handles one query
    position for head h and head h+H, subtracting in FP32 before the store. Strides
    are passed as symbolic extras so the kernel works across the dynamic N axis.
    """
    name, ptx = _ptx_for(num_heads)
    B = q_bat.shape_expr[0]
    N = q_bat.shape_expr[1]
    H2 = q_bat.shape_expr[2]
    D = q_bat.shape_expr[3]
    H_out = H2 // 2

    extra = trtp.SymIntExprs(5)
    extra[0] = N
    extra[1] = H2 * D        # stride between positions, input
    extra[2] = D             # stride between heads, input
    extra[3] = H_out * D     # stride between positions, output
    extra[4] = D             # stride between heads, output

    return (name, ptx,
            trtp.KernelLaunchParams(
                grid_x=trtp.cdiv(N, WARPS_PER_BLOCK),
                grid_y=H_out,
                grid_z=B,
                block_x=WARPS_PER_BLOCK * 32),
            extra)
