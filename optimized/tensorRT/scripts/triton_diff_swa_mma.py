"""Block-tiled differential SWA attention in Triton, shaped for AOT compilation.

Why this exists
---------------
The SAME-L decoder's SWA plugin needs an ahead-of-time (PTX) implementation so its TRT
engine is CUDA-graph-capturable — the JIT path re-enters Python per enqueue, which
makes the engine non-capturable on sm_120 and silently drops the decode from the
mega-graph. The existing AOT kernel (diff_swa_ptx_kernel.py) is hand-written PTX that
is one warp per query, scalar FP32, no K/V reuse, and emits zero mma instructions —
24% slower than the Triton JIT path on sm_90, and sweeping its only knob
(warps_per_block over 1..16) moves it 7%, so the gap is algorithmic.

This module supplies the missing algorithm: BLOCK_N queries per block sharing the K/V
window in SRAM, with tl.dot so QK^T and P·V land on tensor cores. Compiling it with
triton.compile() yields PTX containing mma.sync instructions, which can then be handed
to trtp.aot_impl — Triton's own codegen instead of hand-written fragment layouts.

The one substantive difference from triton_swa_v2.swa_attn_v2_kernel: that kernel
computes all 2H heads and leaves the differential subtraction to the caller
(o[:, :, :H] - o[:, :, H:]) in PyTorch. aot_impl returns a SINGLE kernel, so there is
no caller to do that — the subtraction has to happen inside. Each block therefore owns
one OUTPUT head h and processes two input heads, h (primary) and h + H (diff),
subtracting before the store. That also saves materialising a 2H-head intermediate.

Precision note: tl.dot on fp32 inputs lowers to TF32 mma (10 mantissa bits), which is
what the existing Triton path already does. The subtraction itself stays in FP32 on
FP32 accumulator values, so the cancellation-sensitive step is unaffected — only the
two attention products are TF32, exactly as in the shipped JIT path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def diff_swa_mma_kernel(
    # PARAMETER ORDER IS LOAD-BEARING: TRT's AOT plugin launcher passes
    #   inputs..., extras..., outputs...
    # (confirmed against the working hand-written kernel in diff_swa_ptx_kernel.py,
    # whose entry is Q, K, V, N, 4 strides, Out — output LAST, after the extras).
    # An earlier version declared (Q, K, V, Out, N) and TRT therefore handed N where
    # the kernel expected the output pointer: it dereferenced 4352 as an address and
    # the engine died with "illegal memory access". Runtime scalars must sit between
    # the inputs and the outputs.
    Q, K, V,
    N,
    Out,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_ob, stride_on, stride_oh, stride_od,
    scale, H_OUT,
    WINDOW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Differential SWA for one output head, one query block.

    grid = (ceil(N / BLOCK_N), H_OUT, B)
    Out[b, n, h, :] = softmax(Qh Kh^T)Vh - softmax(Qh' Kh'^T)Vh'  where h' = h + H_OUT
    """
    block_n = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    q_start = block_n * BLOCK_N
    q_offs = tl.arange(0, BLOCK_N)
    q_pos = q_start + q_offs
    q_mask = q_pos < N

    # K/V window spans [q_start - WINDOW, q_start + BLOCK_N + WINDOW)
    kv_start = q_start - WINDOW
    kv_offs = tl.arange(0, BLOCK_KV)
    kv_pos = kv_start + kv_offs
    kv_mask = (kv_pos >= 0) & (kv_pos < N)

    d_offs = tl.arange(0, BLOCK_D)
    d_ok = d_offs[None, :] < BLOCK_D

    # SWA mask is purely relative, so it is shared by both heads. kv_start is offset by
    # -WINDOW, hence |q_offs + WINDOW - kv_offs| <= WINDOW.
    rel = q_offs[:, None] + WINDOW - kv_offs[None, :]
    valid = (rel >= -WINDOW) & (rel <= WINDOW) & kv_mask[None, :] & q_mask[:, None]

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # head_sel 0 -> primary head h, 1 -> diff head h + H_OUT.
    #
    # tl.static_range, NOT a dynamic range(2). Measured: unrolled needs 64 KB of shared
    # memory (both heads' tiles live) and emits 512 mma; a dynamic loop is WORSE, not
    # better — Triton's pipeliner allocates multi-stage buffers for it and shared memory
    # jumps to 160 KB, with mma halved to 256. 64 KB exceeds the 48 KB *static* limit
    # and needs a cudaFuncSetAttribute opt-in from whoever launches it, which is the
    # open question for the TRT AOT path (see AOT_KERNEL_STATUS.md).
    for head_sel in tl.static_range(2):
        hh = h + head_sel * H_OUT

        q_ptrs = (Q + b * stride_qb + q_pos[:, None] * stride_qn
                  + hh * stride_qh + d_offs[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=q_mask[:, None] & d_ok, other=0.0)

        k_ptrs = (K + b * stride_kb + kv_pos[:, None] * stride_kn
                  + hh * stride_kh + d_offs[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=kv_mask[:, None] & d_ok, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale          # tensor cores
        scores = tl.where(valid, scores, float("-inf"))

        row_max = tl.max(scores, axis=1)
        e = tl.exp(scores - row_max[:, None])
        e = tl.where(valid, e, 0.0)
        row_sum = tl.sum(e, axis=1)
        w = e / row_sum[:, None]

        v_ptrs = (V + b * stride_vb + kv_pos[:, None] * stride_vn
                  + hh * stride_vh + d_offs[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & d_ok, other=0.0)

        out_h = tl.dot(w, v)                              # tensor cores

        # FP32 accumulator: primary minus diff. The cancellation-sensitive step stays
        # in FP32 even though the two products above are TF32.
        acc += tl.where(head_sel == 0, out_h, -out_h)

    o_ptrs = (Out + b * stride_ob + q_pos[:, None] * stride_on
              + h * stride_oh + d_offs[None, :] * stride_od)
    tl.store(o_ptrs, acc, mask=q_mask[:, None] & d_ok)


# Launch geometry, kept here so the AOT path and the reference path agree exactly.
#
# BLOCK_KV MUST be >= BLOCK_N + 2*WINDOW or the K/V tile does not cover every position
# the block's queries can attend to, and attention contributions are silently dropped.
# (BLOCK_N=32 with BLOCK_KV=64 needs 66 and is therefore WRONG, even though it compiles
# and fits in shared memory.)
#
# BLOCK_N=16 rather than 64, for two reasons measured on the real shapes:
#   speed   0.306 ms/layer vs 0.432 at BLOCK_N=64 (JIT reference: 0.421). With a window
#           of only +/-17, a 128-wide K/V tile is mostly masked-out waste; a 64-wide tile
#           keeps the useful fraction high.
#   memory  40 KB of shared memory vs 64 KB. 64 KB exceeds the 48 KB static limit and
#           needs a cudaFuncSetAttribute opt-in that the TRT AOT launcher does not do —
#           the BLOCK_N=64 engine built fine and then failed at enqueue with
#           "Failed to enqueue status -1", returning zeros.
WINDOW = 17
BLOCK_N = 16
BLOCK_D = 64
BLOCK_KV = 64           # >= BLOCK_N + 2 * WINDOW = 50


def diff_swa_mma(q, k, v, window: int = WINDOW):
    """Reference launcher. q, k, v: (B, N, 2H, D) fp32 -> (B, N, H, D) fp32.

    Used to validate the kernel before it is compiled AOT; the shipped path drives the
    same kernel through trtp.aot_impl instead.
    """
    B, Nn, H2, D = q.shape
    assert H2 % 2 == 0, f"expected an even head count, got {H2}"
    assert D == BLOCK_D, f"kernel is specialised for D={BLOCK_D}, got {D}"
    H = H2 // 2
    out = torch.empty((B, Nn, H, D), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(Nn, BLOCK_N), H, B)
    diff_swa_mma_kernel[grid](
        q, k, v, Nn, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        D ** -0.5, H,
        WINDOW=window, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
    )
    return out
