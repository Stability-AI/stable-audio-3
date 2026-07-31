"""Triton-compiled block-tiled MMA kernel, packaged as AOT PTX for trtp.aot_impl.

The SAME-L SWA plugin needs an ahead-of-time kernel so its TRT engine is
CUDA-graph-capturable. The existing AOT kernel (diff_swa_ptx_kernel.py) is hand-written
scalar-FP32 PTX — one warp per query, no K/V reuse, zero mma instructions — and is ~3.6x
slower per layer than the Triton JIT path, which across the decoder's 12 attention
layers accounts for the whole 55 -> 68 ms engine-level gap on sm_90.

This module closes that gap by compiling triton_diff_swa_mma's block-tiled kernel
ahead of time and exposing its PTX. Using Triton as the code generator rather than
hand-writing mma fragment layouts is deliberate: it already emits exactly the wanted
codegen (verified: 512 x mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32), and PTX
MMA fragment register assignment is error-prone to write by hand.

Two things have to be reconciled with what trtp.aot_impl can actually pass a kernel:

  1. The `extra` channel carries INT32s only, so every non-int runtime scalar must be a
     constexpr. `scale` (fp32), `H_OUT`, `WINDOW`, the block sizes and 15 of the 16
     strides are therefore pinned at compile time, leaving only `N` at runtime.
  2. Triton appends two global-scratch pointers to the entry signature. TRT passes
     input/output pointers plus the extras and has no channel for them — but they are
     DECLARED AND NEVER LOADED (checked: only the declaration mentions them, there is no
     ld.param), so they are stripped from the PTX entry signature. That makes the
     kernel's declared parameter space exactly match what TRT supplies, rather than
     relying on the driver tolerating a short argument list.
"""
import re

import triton
from triton.compiler.compiler import ASTSource

from triton_diff_swa_mma import (diff_swa_mma_kernel, WINDOW, BLOCK_N, BLOCK_D,
                                 BLOCK_KV)

HEAD_DIM = BLOCK_D
_cache: dict = {}


def _strip_unused_tail_params(ptx: str, keep: int) -> str:
    """Remove entry parameters after the first `keep`, asserting they are unread.

    Triton's trailing global-scratch pointers have no channel through aot_impl. They
    are never dereferenced, so dropping the declarations is safe — but verify that
    before cutting, so this fails loudly if a future Triton starts using them.
    """
    m = re.search(r"(\.visible \.entry \w+\()([^)]*)(\))", ptx, re.S)
    if m is None:
        raise RuntimeError("could not locate the PTX entry signature")
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    if len(params) <= keep:
        return ptx
    for p in params[keep:]:
        name = p.split()[-1]
        if re.search(rf"ld\.param[^\n]*\[{re.escape(name)}\]", ptx):
            raise RuntimeError(
                f"refusing to strip {name}: it IS loaded by the kernel, so TRT must "
                f"pass it and this packaging is invalid")
    new_sig = ",\n\t".join(params[:keep])
    return ptx[:m.start()] + m.group(1) + "\n\t" + new_sig + "\n" + m.group(3) + ptx[m.end():]


def build(num_heads: int, arch: str):
    """(kernel_name, ptx, shared_mem_bytes, num_warps) for this head count.

    `arch` is the sm_XX being built for; the PTX `.target` is rewritten to it without
    the trailing 'a' (architecture-specific) suffix, which TRT's loader rejects. The
    m16n8k8 TF32 mma this kernel uses is available from sm_80, so the plain target is
    sufficient.
    """
    key = (num_heads, arch)
    if key in _cache:
        return _cache[key]

    h2 = 2 * num_heads
    # Order must match the kernel: inputs, then the runtime extra, then outputs.
    sig = {"Q": "*fp32", "K": "*fp32", "V": "*fp32", "N": "i32", "Out": "*fp32"}
    sig.update({f"stride_{t}{a}": "constexpr" for t in "qkvo" for a in "bnhd"})
    sig.update({"scale": "constexpr", "H_OUT": "constexpr",
                "WINDOW": "constexpr", "BLOCK_N": "constexpr",
                "BLOCK_D": "constexpr", "BLOCK_KV": "constexpr"})
    # Contiguous (B, N, 2H, D) in and (B, N, H, D) out. The batch strides are only ever
    # multiplied by program_id(2), which is 0 for the decoder's batch of 1, so pinning
    # them to 0 is exact here rather than an approximation.
    consts = {
        "stride_qb": 0, "stride_qn": h2 * HEAD_DIM, "stride_qh": HEAD_DIM, "stride_qd": 1,
        "stride_kb": 0, "stride_kn": h2 * HEAD_DIM, "stride_kh": HEAD_DIM, "stride_kd": 1,
        "stride_vb": 0, "stride_vn": h2 * HEAD_DIM, "stride_vh": HEAD_DIM, "stride_vd": 1,
        "stride_ob": 0, "stride_on": num_heads * HEAD_DIM, "stride_oh": HEAD_DIM,
        "stride_od": 1,
        "scale": HEAD_DIM ** -0.5, "H_OUT": num_heads, "WINDOW": WINDOW,
        "BLOCK_N": BLOCK_N, "BLOCK_D": BLOCK_D, "BLOCK_KV": BLOCK_KV,
    }
    try:
        src = ASTSource(diff_swa_mma_kernel, signature=sig, constexprs=consts)
    except TypeError:                     # older Triton spells it `constants`
        src = ASTSource(diff_swa_mma_kernel, signature=sig, constants=consts)

    compiled = triton.compile(src)
    ptx = compiled.asm["ptx"]
    md = compiled.metadata

    n_mma = len(re.findall(r"mma\.sync", ptx))
    if n_mma == 0:
        raise RuntimeError("compiled PTX has no mma instructions — tl.dot did not "
                           "lower to tensor cores; the kernel would be no faster than "
                           "the scalar hand-written one")

    # TRT passes: Q, K, V (inputs), Out (output), then the extras. One extra (N).
    ptx = _strip_unused_tail_params(ptx, keep=5)
    ptx = re.sub(r"\.target\s+sm_\w+", f".target {arch}", ptx)
    # fp32 loads need 4-byte alignment; Triton emits .align 1 on the pointer params.
    ptx = ptx.replace(".align 1 .b8", ".align 4 .b8")

    _cache[key] = (md.name, ptx, int(md.shared), int(md.num_warps))
    return _cache[key]
