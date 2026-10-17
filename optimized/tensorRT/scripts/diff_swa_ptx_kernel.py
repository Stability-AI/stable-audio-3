"""
Generate PTX for DIFFERENTIAL SWA attention — subtraction built into the kernel.
Avoids FP16 catastrophic cancellation by computing (attn_primary - attn_diff) in FP32.

Grid: (ceil(N/warps_per_block), H, B)  — H is num_heads (24), NOT 2*H
Each warp: computes SWA for head h (primary) and head h+H (diff), then subtracts.
Output: (B, N, H, D) — already-subtracted differential result.

Plugin signature: diff_swa_attn(Q_bat, K_bat, V_bat) where Q_bat is (B, N, 2H, D)
Output: (B, N, H, D)
"""

def generate_diff_swa_ptx(window=17, D=64, H=24, warps_per_block=4):
    """Generate PTX for differential SWA. Returns (kernel_name, ptx_string)."""
    WIN = 2 * window + 1
    H2 = 2 * H  # Total heads in input (48)
    THREADS_PER_BLOCK = warps_per_block * 32
    elem_shift = 2  # log2(4 bytes for fp32)
    kernel_name = "diff_swa_attn_ptx"

    ptx = f"""
.version 8.0
.target sm_90
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 .ptr .global .align 4 Q_ptr,
    .param .u64 .ptr .global .align 4 K_ptr,
    .param .u64 .ptr .global .align 4 V_ptr,
    .param .u32 N_param,
    .param .u32 stride_n_in,
    .param .u32 stride_h_in,
    .param .u32 stride_n_out,
    .param .u32 stride_h_out,
    .param .u64 .ptr .global .align 4 Out_ptr
) {{
    .reg .pred %p<16>;
    .reg .b32 %r<80>;
    .reg .f32 %f<160>;
    .reg .b64 %rd<40>;

    // Thread indexing: {warps_per_block} warps per block
    mov.u32 %r3, %tid.x;
    shr.u32 %r40, %r3, 5;    // warp_id
    and.b32 %r41, %r3, 31;   // lane_id

    // Position: block_n * {warps_per_block} + warp_id
    mov.u32 %r42, %ctaid.x;
    mov.u32 %r43, {warps_per_block};
    mul.lo.u32 %r44, %r42, %r43;
    add.u32 %r0, %r44, %r40; // n = position

    mov.u32 %r1, %ctaid.y;   // h = output head (0..{H-1})
    mov.u32 %r2, %ctaid.z;   // b = batch

    ld.param.u32 %r4, [N_param];
    ld.param.u32 %r5, [stride_n_in];   // input stride_n (for 2H heads)
    ld.param.u32 %r7, [stride_h_in];   // input stride_h
    ld.param.u32 %r50, [stride_n_out]; // output stride_n (for H heads)
    ld.param.u32 %r51, [stride_h_out]; // output stride_h

    setp.ge.u32 %p0, %r0, %r4;
    @%p0 bra DONE;

    // Compute INPUT base offset for PRIMARY head h: n*stride_n + h*stride_h + lane*2
    mul.lo.u32 %r6, %r0, %r5;
    mad.lo.u32 %r8, %r1, %r7, %r6;
    shl.b32 %r9, %r41, 1;
    add.u32 %r10, %r8, %r9;

    // Compute INPUT base offset for DIFF head h+{H}: n*stride_n + (h+{H})*stride_h + lane*2
    mov.u32 %r52, {H};
    add.u32 %r53, %r1, %r52;  // h + H
    mad.lo.u32 %r54, %r53, %r7, %r6;  // n*stride_n + (h+H)*stride_h
    add.u32 %r55, %r54, %r9;  // + lane*2

    // Load Q for PRIMARY head
    cvt.u64.u32 %rd0, %r10;
    shl.b64 %rd1, %rd0, {elem_shift};
    ld.param.u64 %rd2, [Q_ptr];
    add.u64 %rd3, %rd2, %rd1;
    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd3 + 4];

    // Load Q for DIFF head
    cvt.u64.u32 %rd20, %r55;
    shl.b64 %rd21, %rd20, {elem_shift};
    add.u64 %rd22, %rd2, %rd21;
    ld.global.f32 %f2, [%rd22];
    ld.global.f32 %f3, [%rd22 + 4];

    // Scale
    mov.f32 %f14, 0f3E000000;  // 0.125 = 1/sqrt(64)
    mov.f32 %f40, 0f3FB8AA3B;  // log2(e)

    // Online softmax for PRIMARY
    mov.f32 %f10, 0fFF800000; mov.f32 %f11, 0f00000000;
    mov.f32 %f12, 0f00000000; mov.f32 %f13, 0f00000000;
    // Online softmax for DIFF
    mov.f32 %f60, 0fFF800000; mov.f32 %f61, 0f00000000;
    mov.f32 %f62, 0f00000000; mov.f32 %f63, 0f00000000;
"""

    for wi in range(WIN):
        ptx += f"""
    // --- Window position {wi} ---
    mov.s32 %r21, %r0;
    add.s32 %r21, %r21, {wi - window};
    setp.lt.s32 %p1, %r21, 0;
    setp.ge.s32 %p2, %r21, %r4;
    or.pred %p3, %p1, %p2;
    @%p3 bra SKIP_{wi};

    // K offset for kp: kp*stride_n + h*stride_h + lane*2 (PRIMARY)
    mul.lo.u32 %r22, %r21, %r5;
    mad.lo.u32 %r23, %r1, %r7, %r22;
    add.u32 %r24, %r23, %r9;
    cvt.u64.u32 %rd4, %r24;
    shl.b64 %rd5, %rd4, {elem_shift};
    ld.param.u64 %rd6, [K_ptr];
    add.u64 %rd7, %rd6, %rd5;
    ld.global.f32 %f20, [%rd7];
    ld.global.f32 %f21, [%rd7 + 4];

    // K offset for DIFF head: kp*stride_n + (h+H)*stride_h + lane*2
    mad.lo.u32 %r56, %r53, %r7, %r22;
    add.u32 %r57, %r56, %r9;
    cvt.u64.u32 %rd23, %r57;
    shl.b64 %rd24, %rd23, {elem_shift};
    add.u64 %rd25, %rd6, %rd24;
    ld.global.f32 %f70, [%rd25];
    ld.global.f32 %f71, [%rd25 + 4];

    // PRIMARY dot product
    mul.f32 %f22, %f0, %f20;
    fma.rn.f32 %f22, %f1, %f21, %f22;
    mov.b32 %r30, %f22;
    shfl.sync.bfly.b32 %r30, %r30, 16, 0x1f, 0xffffffff; mov.b32 %f23, %r30; add.f32 %f22, %f22, %f23;
    mov.b32 %r30, %f22;
    shfl.sync.bfly.b32 %r30, %r30, 8, 0x1f, 0xffffffff; mov.b32 %f23, %r30; add.f32 %f22, %f22, %f23;
    mov.b32 %r30, %f22;
    shfl.sync.bfly.b32 %r30, %r30, 4, 0x1f, 0xffffffff; mov.b32 %f23, %r30; add.f32 %f22, %f22, %f23;
    mov.b32 %r30, %f22;
    shfl.sync.bfly.b32 %r30, %r30, 2, 0x1f, 0xffffffff; mov.b32 %f23, %r30; add.f32 %f22, %f22, %f23;
    mov.b32 %r30, %f22;
    shfl.sync.bfly.b32 %r30, %r30, 1, 0x1f, 0xffffffff; mov.b32 %f23, %r30; add.f32 %f22, %f22, %f23;
    mul.f32 %f22, %f22, %f14;

    // DIFF dot product
    mul.f32 %f72, %f2, %f70;
    fma.rn.f32 %f72, %f3, %f71, %f72;
    mov.b32 %r30, %f72;
    shfl.sync.bfly.b32 %r30, %r30, 16, 0x1f, 0xffffffff; mov.b32 %f73, %r30; add.f32 %f72, %f72, %f73;
    mov.b32 %r30, %f72;
    shfl.sync.bfly.b32 %r30, %r30, 8, 0x1f, 0xffffffff; mov.b32 %f73, %r30; add.f32 %f72, %f72, %f73;
    mov.b32 %r30, %f72;
    shfl.sync.bfly.b32 %r30, %r30, 4, 0x1f, 0xffffffff; mov.b32 %f73, %r30; add.f32 %f72, %f72, %f73;
    mov.b32 %r30, %f72;
    shfl.sync.bfly.b32 %r30, %r30, 2, 0x1f, 0xffffffff; mov.b32 %f73, %r30; add.f32 %f72, %f72, %f73;
    mov.b32 %r30, %f72;
    shfl.sync.bfly.b32 %r30, %r30, 1, 0x1f, 0xffffffff; mov.b32 %f73, %r30; add.f32 %f72, %f72, %f73;
    mul.f32 %f72, %f72, %f14;

    // PRIMARY online softmax update
    max.f32 %f30, %f10, %f22;
    sub.f32 %f32, %f10, %f30; mul.f32 %f32, %f32, %f40; ex2.approx.f32 %f32, %f32;
    sub.f32 %f33, %f22, %f30; mul.f32 %f33, %f33, %f40; ex2.approx.f32 %f33, %f33;
    mul.f32 %f11, %f11, %f32; add.f32 %f11, %f11, %f33;
    mul.f32 %f12, %f12, %f32; mul.f32 %f13, %f13, %f32;

    // DIFF online softmax update
    max.f32 %f80, %f60, %f72;
    sub.f32 %f82, %f60, %f80; mul.f32 %f82, %f82, %f40; ex2.approx.f32 %f82, %f82;
    sub.f32 %f83, %f72, %f80; mul.f32 %f83, %f83, %f40; ex2.approx.f32 %f83, %f83;
    mul.f32 %f61, %f61, %f82; add.f32 %f61, %f61, %f83;
    mul.f32 %f62, %f62, %f82; mul.f32 %f63, %f63, %f82;

    // Load V (shared between primary and diff - V[:,:,h,:] == V[:,:,h+H,:])
    ld.param.u64 %rd8, [V_ptr];
    add.u64 %rd9, %rd8, %rd5;
    ld.global.f32 %f34, [%rd9];
    ld.global.f32 %f35, [%rd9 + 4];

    // PRIMARY accumulate
    fma.rn.f32 %f12, %f33, %f34, %f12;
    fma.rn.f32 %f13, %f33, %f35, %f13;

    // DIFF accumulate
    fma.rn.f32 %f62, %f83, %f34, %f62;
    fma.rn.f32 %f63, %f83, %f35, %f63;

    mov.f32 %f10, %f30;
    mov.f32 %f60, %f80;

    SKIP_{wi}:
"""

    ptx += f"""
    // Normalize and SUBTRACT in FP32
    rcp.approx.f32 %f50, %f11;
    mul.f32 %f12, %f12, %f50;
    mul.f32 %f13, %f13, %f50;

    rcp.approx.f32 %f90, %f61;
    mul.f32 %f62, %f62, %f90;
    mul.f32 %f63, %f63, %f90;

    // DIFFERENTIAL: primary - diff (in FP32!)
    sub.f32 %f12, %f12, %f62;
    sub.f32 %f13, %f13, %f63;

    // Store to OUTPUT: (B, N, H, D) layout using output strides
    mul.lo.u32 %r60, %r0, %r50;      // n * stride_n_out
    mad.lo.u32 %r61, %r1, %r51, %r60; // + h * stride_h_out
    add.u32 %r62, %r61, %r9;          // + lane*2
    cvt.u64.u32 %rd10, %r62;
    shl.b64 %rd11, %rd10, {elem_shift};
    ld.param.u64 %rd12, [Out_ptr];
    add.u64 %rd13, %rd12, %rd11;
    st.global.f32 [%rd13], %f12;
    st.global.f32 [%rd13 + 4], %f13;

    DONE:
    ret;
}}
"""
    return kernel_name, ptx


if __name__ == "__main__":
    name, ptx = generate_diff_swa_ptx()
    print(f"Kernel: {name}, PTX: {len(ptx)} bytes, shared: {'.shared' in ptx}")
    import subprocess
    with open('/tmp/diff_swa.ptx', 'w') as f: f.write(ptx)
    r = subprocess.run(['ptxas', '--gpu-name=sm_90', '-o', '/tmp/diff_swa.cubin', '/tmp/diff_swa.ptx'],
                       capture_output=True, text=True)
    print(f"ptxas: {'OK' if r.returncode == 0 else 'FAILED'}")
    if r.returncode != 0: print(r.stderr[:300])
