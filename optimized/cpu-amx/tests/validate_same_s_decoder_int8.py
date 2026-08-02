#!/usr/bin/env python3
"""Quality gate for the FUSED INT8 (w8a8) C++ SAME-S engine: audio-PSNR vs the fp32 torch decode,
side-by-side with the NAIVE int8 engine (must match: same requant grid) and the bf16 anchor.
The fused-vs-naive PSNR proves fusion is a pure speed/memory optimization (>=~80 dB = numerically
identical up to fp-contraction).

usage: python validate.py [L ...]          # default 320 1292
       python validate.py --medfile        # precomputed real medium C++/DiT latent
"""
import sys, os
sys.path.insert(0, "/weka2/cj/clod/same_s_bf16_amx")
sys.path.insert(0, "/weka2/cj/clod/same_s_int8fused_cpu_amx")
sys.path.insert(0, "/weka2/cj/clod/same_s_int8_cpu_amx")
sys.path.insert(0, "/weka2/cj/clod/same_s_cpu_amx")
import numpy as np
import torch
import ref_common as R
from same_s_int8fused_backend import SamesInt8FusedCPU
from same_s_int8_backend import SamesInt8CPU

torch.set_num_threads(16)


def run_lengths(Ls):
    mf = SamesInt8FusedCPU(threads=16)
    mn = SamesInt8CPU(threads=16)
    tm = R.load_torch_fp32(output_audio=False)
    try:
        from same_s_cpu_backend import SamesCPU
        mbf = SamesCPU(threads=16)
    except Exception as e:
        mbf = None
        print("(bf16 engine unavailable:", e, ")")

    print(f"{'L':>6} {'aud-s':>7} | {'fused dB':>9} {'naive dB':>9} {'bf16 dB':>9} | "
          f"{'fused-vs-naive':>14} {'fused cos':>10}")
    for L in Ls:
        lat = R.make_latent(L)
        with torch.no_grad():
            ref_patch = tm(torch.from_numpy(lat)).numpy()
        ref_audio = R.unpatch(ref_patch).numpy()

        af = R.unpatch(mf.forward(lat)).numpy()
        an = R.unpatch(mn.forward(lat)).numpy()
        d_f = R.psnr(ref_audio, af)
        d_n = R.psnr(ref_audio, an)
        d_fn = R.psnr(an, af)                 # fused vs naive (quality-neutrality proof)
        cs = R.cos(ref_audio, af)
        if mbf is not None:
            abf = R.unpatch(mbf.forward(lat)).numpy()
            d_bf = R.psnr(ref_audio, abf)
            s_bf = f"{d_bf:9.2f}"
        else:
            s_bf = f"{'-':>9}"
        print(f"{L:>6} {L*4096/44100:>7.1f} | {d_f:9.2f} {d_n:9.2f} {s_bf} | {d_fn:14.2f} {cs:10.5f}")


def run_medfile(L=320):
    MED = "/weka2/cj/clod/q4/sa3-w4-cluster/scripts/w4_results/latents_medium_ho_fp32.npz"
    lat = np.ascontiguousarray(np.load(MED)["l0"][:, :, :L].astype(np.float32))
    tm = R.load_torch_fp32(output_audio=False)
    with torch.no_grad():
        refa = R.unpatch(tm(torch.from_numpy(lat)).numpy()).numpy()
    mf = SamesInt8FusedCPU(threads=16)
    mn = SamesInt8CPU(threads=16)
    af = R.unpatch(mf.forward(lat)).numpy()
    an = R.unpatch(mn.forward(lat)).numpy()
    print(f"MEDIUM C++/DiT latent (L={L}): fused={R.psnr(refa, af):.2f} dB  naive={R.psnr(refa, an):.2f} dB  "
          f"fused-vs-naive={R.psnr(an, af):.2f} dB  cos={R.cos(refa, af):.5f}")


if __name__ == "__main__":
    if "--medfile" in sys.argv:
        run_medfile()
    else:
        Ls = [int(x) for x in sys.argv[1:] if x.isdigit()] or [320, 1292]
        run_lengths(Ls)
