#!/usr/bin/env python3
"""Quality gate for the FUSED INT8 (w8a8) C++ SAME-L engine: audio-PSNR vs the fp32 torch SAME-L
decode, side-by-side with the NAIVE int8 engine (must match: same requant grid) and the bf16 anchor.
The fused-vs-naive PSNR proves fusion is a pure speed/memory optimization.

usage: python validate.py [L]      # default 320
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch

SA3 = "/weka2/cj/clod/q4/sa3-w4-cluster"
sys.path.insert(0, SA3)
sys.path.insert(0, os.path.join(SA3, "models", "defs"))
sys.path.insert(0, "/weka2/cj/clod/same_l_int8fused_cpu_amx")
sys.path.insert(0, "/weka2/cj/clod/same_l_int8_cpu_amx")
sys.path.insert(0, "/weka2/cj/clod/same_l_cpu_amx")
import same_l_decoder_torch as ML
from same_l_int8fused_backend import SamelInt8FusedCPU
from same_l_int8_backend import SamelInt8CPU

torch.set_num_threads(16)
WEIGHTS = os.path.join(SA3, "models", "mlx", "same_l_decoder_f32.npz")
GT_LATENTS = os.path.join(SA3, "scripts", "w4_results", "latents_sm-music_semeval_sets.npz")
MED_LATENTS = os.path.join(SA3, "scripts", "w4_results", "latents_medium_ho_fp32.npz")


def unpatch(patches):
    if isinstance(patches, np.ndarray):
        patches = torch.from_numpy(patches)
    B, C, L = patches.shape
    return patches.reshape(B, 2, 256, L).permute(0, 1, 3, 2).reshape(B, 2, L * 256).numpy()


def psnr(ref, test):
    ref = np.asarray(ref, np.float64).ravel(); test = np.asarray(test, np.float64).ravel()
    n = min(ref.size, test.size); ref, test = ref[:n], test[:n]
    mse = np.mean((ref - test) ** 2)
    return float("inf") if mse <= 0 else 20.0 * np.log10(np.max(np.abs(ref)) / np.sqrt(mse))


def cos(ref, test):
    a = np.asarray(ref, np.float64).ravel(); b = np.asarray(test, np.float64).ravel()
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def get_gt(L):
    d = np.load(GT_LATENTS)
    keys = sorted(int(k[6:]) for k in d.files if k.startswith("fp32_l"))
    return np.ascontiguousarray(d[f"fp32_l{keys[0]}"][:, :, :L].astype(np.float32))


def get_med(L):
    d = np.load(MED_LATENTS)
    return np.ascontiguousarray(d["l0"][:, :, :L].astype(np.float32))


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 320
    tm = ML.load_model(WEIGHTS, T_lat=None, dtype=torch.float32, output_audio=False).eval()
    mf = SamelInt8FusedCPU(threads=16)
    mn = SamelInt8CPU(threads=16)
    try:
        from same_l_cpu_backend import SamelCPU
        mbf = SamelCPU(threads=16)
    except Exception as e:
        mbf = None; print("(bf16 engine unavailable:", e, ")")

    print(f"\n{'case':22s} {'aud-s':>6} | {'fused dB':>9} {'naive dB':>9} {'bf16 dB':>9} | "
          f"{'fused-chunk dB':>14} {'fused-vs-naive':>14} {'cos':>8}")
    for name, lat in [("groundtruth sm-music", get_gt(L)), ("medium C++/DiT", get_med(L))]:
        with torch.no_grad():
            refa = unpatch(tm(torch.from_numpy(lat)).numpy())
        af = unpatch(mf.forward(lat))
        an = unpatch(mn.forward(lat))
        afc = unpatch(mf.forward_chunked(lat, C=64, overlap=8, parallel=0))
        d_f, d_n, d_fc = psnr(refa, af), psnr(refa, an), psnr(refa, afc)
        d_fn, cc = psnr(an, af), cos(refa, af)
        s_bf = f"{psnr(refa, unpatch(mbf.forward(lat))):9.2f}" if mbf is not None else f"{'-':>9}"
        print(f"{name:22s} {L*4096/44100:>6.1f} | {d_f:9.2f} {d_n:9.2f} {s_bf} | "
              f"{d_fc:14.2f} {d_fn:14.2f} {cc:8.5f}")


if __name__ == "__main__":
    main()
