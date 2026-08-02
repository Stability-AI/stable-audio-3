#!/usr/bin/env python3
"""Quality gate for the torch-free C++ AMX-BF16 SAME-L encoder.

GATE: per-token cosine >= 0.999 and PSNR >= 40 dB vs the TFLite fp32 latent, on >=3 real
clips. Plus: numpy-ref cross-check, and a ROUND-TRIP (encoder latent -> C++ SAME-L decoder
-> audio; finite, not-silent, resembles input).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "16")

DIR = "/weka2/cj/clod/same_l_encoder_cpu_amx"
sys.path.insert(0, DIR)
sys.path.insert(0, "/weka2/cj/clod/same_l_cpu_amx")   # C++ decoder for round-trip

from _enc_valcommon import run_full_gate
from same_l_encoder_backend import SamelEncoderCPU
from same_l_encoder_numpy import SamelEncNumpy

TFL = os.path.join(DIR, "hf/tflite/same-l/enc_fp32.tflite")


def main():
    # precision: fp32 (default; clears the strict 0.999 gate) or bf16 (fast AMX path).
    # pass a 2nd arg "np" to add the (slow) numpy fp32 cross-check column.
    prec = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    use_np = len(sys.argv) > 2 and sys.argv[2] == "np"
    enc = SamelEncoderCPU(threads=16, precision=prec)
    npref = SamelEncNumpy() if use_np else None
    try:
        from same_l_cpu_backend import SamelCPU
        dec = SamelCPU(threads=16)
    except Exception as e:
        print(f"(round-trip decoder unavailable: {e})")
        dec = None
    ok = run_full_gate(f"SAME-L {prec}", enc, TFL, decoder=dec, npmodel=npref, k=3)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
