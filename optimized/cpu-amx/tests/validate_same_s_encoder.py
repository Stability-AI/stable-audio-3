#!/usr/bin/env python3
"""Quality gate for the torch-free C++ AMX-BF16 SAME-S encoder.

GATE: per-token cosine >= 0.999 and PSNR >= 40 dB vs the TFLite fp32 latent, on >=3 real
clips. Plus: numpy-ref cross-check, and a ROUND-TRIP (encoder latent -> C++ SAME-S decoder
-> audio; finite, not-silent, resembles input).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "16")

DIR = "/weka2/cj/clod/same_s_encoder_cpu_amx"
sys.path.insert(0, DIR)
sys.path.insert(0, "/weka2/cj/clod/same_s_cpu_amx")   # C++ decoder for round-trip

from _enc_valcommon import run_full_gate
from same_s_encoder_backend import SamesEncoderCPU
from same_s_encoder_numpy import SamesEncNumpy

TFL = os.path.join(DIR, "hf/tflite/same-s/enc_fp32.tflite")


def main():
    enc = SamesEncoderCPU(threads=16)
    npref = SamesEncNumpy()
    try:
        from same_s_cpu_backend import SamesCPU
        dec = SamesCPU(threads=16)
    except Exception as e:
        print(f"(round-trip decoder unavailable: {e})")
        dec = None
    ok = run_full_gate("SAME-S", enc, TFL, decoder=dec, npmodel=npref, k=3)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
