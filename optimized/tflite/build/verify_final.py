"""Structural verify of the 8 built rung files (runtime env: ai_edge_litert >= 2.2.0). Loads each,
prints the discovered rung ladder, and checks it dispatches + produces correct shapes at a small exact
rung and a tiling length. No ground-truth needed — round-trip QUALITY is a separate check on real audio.
Reads from $SA3_BUILD_WORK/same-{l,s}/."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
HERE = Path(__file__).resolve().parent                 # build/
sys.path.insert(0, str(HERE))                          # build_paths
for cand in (HERE.parent / "scripts", HERE.parent / "runtime"):   # runtime: repo=../scripts, staging=../runtime
    if (cand / "rung_encoder.py").exists():
        sys.path.insert(0, str(cand)); break
import numpy as np
from build_paths import WORK
from rung_encoder import RungEncoder
from rung_decoder import RungDecoder

TRIM = {"same-l": 12, "same-s": 16}
ok = True
for size in ("same-l", "same-s"):
    for kind in ("enc", "dec"):
        for prec in ("fp32", "w8a8"):
            p = WORK / size / f"{kind}_{prec}.tflite"
            if not p.exists():
                print(f"  MISSING {p}"); ok = False; continue
            m = (RungEncoder if kind == "enc" else RungDecoder)(str(p), threads=8, trim=TRIM[size])
            for L in (2, 100):                          # exact tiny rung + a tiling length (both even)
                if kind == "enc":
                    y = m.encode(np.zeros((1, 2, L * 4096), np.float32)); good = y.shape == (1, 256, L)
                else:
                    y = m.decode(np.zeros((1, 256, L), np.float32)); good = y.shape[2] == L * 4096
                ok &= good
            print(f"  {size}/{kind}_{prec:4s}  rungs={m.sizes}  shapes OK={good}")
print("VERIFY OK" if ok else "VERIFY FAILED")
sys.exit(0 if ok else 1)
