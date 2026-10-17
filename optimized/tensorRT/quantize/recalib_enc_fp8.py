#!/usr/bin/env python3
"""Recompute the SAME-L fp8 ENCODER activation scales with AMAX.

WHY. enc_quant_samel.py set each activation quantiser's scale to

    max(percentile(p99.9_per_calibration_clip, 90) / 448, 1e-4)

— the 99.9th percentile within each clip, then the 90th percentile of *that* across clips, then a
floor. Three tail-discarding steps on a heavy-tailed distribution. The decoder's own quantiser
(fp8_gptq_samel.py) used plain `amax`, and the two disagree wildly:

    decoder   clip point = scale*448 = 22.4 on all 24 quantisers (one global amax)
    encoder   clip point = 0.04 .. 1.22, per layer, with every ff_out pinned to the 1e-4 floor

Measured effect: the encoder's latents landed 18.3 dB from the fp16 reference where the decoder's
output landed 28. End to end that cost 0.15-2.16 dB of round trip on real music — worst on clean
acoustic material (jazz), where quantisation noise has nothing to hide behind, and least on loud
clipped sources where the limiter masks it.

WHAT THIS DOES. Re-runs the SAME capture — same 6 calibration clips, same hook points — with
`scale = amax / 448`, and rewrites only the 24 activation-scale initializers. The GPTQ'd fp8
WEIGHTS are untouched, so this is a scale correction rather than a requantisation. Measured: the
old scales were 1.3-28.4x too small (median 3.0x); after the fix latent SNR is 30.9 dB and the
round trip equals fp16 on every source tested.

⚠ The scale initializers are fp16. Writing them back as fp32 makes the DequantizeLinear output
FLOAT while the matching weight stays HALF, and TRT rejects the MatMul with "must have same input
types". Preserve the dtype.

usage: recalib_enc_fp8.py --src enc_samel_fp8.onnx --out enc_fp8_amax.onnx \
                          --ckpt-dir /path/to/SAME-L --calib calib_audio_samel.npz
"""
import argparse
import json
import re

import numpy as np
import onnx
import torch
from onnx import numpy_helper as nh

E4M3_MAX = 448.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the fp8 encoder ONNX to correct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="dir holding SAME-L.json + SAME-L.ckpt")
    ap.add_argument("--calib", required=True, help="calib_audio_samel.npz (key 'audio')")
    a = ap.parse_args()

    from stable_audio_3.factory import create_autoencoder_from_config
    from stable_audio_3.loading_utils import copy_state_dict
    torch.set_grad_enabled(False)

    cfg = json.load(open(f"{a.ckpt_dir}/SAME-L.json"))
    ae = create_autoencoder_from_config(cfg["model"], cfg["sample_rate"])
    ck = torch.load(f"{a.ckpt_dir}/SAME-L.ckpt", map_location="cpu", weights_only=False)
    copy_state_dict(ae, ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
    ae = ae.cuda().eval()

    eproj = [m for nm, m in ae.named_modules() if re.search(r"encoder\..*\.ff\.ff\.0\.proj$", nm)]
    eout = [m for nm, m in ae.named_modules() if re.search(r"encoder\..*\.ff\.ff\.2$", nm)]
    print(f"eager encoder FFN: {len(eproj)} proj + {len(eout)} out", flush=True)

    amax = {}

    def mk(role, i):
        def hook(_m, inp):
            amax[(role, i)] = max(amax.get((role, i), 0.0), float(inp[0].detach().abs().max()))
        return hook

    handles = ([m.register_forward_pre_hook(mk("proj", i)) for i, m in enumerate(eproj)] +
               [m.register_forward_pre_hook(mk("out", i)) for i, m in enumerate(eout)])
    aud = np.load(a.calib)["audio"].astype(np.float32)
    for ci in range(aud.shape[0]):
        ae.encode(torch.tensor(aud[ci:ci + 1], device="cuda"))
        print(f"  captured clip {ci}", flush=True)
    for h in handles:
        h.remove()
    del ae
    torch.cuda.empty_cache()

    new = {k: float(v) / E4M3_MAX for k, v in amax.items()}
    model = onnx.load(a.src, load_external_data=True)
    inits = {i.name: i for i in model.graph.initializer}
    qa = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits and n.input[1] in inits:
            m = re.match(r"blocks\.(\d+)_ff_(proj|out)_MatMul_Qa", n.name or "")
            if not m:
                raise SystemExit(f"unrecognised activation quantiser name: {n.name!r}")
            qa[(m.group(2), int(m.group(1)))] = n.input[1]
    if set(qa) != set(new):
        raise SystemExit(f"mapping mismatch: onnx has {len(qa)}, eager captured {len(new)}")

    ratios = []
    for key, init_name in qa.items():
        init = inits[init_name]
        arr = nh.to_array(init)
        old = float(np.asarray(arr).ravel()[0])
        ratios.append(new[key] / old)
        # dtype preserved on purpose — see the module docstring.
        init.CopyFrom(nh.from_array(np.full(arr.shape, new[key], dtype=arr.dtype), init_name))
    r = np.array(ratios)
    print(f"\n  {len(qa)} activation scales rewritten with amax/448")
    print(f"  amax/old ratio: min {r.min():.1f}x  median {np.median(r):.1f}x  max {r.max():.1f}x")
    onnx.save(model, a.out, save_as_external_data=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
