#!/usr/bin/env python3
"""Quality gate: audio-PSNR of the torch-free C++ AMX-BF16 SAME-L decode vs the fp32 torch
SAME-L decode, on (a) a groundtruth sm-music latent and (b) a real medium C++/DiT latent.
Cross-checks vs the TFLite same_l_dec_fp32.tflite (both are the SAME-L decoder).

Gate: SAME-L is quant-tolerant (weight-only int8 hits 42-46 dB) -> bf16-linears should clear >=45 dB.
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch

SA3 = "/weka2/cj/clod/q4/sa3-w4-cluster"
sys.path.insert(0, SA3)
sys.path.insert(0, os.path.join(SA3, "models", "defs"))
sys.path.insert(0, "/weka2/cj/clod/same_l_cpu_amx")
import same_l_decoder_torch as ML
from same_l_cpu_backend import SamelCPU

torch.set_num_threads(16)

WEIGHTS = os.path.join(SA3, "models", "mlx", "same_l_decoder_f32.npz")
GT_LATENTS = os.path.join(SA3, "scripts", "w4_results", "latents_sm-music_semeval_sets.npz")
MED_LATENTS = os.path.join(SA3, "scripts", "w4_results", "latents_medium_ho_fp32.npz")
TFLITE = os.path.join(SA3, "models", "tflite", "same_l_dec_fp32.tflite")


def unpatch(patches):
    if isinstance(patches, np.ndarray):
        patches = torch.from_numpy(patches)
    B, C, L = patches.shape
    x = patches.reshape(B, 2, 256, L).permute(0, 1, 3, 2).reshape(B, 2, L * 256)
    return x.numpy()


def psnr(ref, test):
    ref = np.asarray(ref, np.float64).ravel(); test = np.asarray(test, np.float64).ravel()
    n = min(ref.size, test.size); ref, test = ref[:n], test[:n]
    mse = np.mean((ref - test) ** 2)
    if mse <= 0: return float("inf")
    return 20.0 * np.log10(np.max(np.abs(ref)) / np.sqrt(mse))


def cos(ref, test):
    a = np.asarray(ref, np.float64).ravel(); b = np.asarray(test, np.float64).ravel()
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def get_gt_latent(L=320):
    d = np.load(GT_LATENTS)
    keys = sorted(int(k[6:]) for k in d.files if k.startswith("fp32_l"))
    return np.ascontiguousarray(d[f"fp32_l{keys[0]}"][:, :, :L].astype(np.float32))


def get_medium_latent(L=320):
    d = np.load(MED_LATENTS)
    return np.ascontiguousarray(d["l0"][:, :, :L].astype(np.float32))


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    print(f"loading torch fp32 SAME-L (dynamic T_lat) ...", flush=True)
    tm = ML.load_model(WEIGHTS, T_lat=None, dtype=torch.float32, output_audio=False).eval()
    m = SamelCPU(threads=16)

    tfl = None
    try:
        from ai_edge_litert import interpreter as _tfl
        it = _tfl.Interpreter(model_path=TFLITE, num_threads=8); it.allocate_tensors()
        tfl = it
    except Exception as e:
        print(f"(TFLite cross-check unavailable: {e})")

    def tfl_decode(lat):
        i = tfl.get_input_details()[0]["index"]; o = tfl.get_output_details()[0]["index"]
        Lc = lat.shape[-1]
        it.resize_tensor_input(i, [1, 256, Lc], strict=False); it.allocate_tensors()
        i = it.get_input_details()[0]["index"]; o = it.get_output_details()[0]["index"]
        it.set_tensor(i, np.ascontiguousarray(lat, np.float32)); it.invoke()
        return it.get_tensor(o)[0]  # [2, L*4096]

    cases = [("groundtruth sm-music", get_gt_latent(L)), ("medium C++/DiT", get_medium_latent(L))]
    print(f"\n{'case':22s} {'audio-s':>7} | {'C++whole patchdB':>16} {'C++whole audB':>13} {'C++chunk audB':>13} "
          f"{'cos(chunk)':>10} | {'TFL audB':>9}")
    for name, lat in cases:
        with torch.no_grad():
            ref_patch = tm(torch.from_numpy(lat)).numpy()
        ref_audio = unpatch(ref_patch)

        cpp_patch_whole = m.forward(lat)
        cpp_audio_whole = unpatch(cpp_patch_whole)
        cpp_patch_chunk = m.forward_chunked(lat, C=64, overlap=8, parallel=0)
        cpp_audio_chunk = unpatch(cpp_patch_chunk)

        dpp = psnr(ref_patch, cpp_patch_whole)
        daw = psnr(ref_audio, cpp_audio_whole)
        dac = psnr(ref_audio, cpp_audio_chunk)
        csc = cos(ref_audio, cpp_audio_chunk)

        if tfl is not None:
            tfl_audio = tfl_decode(lat)
            dtf = psnr(ref_audio, tfl_audio)
            s_tf = f"{dtf:9.2f}"
        else:
            s_tf = f"{'-':>9}"
        print(f"{name:22s} {L*4096/44100:>7.1f} | {dpp:16.2f} {daw:13.2f} {dac:13.2f} {csc:10.5f} | {s_tf}")


if __name__ == "__main__":
    main()
