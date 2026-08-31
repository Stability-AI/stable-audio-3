"""Export a fixed-size SAME-S enc or dec to tflite (argv: which={enc,dec} size). SAME-S is block-local
(O(L)) so no windowing surgery — just trace at the fixed size. Reads weights from / writes rungs to
$SA3_BUILD_WORK.  enc: audio[1,2,S*4096]->lat[1,256,S]   dec: lat[1,256,S]->audio[1,2,S*4096]."""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_paths import WORK
import numpy as np, torch

which, S = sys.argv[1], int(sys.argv[2])
if which == "enc":
    import same_s_encoder_torch as M
    model = M.load_model(weights_path=str(WORK / "same_s_encoder_f32.npz"), summ_idx=16).eval()
    sample = torch.randn(1, 2, S * 4096) * 0.1
    path = str(WORK / f"same-s_enc_fixed_{S}.tflite")
else:
    import same_s_decoder_torch as M
    model = M.load_model(weights_path=str(WORK / "same_s_decoder_f32.npz"), output_audio=True).eval()
    sample = torch.randn(1, 256, S) * 1.0
    path = str(WORK / f"same-s_dec_fixed_{S}.tflite")
with torch.no_grad():
    y_torch = model(sample).numpy()
import ai_edge_torch
t0 = time.time()
edge = ai_edge_torch.convert(model.eval(), (sample,))
edge.export(path)
from ai_edge_litert import interpreter as tfl
it = tfl.Interpreter(model_path=path, num_threads=16); it.allocate_tensors()
i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
it.set_tensor(i, sample.numpy()); it.invoke(); y = it.get_tensor(o)
c = float((y.ravel() @ y_torch.ravel()) / (np.linalg.norm(y) * np.linalg.norm(y_torch) + 1e-9))
print(f"{which} S={S} exported ({os.path.getsize(path)/1e6:.0f} MB, {time.time()-t0:.0f}s) cos={c:.6f}", flush=True)
