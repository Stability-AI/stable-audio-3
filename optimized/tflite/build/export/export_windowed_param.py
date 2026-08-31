"""Export the WINDOWED SAME-L DECODER to tflite at a fixed rung size (argv T_LAT) via ai_edge_torch.
latent [1,256,T] -> audio [1,2,T*4096]. Reads weights from $SA3_BUILD_WORK, writes the fixed rung there."""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_paths import WORK                     # also puts torch_defs on sys.path
import numpy as np, torch
import same_l_decoder_torch as M
import windowed_decoder as W

T_LAT = int(sys.argv[1])
W.patch()                                         # O(S) windowed attention (== dense band mask)
model = M.load_model(weights_path=str(WORK / "same_l_decoder_f32.npz"), T_lat=None, output_audio=True).eval()
sample = torch.randn(1, 256, T_LAT)
with torch.no_grad():
    y_torch = model(sample).numpy()
import ai_edge_torch
t0 = time.time()
edge = ai_edge_torch.convert(model.eval(), (sample,))
path = str(WORK / f"same-l_windowed_{T_LAT}.tflite")
edge.export(path)
from ai_edge_litert import interpreter as tfl
it = tfl.Interpreter(model_path=path, num_threads=16); it.allocate_tensors()
i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
it.set_tensor(i, sample.numpy()); it.invoke(); y = it.get_tensor(o)
c = float((y.ravel() @ y_torch.ravel()) / (np.linalg.norm(y) * np.linalg.norm(y_torch) + 1e-9))
print(f"dec T_LAT={T_LAT} exported ({os.path.getsize(path)/1e6:.0f} MB, {time.time()-t0:.0f}s) cos={c:.6f}", flush=True)
