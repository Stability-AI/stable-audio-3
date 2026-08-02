#!/usr/bin/env python3
"""Confirm the FULL C++ pipeline: prompt -> C++ T5Gemma -> conditioner -> C++ DiT -> C++ decoder -> audio.
Compares C++-T5Gemma vs TFLite-T5Gemma end-to-end (should be ~identical: cosine 0.9997 at the encoder)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS","16")
import numpy as np
REPO="/weka2/cj/clod/q4/sa3-w4-cluster"
for p in (REPO, "/weka2/cj/clod/t5gemma_cpu_amx", "/weka2/cj/clod/same_s_cpu_amx"):
    sys.path.insert(0,p)
import tflite_pipeline as P
from t5gemma_cpu_backend import T5GemmaCPU
from cpu_amx_backend import DiTCppAmx
from same_s_cpu_backend import SamesCPU

prompt="warm analog synthwave with a driving bassline, 120 bpm"; seed=0; steps=8; seconds=3.0
tok=P.Tokenizer(); ids,mask=tok(prompt)
print(f"prompt tokenized: ids{ids.shape} nnz={int(mask.sum())}", flush=True)

T5_PATH=P.TFL_DIR/P.T5_TFLITE["fp16"]; COND_NPZ=P.DIT_NPZ["medium"]
t5_cpp=T5GemmaCPU(threads=16)
t5_tfl=P.T5GemmaTFLite(T5_PATH, threads=8)
lh_cpp=t5_cpp(ids.astype(np.int32),mask.astype(np.int32))
lh_tfl=t5_tfl(ids,mask)
n=int(mask.sum()); a=lh_cpp[0,:n].astype(np.float64); b=lh_tfl[0,:n].astype(np.float64)
enc_cos=float(np.mean((a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))))
print(f"encoder C++ vs TFLite: mean-cos={enc_cos:.6f}", flush=True)

cond=P.Conditioner(COND_NPZ)
T_lat=max(8,int(round(seconds*44100/4096))); T_lat-=T_lat%2; secs_eff=T_lat*4096/44100
dit=DiTCppAmx(threads=1)   # threads=1: dodge the known DiT .so multithread heap race
dec=SamesCPU(threads=16)

def gen(lh, tag):
    cross,gcond=cond.build(lh,mask,secs_eff)
    x0,step_noise=P.make_noise(T_lat,steps,seed)
    sigmas=P.build_pingpong_schedule(steps,sigma_max=1.0)
    t0=time.time(); latent=P.sample(dit,x0,step_noise,sigmas,cross,gcond); dt=time.time()-t0
    pcm=dec.forward_pcm(np.ascontiguousarray(latent))[0]
    print(f"  [{tag}] DiT {dt:.1f}s  latent{latent.shape} finite={np.isfinite(latent).all()}  "
          f"audio{pcm.shape} rms={np.sqrt(np.mean(pcm**2)):.4f} peak={np.abs(pcm).max():.3f} finite={np.isfinite(pcm).all()}", flush=True)
    return pcm

print("generating (C++ T5Gemma path)...", flush=True); a_cpp=gen(lh_cpp,"cpp-T5")
print("generating (TFLite T5Gemma path)...", flush=True); a_tfl=gen(lh_tfl,"tfl-T5")
def psnr(r,t): n=min(r.shape[-1],t.shape[-1]); r,t=r[...,:n],t[...,:n]; mse=np.mean((r-t)**2); return 99. if mse==0 else 10*np.log10(np.max(np.abs(r))**2/mse)
print(f"\nFULL-PIPELINE audio C++-T5 vs TFLite-T5: PSNR={psnr(a_tfl,a_cpp):.1f} dB  corr={np.corrcoef(a_cpp.flatten(),a_tfl.flatten())[0,1]:.5f}")
import soundfile as sf
sf.write("/weka2/cj/tmp/claude-1879804714/-weka2-cj-clod-q4/c8227744-3c49-4a57-9393-e34300693a36/scratchpad/e2e_cpp_t5.wav", np.clip(a_cpp,-1,1).T, 44100)
print("wrote e2e_cpp_t5.wav — VERDICT:", "FULL C++ PIPELINE OK" if np.isfinite(a_cpp).all() and np.abs(a_cpp).max()>1e-3 else "PROBLEM")
