"""Extract SAME-L ENCODER weights from the sa3-medium checkpoint -> flat .npz.

Encoder lives under `pretransform.model.encoder`:
  layers.0.mapping         WNConv1d(512->1536, k=1)   INPUT projection (patches->dim)  [fuse weight_norm]
  layers.0.new_tokens      (1,1,1536)                 single learnable summary token
  layers.0.transformers.N  12x {pre_norm(DyT), self_attn(5xQKV diff-SWA), ff_norm(DyT), ff(GLU)}
  layers.2                 Linear(1536->256)          OUTPUT projection (dim->latent)
  bottleneck.{running_std,scaling_factor,bias}        softnorm bottleneck params
Output keys match torch_defs/same_l_encoder_torch.SAMELEncoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from safetensors import safe_open
from build_paths import WORK, ckpt

CKPT = ckpt("medium")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "same_l_encoder_f32.npz")
NUM_BLOCKS = 12
E = "pretransform.model.encoder"
B = "pretransform.model.bottleneck"


def main():
    f = safe_open(CKPT, "pt")
    g = lambda k: f.get_tensor(k).float().numpy()
    out = {}
    wv = g(f"{E}.layers.0.mapping.weight_v")          # (1536,512,1)
    wg = g(f"{E}.layers.0.mapping.weight_g")          # (1536,1,1)
    norm = np.sqrt((wv ** 2).sum(axis=(1, 2), keepdims=True))
    out["mapping.weight"] = (wg * wv / norm).astype(np.float32)
    out["mapping.bias"] = g(f"{E}.layers.0.mapping.bias").astype(np.float32)
    out["new_tokens"] = g(f"{E}.layers.0.new_tokens").astype(np.float32)
    for i in range(NUM_BLOCKS):
        t = f"{E}.layers.0.transformers.{i}"
        out[f"blocks.{i}.pre_norm.alpha"] = g(f"{t}.pre_norm.alpha")
        out[f"blocks.{i}.pre_norm.gamma"] = g(f"{t}.pre_norm.gamma")
        out[f"blocks.{i}.pre_norm.beta"] = g(f"{t}.pre_norm.beta")
        out[f"blocks.{i}.attn.to_qkv.weight"] = g(f"{t}.self_attn.to_qkv.weight")
        out[f"blocks.{i}.attn.to_out.weight"] = g(f"{t}.self_attn.to_out.weight")
        for qk in ("q_norm", "k_norm"):
            out[f"blocks.{i}.attn.{qk}.alpha"] = g(f"{t}.self_attn.{qk}.alpha")
            out[f"blocks.{i}.attn.{qk}.gamma"] = g(f"{t}.self_attn.{qk}.gamma")
            out[f"blocks.{i}.attn.{qk}.beta"] = g(f"{t}.self_attn.{qk}.beta")
        out[f"blocks.{i}.ff_norm.alpha"] = g(f"{t}.ff_norm.alpha")
        out[f"blocks.{i}.ff_norm.gamma"] = g(f"{t}.ff_norm.gamma")
        out[f"blocks.{i}.ff_norm.beta"] = g(f"{t}.ff_norm.beta")
        out[f"blocks.{i}.ff.glu_proj.weight"] = g(f"{t}.ff.ff.0.proj.weight")
        out[f"blocks.{i}.ff.glu_proj.bias"] = g(f"{t}.ff.ff.0.proj.bias")
        out[f"blocks.{i}.ff.proj_out.weight"] = g(f"{t}.ff.ff.2.weight")
        out[f"blocks.{i}.ff.proj_out.bias"] = g(f"{t}.ff.ff.2.bias")
    out["project_out.weight"] = g(f"{E}.layers.2.weight")
    out["project_out.bias"] = g(f"{E}.layers.2.bias")
    out["bottleneck.running_std"] = g(f"{B}.running_std")
    out["bottleneck.scaling_factor"] = g(f"{B}.scaling_factor")
    out["bottleneck.bias"] = g(f"{B}.bias")
    np.savez(OUT, **out)
    n = sum(v.size for v in out.values())
    print(f"wrote {OUT}  ({n:,} params, {len(out)} keys)")


if __name__ == "__main__":
    main()
