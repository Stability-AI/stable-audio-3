"""Extract SAME-S ENCODER weights from the sa3-sm-music checkpoint (torch pickle) -> flat .npz.
Same structure as SAME-L encoder but dim=768, 6 blocks, FF inner 2304, mapping k=1.
Output keys match torch_defs/same_s_encoder_torch.SAMESEncoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from build_paths import WORK, ckpt

CKPT = ckpt("sm-music")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "same_s_encoder_f32.npz")
NUM_BLOCKS = 6
E = "pretransform.model.encoder"
B = "pretransform.model.bottleneck"


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "module"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]; break
    g = lambda k: sd[k].float().numpy()
    out = {}
    wv = g(f"{E}.layers.0.mapping.weight_v")          # (768,512,1)
    wg = g(f"{E}.layers.0.mapping.weight_g")
    norm = np.sqrt((wv ** 2).sum(axis=(1, 2), keepdims=True))
    out["mapping.weight"] = (wg * wv / norm).astype(np.float32)
    out["mapping.bias"] = g(f"{E}.layers.0.mapping.bias").astype(np.float32)
    out["new_tokens"] = g(f"{E}.layers.0.new_tokens").astype(np.float32)
    for i in range(NUM_BLOCKS):
        t = f"{E}.layers.0.transformers.{i}"
        out[f"blocks.{i}.pre_norm.alpha"] = g(f"{t}.pre_norm.alpha")
        out[f"blocks.{i}.pre_norm.gamma"] = g(f"{t}.pre_norm.gamma")
        out[f"blocks.{i}.pre_norm.beta"] = g(f"{t}.pre_norm.beta")
        out[f"blocks.{i}.attn.to_qkv.weight"] = g(f"{t}.self_attn.to_qkv.weight")   # (3840,768)
        out[f"blocks.{i}.attn.to_out.weight"] = g(f"{t}.self_attn.to_out.weight")
        for qk in ("q_norm", "k_norm"):
            out[f"blocks.{i}.attn.{qk}.alpha"] = g(f"{t}.self_attn.{qk}.alpha")
            out[f"blocks.{i}.attn.{qk}.gamma"] = g(f"{t}.self_attn.{qk}.gamma")
            out[f"blocks.{i}.attn.{qk}.beta"] = g(f"{t}.self_attn.{qk}.beta")
        out[f"blocks.{i}.ff_norm.alpha"] = g(f"{t}.ff_norm.alpha")
        out[f"blocks.{i}.ff_norm.gamma"] = g(f"{t}.ff_norm.gamma")
        out[f"blocks.{i}.ff_norm.beta"] = g(f"{t}.ff_norm.beta")
        out[f"blocks.{i}.ff.glu_proj.weight"] = g(f"{t}.ff.ff.0.proj.weight")       # (4608,768)
        out[f"blocks.{i}.ff.glu_proj.bias"] = g(f"{t}.ff.ff.0.proj.bias")
        out[f"blocks.{i}.ff.proj_out.weight"] = g(f"{t}.ff.ff.2.weight")            # (768,2304)
        out[f"blocks.{i}.ff.proj_out.bias"] = g(f"{t}.ff.ff.2.bias")
    out["project_out.weight"] = g(f"{E}.layers.2.weight")   # (256,768)
    out["project_out.bias"] = g(f"{E}.layers.2.bias")
    out["bottleneck.running_std"] = g(f"{B}.running_std")
    out["bottleneck.scaling_factor"] = g(f"{B}.scaling_factor")
    out["bottleneck.bias"] = g(f"{B}.bias")
    np.savez(OUT, **out)
    n = sum(v.size for v in out.values())
    print(f"wrote {OUT}  ({n:,} params, {len(out)} keys)")


if __name__ == "__main__":
    main()
