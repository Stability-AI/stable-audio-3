"""Extract SAME-S DECODER weights from the sa3-sm-music checkpoint (torch pickle) -> flat .npz.
Same structure as SAME-L decoder but dim=768, 6 blocks; OUTPUT mapping is WNConv1d(768->512, k=3).
Output keys match torch_defs/same_s_decoder_torch.SAMESDecoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from build_paths import WORK, ckpt

CKPT = ckpt("sm-music")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "same_s_decoder_f32.npz")
NUM_BLOCKS = 6
D = "pretransform.model.decoder"


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "module"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]; break
    g = lambda k: sd[k].float().numpy()
    out = {}
    out["project_in.weight"] = g(f"{D}.layers.1.weight")   # (768,256)
    out["project_in.bias"] = g(f"{D}.layers.1.bias")
    out["new_tokens"] = g(f"{D}.layers.3.new_tokens")      # (1,1,768)
    for i in range(NUM_BLOCKS):
        t = f"{D}.layers.3.transformers.{i}"
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
    wv = g(f"{D}.layers.3.mapping.weight_v")               # (512,768,3)
    wg = g(f"{D}.layers.3.mapping.weight_g")
    norm = np.sqrt((wv ** 2).sum(axis=(1, 2), keepdims=True))
    out["mapping.weight"] = (wg * wv / norm).astype(np.float32)
    out["mapping.bias"] = g(f"{D}.layers.3.mapping.bias")
    out["running_std"] = g("pretransform.model.bottleneck.running_std")
    np.savez(OUT, **out)
    n = sum(v.size for v in out.values())
    print(f"wrote {OUT}  ({n:,} params, {len(out)} keys)")


if __name__ == "__main__":
    main()
