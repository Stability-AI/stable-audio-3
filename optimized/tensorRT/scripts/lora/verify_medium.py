"""Ground-truth gate for the SA3-medium branch engine against a REAL underfit adapter.

Compares three things on identical inputs:
  ref169  canonical load_and_apply_loras() -- all 169 adapted layers, incl. the one
          conditioner linear (conditioners.seconds_total.embedder.embedding.1)
  ref168  same, but the conditioner linear left at base -- what a branch engine whose
          seconds-embed chain is a baked constant can actually reach
  trt     the branch engine + RefoldLora fold of the norm-baked adapter

ref169 vs ref168 prices the one layer the engine cannot serve, as a fraction of the whole
adapter's effect (ref169 vs base). trt vs ref168 is the wiring check: it validates the fold,
the sign of a negative DoRA magnitude, the frozen SVD bases, srow and pout at once.
"""
import json, argparse, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
# stable_audio_3 / stable_audio_tools provide the TORCH reference this harness compares
# the engine against; install them in the same venv.
import constants as LC
import dit_loader as DL
import paths


def snr_db(ref, got):
    ref, got = ref.float(), got.float()
    n = (ref - got).pow(2).mean()
    return float('inf') if n == 0 else 10 * torch.log10(ref.pow(2).mean() / n).item()


def rel(ref, got):
    return float((ref.float() - got.float()).norm() / ref.float().norm())


def _register_t5_alias():
    """medium's config names stabilityai/t5gemma-b-b-ul2; SAT's allowlist has google/... .

    Same architecture and the same 768 dims -- and this harness feeds t5_hidden in directly,
    so the text encoder is never run. Only the allowlist assert is in the way.
    """
    from stable_audio_tools.models.conditioners import T5GemmaConditioner as C
    for name in ("stabilityai/t5gemma-b-b-ul2",):
        if name not in C.T5GEMMA_MODELS:
            C.T5GEMMA_MODELS.append(name)
            C.T5GEMMA_MODEL_DIMS[name] = 768


def load_medium(variant="base"):
    """The cond-baked DiT wrapper. Same wrapper => same graph the engine was traced from."""
    DL.CKPT, DL.CONFIG = str(paths.checkpoint(variant)), str(paths.config(variant))
    DL.patch_for_onnx()
    _register_t5_alias()
    return DL.load_model(dtype=torch.float32), variant


def wrapper_with(dit, sd, seconds_w, seconds_b):
    ct = {"padding_embedding": sd[DL.COND_KEYS["padding_embedding"]].clone(),
          "seconds_weight": seconds_w.detach().clone(),
          "seconds_bias": seconds_b.detach().clone()}
    return DL.DiTWithCond(dit, ct).to("cuda").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("ADAPTER", "BAKED"),
                    default=[], help="repeatable; the torch load is shared across all pairs")
    ap.add_argument("--adapter"); ap.add_argument("--baked")
    # A manifest pins {source checkpoint -> baked file} together. Assembling the pair by
    # hand is how a bake of step=15000 got compared against a reference at step=9000:
    # `ls | sort | tail -1` orders step=9000 after step=15000.
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--L", type=int, default=1292)
    ap.add_argument("--strength", type=float, default=1.0)
    # The engine's own weights, not a guess: the medium branch engine is built from the
    # HF-published onnx/sa3-m/dit_fp16.onnx, which is ARC.
    ap.add_argument("--variant", default=paths.ENGINE_VARIANT,
                    help="which weights to compare the engine to")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--branch-map", dest="branch_map", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pairs = list(a.pair) + ([[a.adapter, a.baked]] if a.adapter else [])
    if a.manifest:
        man = json.load(open(a.manifest))
        pairs += [[d["src"], d["baked"]] for d in man.values()]
    assert pairs, "give at least one --pair ADAPTER BAKED"

    map_path = a.branch_map or paths.BRANCH_MAP

    torch.manual_seed(0)
    L = a.L
    x = torch.randn(1, LC.IO_CHANNELS, L, device="cuda")
    t = torch.tensor([0.7], device="cuda")
    t5 = torch.randn(1, LC.T5_TOKENS, LC.T5_HIDDEN_DIM, device="cuda") * 0.5
    mask = torch.ones(1, LC.T5_TOKENS, device="cuda"); mask[:, 200:] = 0
    sec = torch.tensor([120.0], device="cuda")
    lac = torch.zeros(1, LC.LOCAL_ADD_COND_DIM, L, device="cuda")
    args6 = (x, t, t5, mask, sec, lac)

    (model, cfg, sd), _variant = load_medium(a.variant)
    cond_lin = model.conditioner.conditioners["seconds_total"].embedder.embedding[1]
    w0, b0 = cond_lin.weight.detach().clone(), cond_lin.bias.detach().clone()

    with torch.no_grad():
        v_base = wrapper_with(model.model.model, sd, w0, b0)(*args6).clone()

    # ---- torch phase: every adapter, one model load ------------------------------------
    from stable_audio_3.models.lora.loader import load_and_apply_loras
    from stable_audio_3.models.lora.model import remove_lora
    rows = []
    for adapter, baked in pairs:
        load_and_apply_loras(model, [adapter], cfg["model_type"],
                             svd_bases_path=str(paths.SVD_BASES))
        if a.strength != 1.0:
            # lora_strength is a registered BUFFER, not a float attribute -- use the repo's
            # own setter rather than assigning through it.
            from stable_audio_3.models.lora.model import set_lora_strength
            set_lora_strength(model, a.strength)
        w_ad = cond_lin.weight.detach().clone()      # parametrised => adapted
        with torch.no_grad():
            v169 = wrapper_with(model.model.model, sd, w_ad, b0)(*args6).clone()
            v168 = wrapper_with(model.model.model, sd, w0,  b0)(*args6).clone()

        remove_lora(model)
        # Self-validating strip: leftover parametrisation would contaminate every later
        # adapter in the sweep with this one's delta, and nothing else would report it.
        with torch.no_grad():
            back = wrapper_with(model.model.model, sd, w0, b0)(*args6)
        resid = rel(v_base, back)
        assert resid < 1e-6, f"remove_lora left residue after {Path(adapter).name}: rel {resid:.3e}"

        eff = rel(v_base, v169)
        rows.append(dict(adapter=adapter, baked=baked, v169=v169, v168=v168, eff=eff,
                         cond=float((w_ad - w0).norm() / w0.norm()),
                         l169=rel(v169, v168), match=count_matches(adapter, map_path)))

    del model; torch.cuda.empty_cache()

    # ---- engine phase: one engine load, adapters swapped in place ----------------------
    from refold_runtime import RefoldLora
    B = RefoldLora(model="sa3-medium", engine_path=a.engine, branch_map=a.branch_map)
    B.zero(1)
    v_trt0 = run_engine(B, args6, L)
    nb = float(v_base.float().norm())
    floor = rel(v_base, v_trt0)
    print(f"\n  ── control: engine, NO adapter, vs torch {a.variant} ──")
    print(f"    trt(zero) vs base : {snr_db(v_base, v_trt0):+6.1f} dB  rel {floor:.3e}"
          f"   {'ok  <- this is the engine fp16 floor' if snr_db(v_base, v_trt0) > 30 else '<-- BASELINE MISMATCH'}")

    # What the engine can actually REACH. The 229-target map still carries a seconds-embedder
    # branch, but merge.EXCLUDED_LAYERS strips that layer from every adapter at load, so the
    # branch is fed zeros and the engine lands on v168 regardless of which map it was built
    # with. (Set merge.EXCLUDED_LAYERS = () to measure the old behaviour.)
    import merge as _M
    _excl = set(_M.EXCLUDED_LAYERS)
    n169 = any(r.get("adapter_key", "").startswith("conditioners.") or "seconds" in nm
               for nm, r in B.map.items()
               if (r.get("adapter_key") or r["ckpt_suffix"]) not in _excl)
    print(f"\n  ── {B.engine_path.name} ({len(B.map)} branch targets, "
          f"seconds embedder {'BRANCHED' if n169 else 'baked constant'}), "
          f"L={L}, strength {a.strength} ──")
    hdr = (f"    {'adapter':34s} {'rank':>4s} {'match':>7s} {'effect':>9s} {'L169':>7s} "
           f"{'fold err':>11s} {'delta err':>10s} {'168 only':>9s} {'vs canon':>9s} {'swap ms':>8s}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for r in rows:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        B.set_stack_gpu([(r["baked"], a.strength)], {})
        torch.cuda.synchronize(); swap_ms = (time.perf_counter() - t0) * 1e3
        v_trt = run_engine(B, args6, L)
        reach = r["v169"] if n169 else r["v168"]      # what this engine can actually reach

        # Split the error: zeroing the seconds-embedder operands makes this engine compute
        # exactly what the 168-target engine computes, so the remainder is attributable to
        # that one branch -- the only target in an fp32 island fed by fp16 operands.
        sec = [nm for nm in B.map if "seconds" in nm]
        if sec:
            rec = B.map[sec[0]]
            keep = {k: B.bufs[rec[k]].clone() for k in ("A", "B", "P")}
            for k in ("A", "B", "P"):
                B.bufs[rec[k]].zero_()
            r["fold_168_only"] = rel(r["v168"], run_engine(B, args6, L))
            for k in ("A", "B", "P"):
                B.bufs[rec[k]].copy_(keep[k])
        else:
            r["fold_168_only"] = float("nan")

        r["fold"] = rel(reach, v_trt)
        # The sharpest test of the branch: compare the DELTA the adapter produces, engine vs
        # torch. The fp16 base error is common to v_trt and v_trt0 and largely cancels, so a
        # weakly-trained adapter is still measured on its own contribution rather than on the
        # engine's precision.
        r["delta"] = rel(reach.float() - v_base.float(), v_trt.float() - v_trt0.float())
        # same denominator as the fp16 floor, so the two are directly comparable
        r["fold_common"] = float((reach.float() - v_trt.float()).norm() / nb)  # kept in json
        r["vs169"] = rel(r["v169"], v_trt)
        r["rank"] = B.rank; r["swap_ms"] = swap_ms
        # both sides relative to the signal each produces, so they are directly comparable
        flag = "" if r["fold"] <= 1.5 * floor else "  <-- ABOVE fp16 floor"
        print(f"    {Path(r['adapter']).name[:34]:34s} {B.rank:4d} "
              f"{r['match']['hit']:3d}/{r['match']['tot']:<3d} {r['eff']:9.3e} "
              f"{100*r['l169']/max(r['eff'],1e-30):6.2f}% {r['fold']:11.3e} "
              f"{r['delta']:10.3e} {r['fold_168_only']:9.3e} {r['vs169']:9.3e} "
              f"{swap_ms:8.1f}{flag}")
    print(f"\n    fp16 floor for reference: {floor:.3e}   "
          f"(every error above is relative to the signal it belongs to)")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "L": L, "strength": a.strength, "variant": a.variant, "engine": str(B.engine_path),
            "fp16_floor": floor,
            "rows": [{k: v for k, v in r.items() if k not in ("v168", "v169")} for r in rows],
        }, indent=1, default=str))


def count_matches(adapter, map_path=None):
    """How many of the adapter's layers the branch map actually addresses.

    A silent partial match (some layers folded, the rest left as zeros) looks exactly like a
    precision problem, so it is measured rather than assumed.
    """
    import lora_core as lc
    from branch_runtime import ADAPTER_PFX
    _, _, layers = lc.parse_adapter(str(adapter))
    mp = json.load(open(map_path or
                        paths.BRANCH_MAP))["layers"]
    from branch_runtime import adapter_key
    hit = sum(1 for rec in mp.values() if adapter_key(rec) in layers)
    return {"hit": hit, "tot": len(mp), "pfx": ADAPTER_PFX,
            "example_adapter_key": sorted(layers)[0]}


def run_engine(B, args6, L):
    import tensorrt as trt
    eng = B.engine
    ctx = eng.create_execution_context()
    x, t, t5, mask, sec, lac = args6
    feed = {"x": x, "t": t, "t5_hidden": t5, "t5_mask": mask,
            "seconds_total": sec, "local_add_cond": lac}
    out = None
    for i in range(eng.num_io_tensors):
        nm = eng.get_tensor_name(i)
        if eng.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            v = feed.get(nm, B.bufs.get(nm))
            assert v is not None, f"unbound input {nm}"
            ctx.set_input_shape(nm, tuple(v.shape))
            ctx.set_tensor_address(nm, int(v.data_ptr()))
        else:
            dt = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                  trt.DataType.BF16: torch.bfloat16}[eng.get_tensor_dtype(nm)]
            out = torch.empty(tuple(ctx.get_tensor_shape(nm)), dtype=dt, device="cuda")
            ctx.set_tensor_address(nm, int(out.data_ptr()))
    s = torch.cuda.Stream()
    assert ctx.execute_async_v3(s.cuda_stream), "engine execution failed"
    s.synchronize()
    return out


if __name__ == "__main__":
    main()
