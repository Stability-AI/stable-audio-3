"""Verify ARBITRARY adapter stacks: mixed families, mixed ranks, against canonical torch.

A stack concatenates along the rank axis, so the fold has to get every adapter's block, its
scaling, and its DoRA renormalisation right SIMULTANEOUSLY -- a single-adapter test cannot
catch a block-offset error. Compares the engine to load_and_apply_loras() with the same list.
"""
import json, argparse, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
import constants as LC
import paths
from verify_medium import (rel, load_medium, wrapper_with,
                           run_engine)

AD = ROOT / "lora" / "adapters" / "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", action="append", required=True,
                    help="comma-separated adapter files (relative to adapters/medium)")
    ap.add_argument("--L", type=int, default=1292)
    ap.add_argument("--engine", default=None); ap.add_argument("--branch-map", dest="bmap", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    stacks = [[s.strip() for s in spec.split(",")] for spec in a.stack]

    torch.manual_seed(0); L = a.L
    x = torch.randn(1, LC.IO_CHANNELS, L, device="cuda")
    t = torch.tensor([0.7], device="cuda")
    t5 = torch.randn(1, LC.T5_TOKENS, LC.T5_HIDDEN_DIM, device="cuda") * 0.5
    mask = torch.ones(1, LC.T5_TOKENS, device="cuda"); mask[:, 200:] = 0
    sec = torch.tensor([120.0], device="cuda")
    lac = torch.zeros(1, LC.LOCAL_ADD_COND_DIM, L, device="cuda")
    args6 = (x, t, t5, mask, sec, lac)

    (model, cfg, sd), _variant = load_medium(paths.ENGINE_VARIANT)
    cond = model.conditioner.conditioners["seconds_total"].embedder.embedding[1]
    w0, b0 = cond.weight.detach().clone(), cond.bias.detach().clone()
    with torch.no_grad():
        v_base = wrapper_with(model.model.model, sd, w0, b0)(*args6).clone()

    from stable_audio_3.models.lora.loader import load_and_apply_loras
    from stable_audio_3.models.lora.model import remove_lora
    import merge as M

    # ── additive reference ────────────────────────────────────────────────────────────────
    # The branch engine SUMS each adapter's delta against the same W0 (merge_loras_into_base_
    # _model semantics); load_and_apply_loras CHAINS. For a stack of drifted DoRAs those two
    # differ by more than the adapters' own effect, so comparing the engine to the chained
    # reference measures the semantics choice and hides any real wiring error underneath it.
    # Build the summed reference by writing the merged weights straight into the DiT.
    dit = model.model.model
    mods = dict(dit.named_modules())

    def _mod(layer_key):
        k = layer_key[len("model."):] if layer_key.startswith("model.") else layer_key
        mod = mods.get(k)
        return mod if mod is not None and hasattr(mod, "weight") else None

    def additive_ref(adapter_paths, args6):
        stack = M.load_stack([(p_, 1.0) for p_ in adapter_paths])
        saved = {}
        with torch.no_grad():
            for key in sorted(M.stack_layer_keys(stack)):
                mod = _mod(key)
                if mod is None:
                    continue
                # project_in / project_out are Conv1d with kernel 1, so their weight is
                # [out, in, 1] while every adapter -- and lora_core's merge math -- is 2-D.
                # Squeeze for the merge, restore the kernel axis for the copy back.
                wt = mod.weight.detach()
                conv = wt.ndim == 3 and wt.shape[-1] == 1
                # Stay on the device. The numpy twin needs a host round trip per layer and
                # does the DoRA row norms on CPU -- together ~5 s on the widest weights.
                W0 = (wt[..., 0] if conv else wt).float()
                W = M.merge_layer_additive_torch(W0, stack, key)
                if W is None:
                    continue
                if conv:
                    W = W[..., None]
                # snapshot on the HOST: keeping 168 fp32 clones on device is a second copy
                # of the DiT and OOMs a shared GPU.
                saved[key] = mod.weight.detach().to("cpu", copy=True)
                mod.weight.copy_(W.to(mod.weight.dtype))
            v = wrapper_with(dit, sd, w0, b0)(*args6).clone()
            for key, Wsv in saved.items():
                mod = _mod(key)
                mod.weight.copy_(Wsv.to(mod.weight.device))
        return v, len(saved)

    rows = []
    for names in stacks:
        adapter_paths = [str(AD / n) for n in names]
        v_add, n_add = additive_ref(adapter_paths, args6)
        load_and_apply_loras(model, adapter_paths, cfg["model_type"], svd_bases_path=str(paths.SVD_BASES))
        w_ad = cond.weight.detach().clone()
        with torch.no_grad():
            # The reference is torch WITHOUT the seconds embedder adapted, because
            # merge.EXCLUDED_LAYERS now drops that layer from every adapter (it is global
            # adaLN conditioning, and summing two DoRA rescales there sign-flips the gain and
            # NaNs the render -- see the note there). v_ref169 keeps it, so the table can
            # still price what the exclusion costs.
            v_ref = wrapper_with(model.model.model, sd, w0, b0)(*args6).clone()
            v_ref169 = wrapper_with(model.model.model, sd, w_ad, b0)(*args6).clone()
        remove_lora(model)
        with torch.no_grad():
            back = wrapper_with(model.model.model, sd, w0, b0)(*args6)
        assert rel(v_base, back) < 1e-6, "remove_lora left residue"
        # effect = the whole adapter stack, canonical torch. sec_cost = the part of it the
        # excluded seconds embedder was responsible for, on the same denominator. chain_gap =
        # chained vs summed composition: the semantics choice, NOT an error of ours.
        sec_cost = float((v_ref169.float() - v_ref.float()).norm() / v_base.float().norm())
        rows.append({"names": names, "paths": adapter_paths, "v_ref": v_add, "v_chain": v_ref,
                     "eff": rel(v_base, v_ref169), "sec_cost": sec_cost,
                     "chain_gap": rel(v_ref, v_add), "n_merged": n_add})
        print(f"    torch ref  {' + '.join(names):66s} effect {rows[-1]['eff']:.3e}"
              f"   sec {sec_cost:.3e}   chain-vs-sum {rows[-1]['chain_gap']:.3e}"
              f"   ({n_add} layers merged)", flush=True)

    del model; torch.cuda.empty_cache()

    from refold_runtime import RefoldLora
    B = RefoldLora(model="sa3-medium", engine_path=a.engine, branch_map=a.bmap)
    B.zero(1); v0 = run_engine(B, args6, L)
    floor = rel(v_base, v0)
    print(f"\n  engine {Path(B.engine_path).name}  ({len(B.map)} targets)   "
          f"fp16 floor {floor:.3e}\n")
    hdr = (f"    {'stack':62s}{'rank':>6s}{'effect':>11s}{'fold err':>11s}{'delta err':>11s}"
           f"{'chain gap':>11s}{'fold ms':>9s}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    out = []
    for r in rows:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        B.set_stack_gpu([(p, 1.0) for p in r["paths"]], {})
        torch.cuda.synchronize(); fold_ms = (time.perf_counter() - t0) * 1e3
        v = run_engine(B, args6, L)
        fold = rel(r["v_ref"], v)
        delta = rel(r["v_ref"].float() - v_base.float(), v.float() - v0.float())
        flag = "" if fold <= 1.5 * floor else "  <-- ABOVE fp16 floor"
        print(f"    {' + '.join(r['names'])[:62]:62s}{B.rank:6d}{r['eff']:11.3e}"
              f"{fold:11.3e}{delta:11.3e}{r['chain_gap']:11.3e}{fold_ms:9.1f}{flag}", flush=True)
        out.append({"stack": r["names"], "rank": B.rank, "effect": r["eff"],
                    "fold": fold, "delta": delta, "fold_ms": fold_ms,
                    "chain_gap": r["chain_gap"], "sec_cost": r["sec_cost"]})
    print(f"\n    fp16 floor {floor:.3e} — fold err at or under it means the stack is exact "
          f"to the engine's own precision.\n    'fold err' is against the SUMMED reference "
          f"(what this engine computes); 'chain gap' is how far chained composition\n    "
          f"(load_and_apply_loras / MLX / our refit engine) lands from it — a semantics "
          f"choice, not an error.")
    if a.json:
        Path(a.json).write_text(json.dumps({"floor": floor, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
