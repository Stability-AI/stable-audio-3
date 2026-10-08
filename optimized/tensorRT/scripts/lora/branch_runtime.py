"""Drive the branch engine: feed per-layer A/Bt operands and per-step, per-adapter strengths.

A stack concatenates along the rank axis, so rank block k belongs to adapter k and scaling that
block's entries in srow scales that adapter -- per-step, per-adapter strength for an R-element
write, with the CUDA graph replayed unchanged.
"""
import json, warnings
from pathlib import Path
import numpy as np, torch, tensorrt as trt

ROOT = Path(__file__).resolve().parent.parent
import merge as M
import lora_core as lc

LOG = trt.Logger(trt.Logger.ERROR)
# |chain - merge| above which the summed-vs-chained distinction is worth telling the
# user about. Trained adapters sit at 0.034 and sound identical under every rule
# (ear-tested 2026-10-08); one drifted adapter reaches 0.240, two reach 1.15-1.30.
_GAP_WARN = 0.1

ADAPTER_PFX = "model.transformer."


def adapter_key(rec):
    """Absolute adapter key for a branch-map entry.

    Block linears all sit under model.transformer., so older maps store only the suffix and
    the prefix is added here. The non-block targets do not share it -- the seconds_total
    embedder is under conditioners., the to_*_embed stack under model. -- so newer maps carry
    the key outright and it is used verbatim.
    """
    return rec.get("adapter_key") or ADAPTER_PFX + rec["ckpt_suffix"]


def low_rank_factors(W0, p, atype, scaling):
    """(Ap[r,in], Bp[out,r]) with `scaling` folded into Bp."""
    if atype.endswith("-xs"):
        r = p["M_xs"].shape[0]
        U, V = lc._svd_bases(np.asarray(W0, np.float32), r)
        return V.T, scaling * (U @ p["M_xs"])
    return p["lora_A"], scaling * p["lora_B"]


def uniform_terms(W0, p, atype, scaling):
    """(Ap, Bp, pchan_out) -- the same decomposition lora_live.uniform_terms produces.

    Branch reproduces  dW.x = pchan_out * (W0.x) + Bp.(Ap.x)  at strength 1, and folding strength
    linearly into BOTH terms gives exactly W(s) = W0 + s*(W_full - W0): the linear fade, exact at
    s=0 and s=1. dora-cols / bora additionally need an extra W0 pass (a 'slot'), which this branch
    has no operand for -- they are routed to the merged engine, where they are already exact.
    """
    W0 = np.asarray(W0, np.float32)
    Ap, Bp = low_rank_factors(W0, p, atype, scaling)
    base = atype[:-3] if atype.endswith("-xs") else atype
    if base == "lora":
        return Ap, Bp, None
    if base == "dora-rows":
        if atype.endswith("-xs"):
            r = p["M_xs"].shape[0]
            U, V = lc._svd_bases(W0, r)
            lr = scaling * (U @ p["M_xs"] @ V.T)
        else:
            lr = scaling * (p["lora_B"] @ p["lora_A"])
        V_ = W0 + lr
        c = np.squeeze(p["magnitude"]) / (np.linalg.norm(V_, axis=1) + 1e-12)
        return Ap, c.reshape(-1, 1) * Bp, (c - 1.0).astype(np.float32)
    raise ValueError(f"{atype}: needs an extra-W0 slot the branch has no operand for; "
                     f"route this adapter to the merged engine")


class BranchLora:
    def __init__(self, engine_path=None, device="cuda", model="sa3-medium",
                 branch_map=None, engine=None):
        # Engine, layer map and SVD bases all come from ONE place, so an engine can never be
        # driven with a map built for a different target set -- which names layers it does not
        # have, mis-sizes every operand, and raises nothing.
        import paths
        if model != "sa3-medium":
            raise ValueError(f"this build serves sa3-medium only, got model={model!r}")
        self.model = model
        self.paths = paths
        self.bases_path = paths.SVD_BASES
        ep = Path(engine_path or paths.BRANCH_ENGINE)
        if engine is not None:
            # Attach to an ALREADY-deserialised engine (the gradio has one loaded). Loading a
            # second copy just to drive its LoRA operands would double 2.9 GB of VRAM.
            self.engine = engine
        else:
            with open(ep, "rb") as f:
                self.engine = trt.Runtime(LOG).deserialize_cuda_engine(f.read())
            assert self.engine is not None, f"failed to load {ep}"
        self.engine_path = ep
        # Overriding the engine WITHOUT its map is the one way back into the mismatch this
        # class exists to prevent, so the two move together.
        mp = Path(branch_map) if branch_map else paths.BRANCH_MAP
        if engine_path and not branch_map and engine is None \
                and ep != Path(paths.BRANCH_ENGINE):
            raise ValueError(f"engine_path={ep.name} overrides the default engine but no "
                             f"branch_map was given; pass the map built alongside it")
        self.branch_map_path = mp
        self.map = json.load(open(mp))["layers"]
        self.dev = torch.device(device)
        self.rank = 0
        self.bufs = {}          # tensor name -> device fp16
        self.blocks = []        # (start, len, strength) per adapter
        self._pchan = {}        # layer -> [n_adapters, out] fp32, for re-folding pout per step

    def zero(self, rank=1):
        """No adapter: rank-1 zeros. The branch still runs -- that is its cost."""
        self.rank = rank
        for nm, rec in self.map.items():
            self.bufs[rec["A"]] = torch.zeros((1, rank, rec["in"]), dtype=torch.float16,
                                              device=self.dev)
            self.bufs[rec["B"]] = torch.zeros((1, rank, rec["out"]), dtype=torch.float16,
                                              device=self.dev)
            self.bufs[rec["P"]] = torch.zeros((1, 1, rec["out"]), dtype=torch.float16,
                                              device=self.dev)
        self.bufs["lora_srow"] = torch.zeros((1, 1, rank), dtype=torch.float16, device=self.dev)
        self.blocks = []
        return self

    def rank_cap(self):
        """Largest concatenated stack rank this engine's profile accepts."""
        _, _, hi = self.engine.get_tensor_profile_shape("lora_srow", 0)
        return int(hi[-1])

    def check_rank(self, stack):
        """Refuse a stack the engine cannot bind.

        Over the cap, TRT rejects every operand shape but set_input_shape only RETURNS False --
        it does not raise -- so the render proceeds on the previous bindings and returns
        confident, wrong audio. This lives in the runtime rather than in a caller because the
        CLI and the gradio reach set_stack/set_stack_gpu by different routes.
        """
        want = 0
        for a in stack:
            for q in a["layers"].values():
                t = q.get("lora_A")
                if t is None:
                    t = q.get("M_xs")          # -xs adapters carry M_xs [r, r]
                if t is not None:
                    want += t.shape[0]
                    break                      # rank is uniform across an adapter's layers
        cap = self.rank_cap()
        if want > cap:
            raise ValueError(
                f"stack rank {want} exceeds this engine's maximum of {cap} "
                f"({len(stack)} adapters). Load fewer adapters, use lower-rank ones, or "
                f"rebuild the engine with --rank-max >= {want}.")

    def check_stack_composable(self, stack, have_base_weights):
        """At most ONE renormalising (DoRA-family) adapter per stack, without base weights.

        Canonical torch CHAINS parametrizations, so a second DoRA renormalises the already-
        renormalised weight:  W2 = mag2 . (W1 + s2.d2)/||W1 + s2.d2||.  Expanding gives
        W2 = (c1.c2).W0 + (c1.c2.s1).d1 + (c2.s2).d2, which this branch *could* express -- but
        c2 = mag2/||c1.(W0+s1.d1) + s2.d2|| depends on the COMPOSED matrix, while the baked
        norm is ||W0 + s2.d2||. So the exact value needs W0. Summing the rescales instead (what
        this branch does) diverges hard once c is far from 1: measured fold error 1.150e+00 --
        larger than the adapter's own effect.

        ⚠ A summed rescale can cross ZERO where the chained product cannot: two adapters at
        c ~ 0.2 sum to 1 + 2(c-1) ~ -0.6 (sign-flipped) where c1.c2 = 0.04 stays positive. On
        the residual-stream targets that is merely wrong -- measured 2026-10-07, up to 100% of
        a layer's rows flipped and the render still finite. On the seconds embedder it was
        fatal, because that output is GLOBAL adaLN conditioning: 20 of 43 real stacks returned
        an all-NaN latent, silently. merge.EXCLUDED_LAYERS now drops that layer from every
        adapter, which took the same 43 stacks to 0 failures.

        Plain lora / lora-xs are linear, so any number of them compose exactly by summation.
        """
        renorm = [a for a in stack if a["type"].startswith(("dora", "bora"))]
        if len(renorm) < 2:
            return

        # NOT an approximation. This branch is bit-exact (1.5e-16 in fp64) against the repo's
        # own merge_loras_into_base_model(): each adapter's delta is taken against the SAME
        # original weight and summed, with our per-adapter strength playing the role of its
        # `application_weight`. What it does NOT match is load_and_apply_loras(), which
        # registers one parametrization per adapter and therefore CHAINS them, so adapter 2
        # renormalises the weight adapter 1 already rescaled. Those two torch paths disagree
        # with each other exactly as much as either disagrees with us (rel 1.27 on a real
        # pair), so this is a semantics choice, not a defect -- warn, do not refuse.
        import numpy as _np
        devs = []
        for a in renorm:
            worst = 0.0
            for q in a["layers"].values():
                mag, vn = q.get("magnitude"), q.get("baked_vnorm_row")
                if mag is None or vn is None:
                    worst = float("inf"); break
                c = _np.squeeze(mag).astype(_np.float64) / _np.clip(
                    _np.squeeze(vn).astype(_np.float64), 1e-12, None)
                worst = max(worst, float(_np.abs(c - 1.0).max()))
            devs.append(worst)
        gap = 1.0
        for d in sorted(devs, reverse=True)[:2]:
            gap *= d          # |chain - merge| is driven by the product of the deviations

        # How much of the BASE weight survives this stack, per output row:
        #     rho = 1 + sum_k (c_k - 1)        at strength 1
        # A row with rho < 0 has its base path SIGN-FLIPPED. That is the real failure, and it
        # is a property of the adapters, not of the engine -- measured 2026-10-08:
        #     a trained pair (c~0.99)   rho mean  0.975    0.00% of rows inverted
        #     an overfit pair (c~0.49)  rho mean -0.016   61.15% of rows inverted
        # The second renders 7.8 dB BELOW the base model: adding two adapters made it quieter
        # than using none. Report that, loudly, because no strength value fixes it.
        rho = []
        keys = set.intersection(*[set(a["layers"]) for a in renorm]) if renorm else set()
        for k in sorted(keys):
            acc = None
            for a in renorm:
                q = a["layers"][k]
                mag, vn = q.get("magnitude"), q.get("baked_vnorm_row")
                if mag is None or vn is None:
                    acc = None
                    break
                p = _np.squeeze(mag).astype(_np.float64) / _np.clip(
                    _np.squeeze(vn).astype(_np.float64), 1e-12, None) - 1.0
                acc = p if acc is None else acc + p
            if acc is not None:
                rho.append(1.0 + acc)
        inverted = 0.0
        if rho:
            r = _np.concatenate(rho)
            inverted = 100.0 * float((r < 0).mean())
            if inverted > 1.0:
                warnings.warn(
                    f"{inverted:.1f}% of rows have a SIGN-FLIPPED base coefficient at strength "
                    f"1.0 (rho = 1 + sum(c-1), mean {float(r.mean()):.3f}, min "
                    f"{float(r.min()):.3f}). The base weight is being cancelled, not adapted: "
                    f"this stack can render QUIETER THAN NO ADAPTER. It is the adapters, not "
                    f"the composition rule -- they are trained far from c=1 (max|c-1| = "
                    f"{', '.join(f'{d:.3g}' for d in devs)}). Lower the strengths, drop the "
                    f"most-drifted adapter, or retrain nearer c=1.",
                    RuntimeWarning, stacklevel=3)

        # The branch SUMS the rescales where load_and_apply_loras CHAINS them. Ear-tested
        # 2026-10-08 on a trained pair (gap 0.034): every composition rule sounded the same,
        # so this is only worth saying when the adapters are drifted enough for it to matter.
        # Measured gaps: trained pair 0.034, one-drifted 0.240, two-drifted 1.15-1.30.
        if gap > _GAP_WARN:
            warnings.warn(
                f"stack has {len(renorm)} renormalising (DoRA/BoRA) adapters: this follows "
                f"merge_loras_into_base_model semantics (deltas summed against the base), which "
                f"is what this engine computes exactly. It is NOT load_and_apply_loras' chained "
                f"behaviour; the two differ by ~{gap:.3g} here (max|c-1| = "
                f"{', '.join(f'{d:.3g}' for d in devs)}). Adapters trained near c=1 make the "
                f"distinction vanish.", RuntimeWarning, stacklevel=3)

    def warn_unserved(self, stack):
        """Adapter layers this engine has no branch for are dropped -- say so, once each.

        Not a rounding error worth swallowing: an adapter can load "successfully" here and
        still sound wrong. (The historical example, conditioners.seconds_total.embedder
        .embedding.1 -- which every sa3-medium adapter trained so far carries, and which the
        228-target engine never served -- no longer reaches this check: merge.EXCLUDED_LAYERS
        strips it at load and warns there instead.)

        Warned once per (adapter, engine) so a UI that swaps repeatedly does not spam.
        """
        served = {adapter_key(rec) for rec in self.map.values()}
        seen = self.__dict__.setdefault("_warned_unserved", set())
        for a in stack:
            missing = sorted(set(a["layers"]) - served)
            key = (a.get("path", "?"), self.engine_path.name)
            if not missing or key in seen:
                continue
            seen.add(key)
            head = ", ".join(missing[:4])
            if len(missing) > 4:
                head += f", +{len(missing) - 4} more"
            warnings.warn(
                f"{Path(a.get('path', '?')).name}: {len(missing)} of {len(a['layers'])} "
                f"adapted layers have no branch in {self.engine_path.name} and are IGNORED "
                f"-- {head}", RuntimeWarning, stacklevel=3)

    def check_operands_finite(self, where="fold"):
        """Refuse to hand the engine a non-finite operand. One sync, once per stack change.

        The branch feeds A, Bt, pout and srow as fp16 ACTIVATIONS, so nothing downstream ever
        validates them -- TRT will happily propagate an inf through 24 blocks and return a
        tensor of NaN with every call returning success. The 2026-10-07 seconds-embedder
        failure was exactly that shape: finite operands, NaN audio, no error. This catches the
        half of it that is visible at fold time (an overflowed Bp or pout); the other half is
        a semantics error, which EXCLUDED_LAYERS in merge.py now removes at the source.

        Deliberately NOT called from set_strengths(): that is the per-STEP path and a device
        sync there would cost more than the write it guards.
        """
        bad = torch.zeros((), dtype=torch.bool, device=self.dev)
        for t in self.bufs.values():
            bad |= ~torch.isfinite(t).all()
        if not bool(bad):            # the one sync
            return self
        for nm, rec in self.map.items():
            for slot in ("A", "B", "P"):
                t = self.bufs.get(rec[slot])
                if t is not None and not bool(torch.isfinite(t).all()):
                    n = int((~torch.isfinite(t)).sum())
                    raise ValueError(
                        f"{where}: operand {rec[slot]} has {n} non-finite entries "
                        f"(layer {nm}, max|finite| "
                        f"{float(t[torch.isfinite(t)].abs().max()) if torch.isfinite(t).any() else float('nan'):.3g}). "
                        f"fp16 overflowed during the fold -- the engine would return NaN audio "
                        f"and report success. Lower the strengths, or drop the adapter whose "
                        f"row rescale is furthest from 1 (see check_stack_composable).")
        raise ValueError(f"{where}: lora_srow has non-finite entries")

    def set_stack(self, specs, base_weights):
        """specs [(path, strength)]; base_weights {weight_name: device fp32} for the -xs SVDs."""
        stack = M.load_stack(specs)
        self.check_rank(stack)
        self.check_stack_composable(stack, bool(base_weights))
        self.warn_unserved(stack)
        per_layer = {nm: [[], [], []] for nm in self.map}   # A blocks, Bt blocks, pchan per adapter
        self.blocks = []
        start = 0
        for a in stack:
            r_seen = 0
            for nm, rec in self.map.items():
                p = a["layers"].get(adapter_key(rec))
                if p is None:
                    continue
                need_w0 = a["type"].endswith("-xs") or a["type"].startswith("dora")
                W0 = base_weights[nm].cpu().numpy() if need_w0 else None
                Ap, Bp, pch = uniform_terms(W0, p, a["type"], a["scaling"])
                per_layer[nm][0].append(np.ascontiguousarray(Ap, np.float32))
                per_layer[nm][1].append(np.ascontiguousarray(Bp.T, np.float32))  # rank-major!
                per_layer[nm][2].append(None if pch is None else
                                        np.ascontiguousarray(pch, np.float32))
                r_seen = Ap.shape[0]
            self.blocks.append((start, r_seen, a["strength"]))
            start += r_seen
        self.rank = start
        assert self.rank >= 1, "empty stack"
        for nm, rec in self.map.items():
            A = np.concatenate(per_layer[nm][0], 0) if per_layer[nm][0] else \
                np.zeros((self.rank, rec["in"]), np.float32)
            B = np.concatenate(per_layer[nm][1], 0) if per_layer[nm][1] else \
                np.zeros((self.rank, rec["out"]), np.float32)
            self.bufs[rec["A"]] = torch.from_numpy(A[None]).to(self.dev, torch.float16).contiguous()
            self.bufs[rec["B"]] = torch.from_numpy(B[None]).to(self.dev, torch.float16).contiguous()
            # keep each adapter's pchan so per-step strength can re-fold pout as sum_k s_k*pchan_k
            pcs = [np.zeros(rec["out"], np.float32) if q is None else q
                   for q in per_layer[nm][2]]
            self._pchan[nm] = (torch.from_numpy(np.stack(pcs)).to(self.dev, torch.float32)
                               if pcs else None)
            self.bufs[rec["P"]] = torch.zeros((1, 1, rec["out"]), dtype=torch.float16,
                                              device=self.dev)
        self.bufs["lora_srow"] = torch.zeros((1, 1, self.rank), dtype=torch.float16,
                                             device=self.dev)
        self.set_strengths([s for _, _, s in self.blocks])
        return self.check_operands_finite("set_stack")

    def set_strengths(self, per_adapter):
        """The per-STEP call: one R-element write, no recapture, graph replays unchanged."""
        v = torch.zeros(self.rank, dtype=torch.float16, device=self.dev)
        for (st, ln, _), s in zip(self.blocks, per_adapter):
            if ln:
                v[st:st + ln] = float(s)
        self.bufs["lora_srow"][0, 0, :] = v
        # pout must be re-folded with the same strengths, or DoRA degrades to plain LoRA
        sv = torch.tensor([float(x) for x in per_adapter], dtype=torch.float32, device=self.dev)
        for nm, rec in self.map.items():
            pc = self._pchan.get(nm)
            if pc is None:
                continue
            self.bufs[rec["P"]][0, 0, :] = (sv @ pc).to(torch.float16)
        return self
