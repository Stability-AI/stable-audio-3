"""Exact per-step strength for dora-rows: re-fold the operands instead of scaling srow.

The branch computes  y = W0.x*(1+pout) + Bp.(srow*(Ap.x)).  Scaling srow is one R-element
write, but it is the LINEAR FADE  W(s) = W0 + s*(W_full - W0) -- exact at s=0 and s=1 and
wrong in between for any variant that renormalises, because the true weight is

    W(s) = magnitude ⊙ (W0 + s·LR) / ||W0 + s·LR||_row ,      LR = (alpha/rank)·B·A

and that row norm is not linear in s. The TFLite lora_fast path handles this by re-folding
the operands on every apply (23-80 ms there). This does the same thing without ever
materialising the [out,in] delta:

    ||W0_i + s·LR_i||² = ||W0_i||² + 2s·<W0_i, LR_i> + s²·||LR_i||²
                       =    n0_i   +   2s·d_i        +   s²·g_i

All three coefficients are independent of s, so they are computed once per adapter, and a
strength change costs O(out) per layer instead of O(out·in) -- no GEMM, no host round trip.
d needs one [out,in]x[in,r] GEMM at load; g needs only the r×r Gram matrix.

Exactness caveat, unchanged from the branch: a STACK is superposed here but merged
sequentially (each adapter sees the previous one's output as its base), so re-folding makes a
single renormalising adapter exact and a stack of them closer, not exact. Stacks of pure
lora / lora-xs are exact either way.
"""
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
import branch_runtime as BR
import merge as M
import lora_core as lc
import paths

ADAPTER_PFX = BR.ADAPTER_PFX
adapter_key = BR.adapter_key


def _effective_factors(W0, p, atype, k):
    """(A_eff[r,in], B_eff[out,r]) with delta(s) = k*s*(B_eff @ A_eff). k = alpha/rank."""
    if atype.endswith("-xs"):
        r = p["M_xs"].shape[0]
        U, V = lc._svd_bases(np.asarray(W0, np.float32), r)
        return V.T, U @ p["M_xs"]
    return p["lora_A"], p["lora_B"]


class RefoldLora(BR.BranchLora):
    """BranchLora + set_strengths_exact(): re-folds pout and Bp at the requested strength."""

    def set_stack_refold(self, specs, base_weights):
        self.set_stack(specs, base_weights)          # buffers, blocks, rank, Ap all set up
        stack = M.load_stack(specs)
        self._rf = {}
        for nm, rec in self.map.items():
            per = []
            for a in stack:
                p = a["layers"].get(adapter_key(rec))
                if p is None:
                    per.append(None); continue
                base = a["type"][:-3] if a["type"].endswith("-xs") else a["type"]
                if base != "dora-rows":
                    per.append(None); continue      # lora/-xs are already exact under srow
                W0 = base_weights[nm].to(self.dev, torch.float32)   # GPU: 182 GEMMs at load
                k = a["scaling"] / max(a["strength"], 1e-12)     # scaling = k * strength
                A_eff, B_eff = _effective_factors(W0.cpu().numpy(), p, a["type"], k)
                A = torch.as_tensor(A_eff, dtype=torch.float32, device=self.dev)   # [r,in]
                B = torch.as_tensor(B_eff, dtype=torch.float32, device=self.dev)   # [out,r]
                n0 = (W0 * W0).sum(1)                                  # [out]
                WA = W0 @ A.T                                          # [out,r]  (one GEMM, once)
                d = k * (B * WA).sum(1)                                # [out]
                G = A @ A.T                                            # [r,r]
                g = (k * k) * ((B @ G) * B).sum(1)                     # [out]
                mag = torch.as_tensor(np.squeeze(p["magnitude"]).astype(np.float32),
                                      device=self.dev)
                per.append(dict(B=B, n0=n0, d=d, g=g, mag=mag, k=k))
            self._rf[nm] = per
        return self

    def set_strengths_exact(self, per_adapter):
        """Per-step call, exact for dora-rows. O(out) per layer; no host round trip."""
        # srow carries the pure-lora blocks; renormalising blocks are folded into Bp instead
        v = torch.zeros(self.rank, dtype=torch.float16, device=self.dev)
        for (st, ln, _), s in zip(self.blocks, per_adapter):
            if ln:
                v[st:st + ln] = float(s)
        for nm, rec in self.map.items():
            recs = self._rf.get(nm) or []
            if not any(recs):
                continue
            pout = torch.zeros(rec["out"], dtype=torch.float32, device=self.dev)
            Bt = self.bufs[rec["B"]][0].to(torch.float32)              # [R,out] rank-major
            for (st, ln, _), s, rf in zip(self.blocks, per_adapter, recs):
                if rf is None or not ln:
                    continue
                s = float(s)
                if s == 0.0:
                    # merge.py short-circuits strength 0 (`if a["strength"] == 0.0: continue`),
                    # so 0 means the adapter is ABSENT, not present with a zero delta.
                    # Renormalising here would still scale the weights by mag/||W0|| — which for
                    # a drifted adapter is a visibly different model at "no adapter" (measured
                    # cos 0.9814 against base). Note this makes the product's own strength curve
                    # DISCONTINUOUS at 0 for drifted DoRA: s=0 is base, s=0+ is renormalised.
                    Bt[st:st + ln, :] = 0.0
                    v[st:st + ln] = 0.0
                    continue
                nrm = torch.sqrt(torch.clamp(rf["n0"] + 2*s*rf["d"] + s*s*rf["g"], min=1e-24))
                # The magnitude fades WITH the direction. Scaling only the direction leaves
                # c -> magnitude/||W0|| as s->0, which is not 1 for a trained DoRA (measured
                # mean 0.383, min -0.188 on one medium layer), so "almost no adapter" was a
                # visibly different model: rel 6.4e-01 from base at s=1e-4. Interpolating the
                # magnitude from ||W0||_row makes s->0 land on base.
                w0n = torch.sqrt(torch.clamp(rf["n0"], min=1e-24))     # ||W0||_row
                mag_s = w0n + s * (rf["mag"] - w0n)
                c = mag_s / nrm                                        # [out]
                pout += c - 1.0
                # Bp = c ⊙ (k·s·B)  → rank-major slice is its transpose
                Bt[st:st+ln, :] = (rf["k"] * s * c.unsqueeze(1) * rf["B"]).T
                v[st:st+ln] = 1.0        # strength already inside Bp; keep srow neutral
            self.bufs[rec["P"]][0, 0, :] = pout.to(torch.float16)
            self.bufs[rec["B"]][0] = Bt.to(torch.float16)
        self.bufs["lora_srow"][0, 0, :] = v
        # Re-folding at a new strength can overflow where strength 1 did not, so this path is
        # checked even though it is per-step. It adds one device sync on top of the ~229
        # per-layer O(out) passes this method already launches -- immaterial here, and the
        # reason the cheap linear-fade set_strengths() is left unguarded.
        return self.check_operands_finite("set_strengths_exact")


# ── GPU fold ──────────────────────────────────────────────────────────────────────────────
# branch_runtime.set_stack folds on the HOST: for any DoRA adapter it pulls every W0 to the
# CPU (10.5 GB of fp32 across the 182 layers) and materialises B@A as a full [out,in] delta
# (another 10.5 GB) purely to take a row norm. Measured at 32.1 s for one swap, against 23 ms
# for the TFLite lora_fast equivalent.
#
# Nothing about that is required. The row norm expands to a quadratic (see the module
# docstring), so the delta never has to exist, and every operand can stay on the device.

_SVD_CACHE = {}
_FROZEN = None
_CKPT_SUFFIX = {}   # layer name -> checkpoint suffix, filled by set_stack_gpu
# Resolved in one place, not hardcoded: the bases must belong to the same model as the
# weights. Two variants' bases reconstruct their own at rel ~2.7e-4 and each other's at
# ~2.2e-2 -- close enough to look plausible, and nothing raises on a mismatch.
BASES_PATH = paths.SVD_BASES          # default; instances override via self.bases_path


def _frozen_bases(path=None):
    """The canonical frozen SVD bases, mmap'd.

    A full thin SVD of each adapted weight, taken on CPU in float32 with a deterministic
    sign convention and stored fp16.
    An -xs adapter's M_xs is trained against THESE, so they are loaded, never recomputed:
    recomputing cost ~43 s per process and required all 10.5 GB of W0 to be resident.
    """
    global _FROZEN
    p = Path(path or BASES_PATH)
    if not isinstance(_FROZEN, dict) or "__paths__" not in _FROZEN:
        _FROZEN = {"__paths__": {}}                     # one entry per bases file
    if p not in _FROZEN["__paths__"]:
        if not p.exists():
            raise FileNotFoundError(
                f"frozen SVD bases missing: {p}\n"
                f"  -xs adapters are trained against a FROZEN basis, so it must be the SAME\n"
                f"  file the adapter was trained with -- recomputing it is not equivalent\n"
                f"  unless the sign convention matches. Set $SA3_SVD_BASES.")
        _FROZEN["__paths__"][p] = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    return _FROZEN["__paths__"][p]


def _svd_bases_gpu(W0, rank, key, ckpt_suffix=None, bases_path=None):
    """(U[:, :r], V[:, :r]) on device, sliced from the frozen bases. No recompute path.

    A fallback that silently recomputed would reintroduce exactly what the frozen file exists
    to prevent, so a missing layer raises instead.
    """
    ck = (str(bases_path or BASES_PATH), key, rank)
    if ck in _SVD_CACHE:
        return _SVD_CACHE[ck]
    d = _frozen_bases(bases_path)
    suf = ckpt_suffix or key
    k = next((n for n in d if n.endswith(suf + ".weight")), None)
    if k is None:
        raise KeyError(f"no frozen SVD basis for {suf!r} in "
                       f"{Path(bases_path or BASES_PATH).name}")
    e = d[k]
    dev = W0.device if torch.is_tensor(W0) else "cuda"
    U = e["U"][:, :rank].to(dev, torch.float32)
    V = e["V"][:, :rank].to(dev, torch.float32)
    _SVD_CACHE[ck] = (U.contiguous(), V.contiguous())
    return _SVD_CACHE[ck]


def uniform_terms_gpu(W0, p, atype, scaling, key, dev, bases_path=None):
    """(Ap[r,in], Bp[out,r], pchan_out[out]|None) — branch_runtime.uniform_terms, on device."""
    T = lambda k: torch.as_tensor(np.ascontiguousarray(p[k]), device=dev, dtype=torch.float32)
    if atype.endswith("-xs"):
        r = p["M_xs"].shape[0]
        U, V = _svd_bases_gpu(W0, r, key, _CKPT_SUFFIX.get(key, key), bases_path)
        A_eff, B_eff = V.T.contiguous(), U @ T("M_xs")
    else:
        A_eff, B_eff = T("lora_A"), T("lora_B")
    Ap, Bp = A_eff, scaling * B_eff
    base = atype[:-3] if atype.endswith("-xs") else atype
    if base == "lora":
        return Ap, Bp, None
    if base == "dora-rows":
        mag = torch.as_tensor(np.squeeze(p["magnitude"]).astype(np.float32), device=dev)
        vn = p.get("baked_vnorm_row")
        if vn is not None:
            # Baked by bake_dora.py / bake_norms.py: ||W0 + scaling·LR||_row, precomputed.
            # W0 is not read at all -- which is the whole point, since it is a second 10.5 GB
            # copy of weights the engine already carries. The branch's strength knob is a
            # LINEAR FADE (set_strengths scales the cached pout and srow), so this one baked
            # vector serves every strength; only set_strengths_exact needs the quadratic terms.
            vn = torch.as_tensor(np.asarray(vn, np.float32), device=dev)
        else:
            if W0 is None:
                raise ValueError(
                    f"{atype}: unbaked adapter needs the base weights. Bake it first:\n"
                    f"    python bake_dora.py <adapter.safetensors>")
            # ||W0 + s·LR||²_row without ever forming LR = s·(B_eff @ A_eff)
            n0 = (W0 * W0).sum(1)
            d = scaling * (B_eff * (W0 @ A_eff.T)).sum(1)
            gq = (scaling ** 2) * ((B_eff @ (A_eff @ A_eff.T)) * B_eff).sum(1)
            vn = torch.sqrt(torch.clamp(n0 + 2 * d + gq, min=1e-24))
        c = mag / vn
        return Ap, c.unsqueeze(1) * Bp, (c - 1.0)
    raise ValueError(f"{atype}: needs an extra-W0 slot the branch has no operand for")


def set_stack_gpu(self, specs, base_weights):
    """set_stack with the fold on the device. Same buffers, same semantics, ~470x faster."""
    stack = M.load_stack(specs)
    self.check_rank(stack)
    self.check_stack_composable(stack, bool(base_weights))
    self.warn_unserved(stack)
    self.blocks, self._pchan, start = [], {}, 0
    per = {nm: [[], [], []] for nm in self.map}
    # Each adapter's rank is read from its OWN tensors, up front. Deriving it from "whatever
    # the last matching layer had" breaks the moment adapters in a stack target different
    # layer sets (e.g. a 229-layer lora stacked with a 169-layer dora): the layers only one of
    # them touches then contribute fewer rows than the stack rank, and the per-layer rescale
    # count stops matching the number of adapters -- which surfaces as
    # `mat1 and mat2 shapes cannot be multiplied` in set_strengths, or silently short buffers.
    ranks = []
    for a in stack:
        r = 0
        for q in a["layers"].values():
            t = q.get("lora_A")
            if t is None:
                t = q.get("M_xs")
            if t is not None:
                r = int(t.shape[0]); break
        ranks.append(r)

    for a, r_a in zip(stack, ranks):
        for nm, rec in self.map.items():
            p = a["layers"].get(adapter_key(rec))
            if p is None:
                # This adapter does not touch this layer: contribute an explicit ZERO block so
                # every layer carries one block per adapter and the offsets stay aligned.
                per[nm][0].append(torch.zeros(r_a, rec["in"], device=self.dev))
                per[nm][1].append(torch.zeros(r_a, rec["out"], device=self.dev))
                per[nm][2].append(None)
                continue
            _CKPT_SUFFIX[nm] = rec["ckpt_suffix"]
            # W0 is needed ONLY for an unbaked renormalising adapter. -xs bases come from the
            # frozen file; baked adapters carry their own norms. So a deployment of baked
            # and/or lora/-xs adapters never touches the 10.5 GB at all.
            need_w0 = a["type"].startswith("dora") and "baked_vnorm_row" not in p
            if need_w0 and nm not in base_weights:
                raise ValueError(
                    f'{a["type"]} adapter has no baked row norms and no base weights were '
                    f'supplied.\n    Bake it once:  python bake_dora.py <adapter.safetensors>'
                    f'\n    (baking removes a ~10 GB / 5-20 s load from every start)')
            W0 = base_weights[nm].to(self.dev, torch.float32) if need_w0 else None
            Ap, Bp, pch = uniform_terms_gpu(W0, p, a["type"], a["scaling"], nm, self.dev,
                                            getattr(self, "bases_path", None))
            per[nm][0].append(Ap)
            per[nm][1].append(Bp.T.contiguous())          # rank-major, as the engine expects
            per[nm][2].append(pch)
        self.blocks.append((start, r_a, a["strength"]))
        start += r_a
    self.rank = start
    assert self.rank >= 1, "empty stack"
    for nm, rec in self.map.items():
        A = (torch.cat(per[nm][0], 0) if per[nm][0]
             else torch.zeros(self.rank, rec["in"], device=self.dev))
        B = (torch.cat(per[nm][1], 0) if per[nm][1]
             else torch.zeros(self.rank, rec["out"], device=self.dev))
        self.bufs[rec["A"]] = A[None].to(torch.float16).contiguous()
        self.bufs[rec["B"]] = B[None].to(torch.float16).contiguous()
        pcs = [torch.zeros(rec["out"], device=self.dev) if q is None else q for q in per[nm][2]]
        self._pchan[nm] = torch.stack(pcs) if pcs else None
        self.bufs[rec["P"]] = torch.zeros((1, 1, rec["out"]), dtype=torch.float16, device=self.dev)
    self.bufs["lora_srow"] = torch.zeros((1, 1, self.rank), dtype=torch.float16, device=self.dev)
    self.set_strengths([s for _, _, s in self.blocks])
    return self.check_operands_finite("set_stack_gpu")


RefoldLora.set_stack_gpu = set_stack_gpu
