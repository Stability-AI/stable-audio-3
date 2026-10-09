"""RungDiT — medium DiT inference via a single multi-signature .tflite (dit_{prec}.tflite: static rung
subgraphs s<R> sharing weight buffers, built by build/build_dit.sh from stable_audio_3.models.dit).
Picks the SMALLEST rung R >= L and runs ONE full-length forward with the extra (R-L) KEYS masked out of
self-attention (attn_mask), so a render of real length L on rung R is EXACT (no SWA tiling — the DiT is a
single forward, unlike the decoder). On litert>=2.2.0 it loads ONE CompiledModel + XNNPACK weight cache so
the 10 rungs share one copy of the weights, and dispatches by rung -> peak RAM is the high-water mark of the
largest rung actually used, not the sum.

The 7 BAKED inputs match export_dit.DiTBaked.forward (args_0..6, in this order):
    x[1,256,R]  gc[1,9216]  t5_hidden[1,256,768]  t5_mask[1,256]  seconds[1]  local_add_cond[1,257,R]
    attn_mask[1,1,1,MEM+R]   (0 for the MEM+L valid keys, -1e9 for the R-L pad keys)
gc (the adaLN global = global_cond_embedder(to_global_embed(seconds)+timestep(t))) is computed OUTSIDE the
rung — by a 'gcond' signature bundled in the SAME .tflite (one shipped file; a sibling gcond.tflite is a
legacy fallback) — and fed in per step. This keeps the 24x-inlined global_cond_embedder FC OUT of the int8
rung, where under the XNNPACK weight cache it corrupts multi-rung int8 (see export_dit / memory
sa3-rung-weightcache-int8-bug). The conditioner (T5 padding + seconds) is baked in-graph. Batch is
baked to 1, so CFG runs as a SEQUENTIAL
dual-pass (cond + uncond), exactly like the static-batch=1 TensorRT engine. Interface mirrors BakedDiT so
sa3_tflite.main() can swap it in: __call__(x, t, cross, gcond) -> v (or cfg-guided v)."""
from __future__ import annotations
import os
import numpy as np

MEM = 64               # memory/register tokens prepended in-graph (must match export_dit.MEM)
COND_TOKENS = 256
COND_DIM = 768
LATENT_CH = 256
GCE_OUT = 9216         # adaLN global gc dim (6*embed_dim); computed by gcond.tflite, fed as a rung input


class RungDiT:
    """model_fn(x,t,cross,gcond)->v compatible with P.sample. cross/gcond are IGNORED (conditioning is
    baked in-graph, driven by the T5 outputs held here). cfg==1.0 -> one forward; cfg!=1.0 -> sequential
    cond+uncond dual-pass combined by cfg_fn (sa3_tflite._apply_cfg) in denoised space, with optional APG."""

    def __init__(self, path, L, t5_hidden, t5_mask, seconds, threads=8, cfg=1.0, apg=1.0,
                 null_hidden=None, null_mask=None, local_add_cond=None,
                 weight_cache_path="auto", cfg_fn=None, xnnpack_flags=None, kernel_mode=None,
                 enable_ynnpack=False, gcond_path="auto", **_ignored):
        from ai_edge_litert.compiled_model import CompiledModel, Options
        from ai_edge_litert.cpu_options import CpuOptions
        from ai_edge_litert.interpreter import Interpreter
        path = str(path)
        wc = (path + ".xnnwc") if weight_cache_path == "auto" else weight_cache_path
        copts = dict(num_threads=int(threads), xnnpack_weight_cache_path=str(wc) if wc else "")
        if xnnpack_flags is not None: copts["xnnpack_flags"] = xnnpack_flags
        if kernel_mode is not None: copts["kernel_mode"] = kernel_mode
        if enable_ynnpack: copts["enable_ynnpack"] = True
        self.m = CompiledModel.from_file(path, options=Options(cpu_options=CpuOptions(**copts)))
        # discover rung sizes from the file's s<N> signatures (any rung set just works)
        keys = {self.m.get_signature_by_index(i)["key"]: i for i in range(self.m.get_num_signatures())}
        self.sig = {int(k[1:]): v for k, v in keys.items() if k.startswith("s") and k[1:].isdigit()}
        # gc[1,9216] is computed OUTSIDE the rung (keeps the global_cond_embedder FC out of the int8 rung —
        # the XNNPACK-cache corruption trigger). It rides in the SAME .tflite as a 'gcond' signature (one
        # shipped file) and runs through THIS CompiledModel, which allocates per-signature on the flat-RAM
        # cache path. ⚠ do NOT open a plain Interpreter on the whole multi-rung file to reach it — that
        # allocates EVERY rung's arena (~18 GB). Legacy fallback (no 'gcond' sig): a sibling gcond.tflite.
        if "gcond" in keys:
            self._gcond_si = keys["gcond"]
            self._gc_inb = self.m.create_input_buffers(self._gcond_si)   # [0]=seconds, [1]=t (export arg order)
            self._gc_outb = self.m.create_output_buffers(self._gcond_si)
            self._gc_run = None
        else:
            sib = os.path.join(os.path.dirname(path), "gcond.tflite") if gcond_path == "auto" else str(gcond_path)
            self._gc_hold = Interpreter(model_path=sib, num_threads=int(threads))
            self._gc_run = self._gc_hold.get_signature_runner()
            self._gc_in = sorted(self._gc_run.get_input_details())
            self._gc_out = next(iter(self._gc_run.get_output_details()))
        self._gc_cache = {}
        if not self.sig:
            raise ValueError(f"{path}: no s<N> rung signatures found")
        self.sizes = sorted(self.sig)
        self.max_rung = self.sizes[-1]
        self._cfg_fn = cfg_fn
        self._slots = self._input_slots(path)          # semantic name -> buffer position (0..6)
        self._bufs = {}                                # si -> (in_bufs, out_bufs) (lazy per rung)
        self.set_conditioning(L, t5_hidden, t5_mask, seconds, cfg=cfg, apg=apg,
                              null_hidden=null_hidden, null_mask=null_mask, local_add_cond=local_add_cond)

    def _input_slots(self, path):
        """Map each of the 7 baked inputs to its CompiledModel buffer position. create_input_buffers()
        follows the SUBGRAPH inputs order == Interpreter.get_input_details() RAW order (NOT the signature
        alias order args_0..6) — validated max|Δ|=0 vs a name-fed run. Identify each by shape: the two rank-2
        inputs split by width (gc[1,9216] vs t5m[1,256]); seconds is the only rank-1 input (t is gone — it is
        folded into gc by gcond.tflite). All rungs share one export graph so subgraph 0's order holds."""
        from ai_edge_litert.interpreter import Interpreter
        det = Interpreter(model_path=path).get_input_details()   # raw list order == buffer order
        slots = {}
        for j, d in enumerate(det):
            shp = [int(s) for s in d["shape"]]
            if len(shp) == 4:
                slots["am"] = j
            elif len(shp) == 2 and shp[1] == GCE_OUT:
                slots["gc"] = j
            elif len(shp) == 2:
                slots["t5m"] = j
            elif len(shp) == 3 and shp[1] == 257:
                slots["lac"] = j
            elif len(shp) == 3 and shp[2] == COND_DIM:
                slots["t5h"] = j
            elif len(shp) == 3 and shp[1] == LATENT_CH:
                slots["x"] = j
            elif len(shp) == 1:
                slots["sec"] = j
        missing = {"x", "gc", "t5h", "t5m", "sec", "lac", "am"} - set(slots)
        if missing:
            raise ValueError(f"{path}: could not map DiT inputs {missing} "
                             f"(shapes={[[int(s) for s in d['shape']] for d in det]})")
        return slots

    def _rung_ge(self, L):
        ge = [s for s in self.sizes if s >= L]
        if not ge:
            raise ValueError(f"requested DiT length {L} exceeds largest rung {self.max_rung}")
        return ge[0]

    def set_conditioning(self, L, t5_hidden, t5_mask, seconds, *, cfg=1.0, apg=1.0,
                         null_hidden=None, null_mask=None, local_add_cond=None, **_ignored):
        """(Re)bind per-generation conditioning. Picks the rung R>=L, builds the pad-key attn_mask, and
        writes the constant inputs (seconds, local_add_cond, attn_mask — and, for cfg==1, t5h/t5m) once
        into the rung's resident buffers; only x and t change per diffusion step."""
        self.L = int(L)
        self.R = self._rung_ge(self.L)
        self.si = self.sig[self.R]
        self.cfg = float(cfg); self.apg = float(apg)
        self.n_fwd = 0
        if self.si not in self._bufs:
            self._bufs[self.si] = (self.m.create_input_buffers(self.si), self.m.create_output_buffers(self.si))
        self.inb, self.outb = self._bufs[self.si]

        self.t5h = self._pad_t5(t5_hidden)
        self.t5m = t5_mask.astype(np.float32).reshape(1, COND_TOKENS)
        self.null_h = None if null_hidden is None else self._pad_t5(null_hidden)
        self.null_m = None if null_mask is None else null_mask.astype(np.float32).reshape(1, COND_TOKENS)
        # seconds + local_add_cond + attn_mask are constant across steps AND cfg branches -> resident.
        self.seconds = float(seconds); self._gc_cache = {}    # seconds drives gc (per-step); invalidate cache
        sec = np.array([np.float32(seconds)], np.float32)
        lac = (np.zeros((1, 257, self.R), np.float32) if local_add_cond is None
               else self._pad_len(local_add_cond.astype(np.float32), 257))
        am = np.zeros((1, 1, 1, MEM + self.R), np.float32)
        am[..., MEM + self.L:] = -1e9                     # mask the (R-L) pad KEYS -> exact at length L
        self._write("sec", sec); self._write("lac", lac); self._write("am", am)
        if self.cfg == 1.0:                               # single branch -> t5h/t5m resident too
            self._write("t5h", self.t5h); self._write("t5m", self.t5m)
        return self

    def _pad_t5(self, h):
        return h.astype(np.float32).reshape(1, COND_TOKENS, COND_DIM)

    def _pad_len(self, a, ch):
        """Pad a [1,ch,L] tensor up to [1,ch,R] (pad content is irrelevant: pad queries are trimmed and
        pad keys are masked out of attention — no cross-position leak except attention, which is masked)."""
        L = a.shape[2]
        if L == self.R:
            return np.ascontiguousarray(a)
        out = np.zeros((1, ch, self.R), np.float32)
        out[:, :, :L] = a[:, :, :self.R]
        return out

    def _write(self, name, arr):
        self.inb[self._slots[name]].write(np.ascontiguousarray(arr, np.float32))

    def _compute_gc(self, t):
        """gc[1,9216] = gcond.tflite(seconds, t), cached by t. seconds is fixed per render and the global
        adaLN cond is seconds+timestep only (independent of the text), so the cond & uncond CFG passes share
        one gc -> one gcond invoke per diffusion step."""
        key = round(float(t), 7)
        gc = self._gc_cache.get(key)
        if gc is None:
            sec = np.array([np.float32(self.seconds)], np.float32); tt = np.array([np.float32(t)], np.float32)
            if self._gc_run is None:                                 # 'gcond' signature on self.m (flat RAM)
                self._gc_inb[0].write(np.ascontiguousarray(sec)); self._gc_inb[1].write(np.ascontiguousarray(tt))
                self.m.run_by_index(self._gcond_si, self._gc_inb, self._gc_outb)
                gc = np.asarray(self._gc_outb[0].read(GCE_OUT, np.float32)).reshape(1, GCE_OUT).astype(np.float32)
            else:                                                    # sibling gcond.tflite SignatureRunner
                out = self._gc_run(**{self._gc_in[0]: sec, self._gc_in[1]: tt})
                gc = np.asarray(out[self._gc_out]).reshape(1, GCE_OUT).astype(np.float32)
            self._gc_cache[key] = gc
        return gc

    def _fwd(self, x, t, t5h=None, t5m=None):
        """One batch=1 forward on the active rung. x is [1,256,L] (padded to R); returns v [1,256,L]."""
        self._write("x", self._pad_len(x.astype(np.float32), LATENT_CH))
        self._write("gc", self._compute_gc(t))            # adaLN global (gcond.tflite); replaces the t input
        if t5h is not None:                               # CFG branches rebind t5h/t5m per pass
            self._write("t5h", t5h); self._write("t5m", t5m)
        self.m.run_by_index(self.si, self.inb, self.outb)
        self.n_fwd += 1
        v = np.asarray(self.outb[0].read(LATENT_CH * self.R, np.float32)).reshape(1, LATENT_CH, self.R)
        return v[:, :, :self.L].copy()

    def __call__(self, x, t, cross=None, gcond=None):
        if self.cfg == 1.0:
            return self._fwd(x, t)
        v_cond = self._fwd(x, t, self.t5h, self.t5m)
        v_uncond = self._fwd(x, t, self.null_h, self.null_m)
        if self._cfg_fn is None:
            raise RuntimeError("RungDiT needs cfg_fn (sa3_tflite._apply_cfg) for cfg != 1.0")
        return self._cfg_fn(x, t, v_cond, v_uncond, self.cfg, self.apg)
