"""RungDecoder — SAME-L decode via a single multi-signature .tflite (same_l_rungs.tflite: static rung
subgraphs {16,32,64,128,256} sharing weight buffers). On litert>=2.2.0 it loads ONE CompiledModel +
XNNPACK weight cache and dispatches each decode to the SMALLEST rung signature >= L (pad up, trim),
tiling at 256 for L>256. Static rungs get a tight, reused arena (unlike the dynamic varlen, which
can't reuse buffers) → below 256 this is BOTH faster and lower-RAM than padding a fixed-256 model:
L=64 => 624ms/1.9GB vs fixed-256 2741ms/2.6GB vs varlen 2199ms/4.0GB. Needs litert>=2.2.0 (CompiledModel)."""
from __future__ import annotations
import numpy as np

RUNGS = (2, 4, 8, 12, 16, 32, 64, 128, 256)   # doc only — rungs are auto-discovered from the file's
                                              # s<N> signatures. Small rungs {2,4,8,12}(+1 SAME-L) make
                                              # tiny-L exact (lossless) & beat whole; {16..256} tiles.
SAMPLES_PER_LATENT = 4096
TRIM = 8                                 # overlap per side when tiling L>256 (SWA receptive field)


class RungDecoder:
    def __init__(self, path, threads=8, weight_cache_path="auto", limiter=None, trim=TRIM, max_rung=None):
        from ai_edge_litert.compiled_model import CompiledModel, Options
        from ai_edge_litert.cpu_options import CpuOptions
        path = str(path)
        self.trim = trim                 # overlap/side; SAME-L=12, SAME-S=16
        self._cap = max_rung             # cap the largest rung used -> lower peak RAM (slower); None=auto
        wc = (path + ".xnnwc") if weight_cache_path == "auto" else weight_cache_path
        self.m = CompiledModel.from_file(path, options=Options(cpu_options=CpuOptions(
            num_threads=int(threads), xnnpack_weight_cache_path=str(wc) if wc else "")))
        # map rung size -> signature index
        keys = {self.m.get_signature_by_index(i)["key"]: i for i in range(self.m.get_num_signatures())}
        # discover rung sizes from the file's signatures (s<N>) — any rung set just works
        self.sig = {int(k[1:]): v for k, v in keys.items() if k.startswith("s") and k[1:].isdigit()}
        self.sizes = sorted(self.sig)
        if self._cap:                    # drop rungs larger than the cap (keeps at least the smallest)
            self.sizes = [s for s in self.sizes if s <= self._cap] or [self.sizes[0]]
        self.max_rung = self.sizes[-1]
        self.limiter = limiter
        self._bufs = {}                  # sig_idx -> (in_bufs, out_bufs) (lazy per rung)

    def _tileable(self, R):
        return R - 2 * self.trim > 0                   # need a positive core between the 2 overlaps

    def _rung_le(self, L):
        """'fixed' dispatch: largest rung that can serve L — largest TILEABLE rung <= L, else the
        smallest rung >= L (whole/pad). `_best_rung` refines this by weighing overhead across rungs."""
        le = [s for s in self.sizes if s <= L and self._tileable(s)]
        if le:
            return le[-1]
        ge = [s for s in self.sizes if s >= L]
        return ge[0] if ge else self.sizes[-1]

    def _nwin(self, L, C):
        if L <= C:
            return 1
        step = C - 2 * self.trim
        if step <= 0:
            return 10 ** 9                            # overlap >= window: this rung can't tile (skip)
        starts = list(range(0, L - C, step))
        if not starts or starts[-1] != L - C:
            starts.append(L - C)
        return len(starts)

    def _best_rung(self, L):
        """Pick the rung minimising TOTAL decoded latents (#windows x rung) = least compute
        ('balancing'), among LOSSLESS candidates only: rung == L (exact) or a TILEABLE rung < L (overlap
        keeps SWA boundaries lossless). Padding UP to a bigger rung is NOT a candidate — even a 1-latent
        pad shifts the SWA boundary ~-29 dB. Tie -> fewer windows. e.g. L=255 -> rung-128 x2 (lossless,
        0.4% overhead), L=63 -> rung-32 x3. Fallback (tiny-L gap 17..31, no tileable rung <= L, and
        L != a rung): smallest rung >= L (one padded window — unavoidable without a smaller rung)."""
        cands = []
        for R in self.sizes:
            if R == L:
                cands.append(((R, 1), R))
            elif R < L and self._tileable(R):
                nw = self._nwin(L, R); cands.append(((nw * R, nw), R))
        if not cands:
            ge = [R for R in self.sizes if R >= L]
            return ge[0] if ge else self.sizes[-1]
        return min(cands)[1]

    def _run_rung(self, lat, size):
        """Decode <=size latents on rung `size` (pad up, trim). Returns [1,2,L*4096]."""
        L = lat.shape[2]
        si = self.sig[size]
        if L < size:
            lat = np.concatenate([lat, np.repeat(lat[:, :, -1:], size - L, 2)], 2)
        if si not in self._bufs:
            self._bufs[si] = (self.m.create_input_buffers(si), self.m.create_output_buffers(si))
        inb, outb = self._bufs[si]
        inb[0].write(np.ascontiguousarray(lat[:, :, :size], np.float32))
        self.m.run_by_index(si, inb, outb)
        y = np.asarray(outb[0].read(1 * 2 * size * SAMPLES_PER_LATENT, np.float32)).reshape(1, 2, size * SAMPLES_PER_LATENT)
        return y[:, :, :L * SAMPLES_PER_LATENT]

    def _limit(self, a):
        return self.limiter.apply(a) if (self.limiter is not None and a is not None) else a

    def decode(self, latents, balance=True):
        """Tile on a rung (overlap TRIM per side, center-stitch); overlap keeps SWA boundaries lossless.
        balance=True -> `_best_rung` (min total decoded latents); False -> `_rung_le` (largest rung <= L).
        L==rung => one exact window; L<smallest rung => one padded window (tiny-L edge only)."""
        L = latents.shape[2]
        C = self._best_rung(L) if balance else self._rung_le(L)
        if L <= C:
            return self._limit(self._run_rung(latents, C))          # exact (L==C) or padded (L<smallest)
        S = SAMPLES_PER_LATENT
        step = C - 2 * self.trim
        starts = list(range(0, L - C, step))
        if not starts or starts[-1] != L - C:
            starts.append(L - C)
        out = np.zeros((1, 2, L * S), np.float32)
        for k, st in enumerate(starts):
            y = self._run_rung(latents[:, :, st:st + C], C)          # [1,2,C*S]
            lo = 0 if k == 0 else TRIM
            hi = C if st + C >= L else C - TRIM
            out[:, :, (st + lo) * S:(st + hi) * S] = y[:, :, lo * S:hi * S]
        return self._limit(out)
