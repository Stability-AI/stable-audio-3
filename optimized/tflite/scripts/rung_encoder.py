"""RungEncoder — SAME-L encode via a single multi-signature .tflite (same_l_enc_rungs.tflite: static
rung subgraphs {16,32,64,128,256} sharing weight buffers). Mirror of rung_decoder.RungDecoder for the
downsampling direction: input audio [1,2,L*4096] -> latent [1,256,L]. Dispatches each encode to the
rung minimising total decoded latents, tiling in LATENT space (overlap TRIM per side, center-stitch);
the audio input is sliced by *4096. Static rungs -> tight reused arena (windowed O(S) attention),
unlike the shipped dense-varlen encoder whose arena bloats to ~16GB@L256. Needs litert>=2.2.0."""
from __future__ import annotations
import numpy as np

RUNGS = (2, 4, 8, 12, 16, 32, 64, 128, 256)   # doc only — actual rungs are auto-discovered from the
                                              # file's s<N> signatures. Small rungs {2,4,8,12}(+1 for
                                              # SAME-L) make tiny-L exact (lossless) & beat whole; the
                                              # {16..256} ladder tiles longer L. Weights dedup -> +~2MB.
SAMPLES_PER_LATENT = 4096
TRIM = 12                                 # overlap/side when tiling. 12 = the 12-block SWA receptive
                                          # field (12 blocks x +-1 latent-group) -> tiling BIT-EXACT
                                          # (-84 dB vs shipped whole-encode); TRIM=8 leaves -34 dB.


class RungEncoder:
    def __init__(self, path, threads=8, weight_cache_path="auto", trim=TRIM, max_rung=None):
        from ai_edge_litert.compiled_model import CompiledModel, Options
        from ai_edge_litert.cpu_options import CpuOptions
        path = str(path)
        self.trim = trim                 # overlap/side; SAME-L=12 (bit-exact), SAME-S=2
        self._cap = max_rung             # cap the largest rung used -> lower peak RAM (slower); None=auto
        wc = (path + ".xnnwc") if weight_cache_path == "auto" else weight_cache_path
        self.m = CompiledModel.from_file(path, options=Options(cpu_options=CpuOptions(
            num_threads=int(threads), xnnpack_weight_cache_path=str(wc) if wc else "")))
        keys = {self.m.get_signature_by_index(i)["key"]: i for i in range(self.m.get_num_signatures())}
        # discover rung sizes from the file's signatures (s<N>) — any rung set just works
        self.sig = {int(k[1:]): v for k, v in keys.items() if k.startswith("s") and k[1:].isdigit()}
        self.sizes = sorted(self.sig)
        if self._cap:                    # drop rungs larger than the cap (keeps at least the smallest)
            self.sizes = [s for s in self.sizes if s <= self._cap] or [self.sizes[0]]
        self.max_rung = self.sizes[-1]
        self._bufs = {}

    def _tileable(self, R):
        return R - 2 * self.trim > 0

    def _nwin(self, L, C):
        if L <= C:
            return 1
        step = C - 2 * self.trim
        if step <= 0:
            return 10 ** 9
        starts = list(range(0, L - C, step))
        if not starts or starts[-1] != L - C:
            starts.append(L - C)
        return len(starts)

    def _best_rung(self, L):
        """Min total encoded latents among LOSSLESS candidates (R==L, or a tileable R<L). Padding up is
        NOT a candidate. Fallback (tiny-L gap, L<smallest tileable and L!=rung): smallest rung >= L."""
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

    def _run_rung(self, audio, size):
        """Encode <=size latents on rung `size`. audio [1,2,<=size*4096] (pad up, trim latents)."""
        S = SAMPLES_PER_LATENT
        L = audio.shape[2] // S
        si = self.sig[size]
        need = size * S
        if audio.shape[2] < need:                       # pad audio by repeating last latent-frame
            last = audio[:, :, (L - 1) * S:L * S] if L > 0 else audio[:, :, :S]
            reps = (need - audio.shape[2] + S - 1) // S
            audio = np.concatenate([audio] + [last] * reps, 2)[:, :, :need]
        if si not in self._bufs:
            self._bufs[si] = (self.m.create_input_buffers(si), self.m.create_output_buffers(si))
        inb, outb = self._bufs[si]
        inb[0].write(np.ascontiguousarray(audio[:, :, :need], np.float32))
        self.m.run_by_index(si, inb, outb)
        y = np.asarray(outb[0].read(1 * 256 * size, np.float32)).reshape(1, 256, size)
        return y[:, :, :L]

    def encode(self, audio, balance=True):
        """audio [1,2,L*4096] -> latent [1,256,L]. Tile on a rung (overlap TRIM latents/side,
        center-stitch); overlap keeps SWA boundaries lossless."""
        S = SAMPLES_PER_LATENT
        L = audio.shape[2] // S
        C = self._best_rung(L)
        if L <= C:
            return self._run_rung(audio, C)             # exact (L==C) or padded (tiny-L)
        step = C - 2 * self.trim
        starts = list(range(0, L - C, step))
        if not starts or starts[-1] != L - C:
            starts.append(L - C)
        out = np.zeros((1, 256, L), np.float32)
        for k, st in enumerate(starts):
            y = self._run_rung(audio[:, :, st * S:(st + C) * S], C)   # [1,256,C]
            lo = 0 if k == 0 else TRIM
            hi = C if st + C >= L else C - TRIM
            out[:, :, st + lo:st + hi] = y[:, :, lo:hi]
        return out
