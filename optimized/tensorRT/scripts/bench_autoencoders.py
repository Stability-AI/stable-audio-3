#!/usr/bin/env python3
"""Sweep the eight shipped SAME autoencoder configurations: VRAM, latency, accuracy.

    {SAME-S, SAME-L} x {16-bit, fp8} x {chunked, single-shot}

16-bit means fp16 on SAME-L and bf16 on SAME-S -- the precision each model actually ships.

    python bench_autoencoders.py --out bench.json                    # full sweep
    python bench_autoencoders.py --out bench.json --skip-accuracy    # ms + VRAM only
    python make_ae_charts.py bench.json charts.html                  # render

── Two things this script exists to get right ────────────────────────────────────────────

1. ACCURACY IS MEASURED WITH CONTENT HELD FIXED. The obvious construction -- take the
   first L latents, round-trip them, score against the original -- makes the L axis a
   content walk, because a longer L is a different, longer piece of music. Measured: at a
   fixed L, moving the excerpt swings SNR by 9.4-15.0 dB, while the whole L axis at a fixed
   offset moves 2.95 dB. Content dominates by ~4x, so that chart plots the music, not the
   model. Instead `--accuracy fixed` (the default) reassembles ONE region from L-sized
   blocks, so only the processing length changes. That comes out flat to 0.002 dB (SAME-L)
   above L=16, which is the evidence that the windowed overlap is exact.
   `--accuracy absolute` keeps the naive construction for the absolute number.

2. REAL MUSIC, NEVER NOISE. The autoencoder is content-dependent -- published figures for
   this family span roughly -23 dB (solo piano) to -4 dB (dense circus) -- and noise
   fabricates numbers that do not hold on music. Point --music at a directory of audio, or
   at a .npy of shape (2, samples) at 44.1 kHz. L=8192 needs 761 s, longer than most single
   songs, so several tracks are concatenated rather than one tiled (tiling one track is
   artificially self-similar).

Waveform SNR systematically understates a perceptual autoencoder: the same pipeline scores
~16 dB on generated audio and ~5 dB on real masters. Compare configurations to each other
at a given L; do not read the absolute height as a quality verdict, and do not let it
substitute for listening.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

ENGINES = {
    ("same-l", "16"):  dict(enc="same-l/enc_fp16_chunkable.trt",
                            dec="same-l/dec_fp16_chunkable_limiter.trt"),
    ("same-l", "fp8"): dict(enc="same-l/enc_fp8_chunkable.trt",
                            dec="same-l/dec_fp8_chunkable_limiter.trt"),
    ("same-s", "16"):  dict(enc="same-s/enc_bf16_chunkable.trt",
                            dec="same-s/dec_bf16_chunkable_limiter.trt"),
    ("same-s", "fp8"): dict(enc="same-s/enc_fp8_chunkable.trt",
                            dec="same-s/dec_fp8_chunkable_limiter.trt"),
}
DEFAULT_LS = [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 255, 256, 257,
              511, 512, 1023, 1024, 1291, 1292, 2047, 2048, 4095, 4096, 4097, 6144, 8192]
CHUNK_LAT = 256
SPL = 4096          # samples per latent
SR = 44100
FIXED_REGION = 2048  # latents (~190 s) for the content-held-fixed accuracy pass


def load_music(spec: str, need_samples: int):
    """(2, N) float32 @ 44.1 kHz from a .npy, a file, or a directory of audio."""
    import numpy as np
    p = Path(spec)
    if p.suffix == ".npy":
        a = np.load(p)
    else:
        import soundfile as sf
        from scipy.signal import resample_poly
        files = sorted(p.glob("**/*")) if p.is_dir() else [p]
        segs = []
        got = 0
        for f in files:
            if f.suffix.lower() not in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
                continue
            try:
                x, sr = sf.read(f, dtype="float32", always_2d=True)
            except Exception:
                continue
            x = x.T
            if x.shape[0] == 1:
                x = np.repeat(x, 2, 0)
            x = x[:2]
            if sr != SR:                     # a 48 kHz corpus silently fails an == check
                x = resample_poly(x, SR, sr, axis=1).astype("float32")
            if x.shape[1] < SR * 5 or float(np.abs(x).max()) < 1e-3:
                continue
            segs.append(x); got += x.shape[1]
            if got >= need_samples:
                break
        if not segs:
            sys.exit(f"error: no usable audio under {spec}")
        # Concatenate DIFFERENT tracks rather than tile one: a tiled track is
        # artificially self-similar and flatters the autoencoder.
        a = np.concatenate(segs, axis=1)
        if a.shape[1] < need_samples:
            print(f"  warning: only {a.shape[1]/SR:.0f}s of audio; lengths needing more "
                  f"will be skipped")
    return a.astype("float32")


def snr_db(ref, y):
    import numpy as np
    n = min(len(ref), len(y)); ref, y = ref[:n], y[:n]
    e = ref - y
    return float(20 * np.log10(np.linalg.norm(ref) / max(np.linalg.norm(e), 1e-12)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="write results here as JSON")
    ap.add_argument("--music", default=None,
                    help="directory / file / .npy of REAL music (required unless "
                         "--skip-accuracy)")
    ap.add_argument("--lengths", default=None,
                    help="comma-separated latent lengths (default: 32 values, 1..8192)")
    ap.add_argument("--accuracy", choices=("fixed", "absolute", "both"), default="fixed",
                    help="fixed: one region reassembled from L-sized blocks, so only the "
                         "processing length varies (default). absolute: one L-length "
                         "excerpt per L -- the naive construction, dominated by content.")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats (min is kept)")
    a = ap.parse_args()

    import sa3_trt_core as canon
    canon._import_heavy()
    import numpy as np
    torch = canon.torch
    from sa3_trt_core import (TRTRunner, encoder_encode, encode_chunked,
                              decoder_decode, decode_chunked, profile_bounds)
    A = canon.ARCH_DIR
    LS = ([int(x) for x in a.lengths.split(",")] if a.lengths else DEFAULT_LS)

    music = None
    if not a.skip_accuracy:
        if not a.music:
            sys.exit("error: --music is required unless --skip-accuracy "
                     "(never benchmark these autoencoders on noise)")
        music = load_music(a.music, max(LS) * SPL + SR)
        print(f"  music: {music.shape[1]/SR:.0f}s  peak {np.abs(music).max():.3f}", flush=True)

    def bind(rel, prof):
        r = TRTRunner(A / rel, None, True, profile=prof)
        need = r.engine.get_device_memory_size_for_profile_v2(prof)
        buf = torch.empty(need, dtype=torch.uint8, device="cuda")
        r.context.set_device_memory(buf.data_ptr(), need)
        return r, buf, need

    def fastest(fn):
        fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(a.repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        return min(ts)

    rows = []
    for (model, prec), files in ENGINES.items():
        ref16 = ENGINES[(model, "16")]
        for chunked in (True, False):
            prof = 0 if chunked else 1
            print(f"\n══ {model} / {prec} / {'chunked' if chunked else 'single-shot'} ══",
                  flush=True)
            e_r, e_b, e_sc = bind(files["enc"], prof)
            d_r, d_b, d_sc = bind(files["dec"], prof)
            # the OTHER stage is pinned to this model's 16-bit chunked engine, so each
            # accuracy figure is attributable to the stage under test
            pe_r, pe_b, _ = bind(ref16["enc"], 0)
            pd_r, pd_b, _ = bind(ref16["dec"], 0)
            e_hi = profile_bounds(e_r, "audio")[1] // SPL
            d_hi = profile_bounds(d_r, "latent")[1]

            def roundtrip(off, L, enc_r, enc_ch, enc_hi, dec_r, dec_ch, dec_hi):
                seg = music[:, off * SPL:(off + L) * SPL]
                x = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0).cuda()
                z = (encode_chunked(enc_r, x, chunk_lat=min(CHUNK_LAT, enc_hi))
                     if enc_ch else encoder_encode(enc_r, x))
                z = z.to(dec_r.in_dtype["latent"])
                o = (decode_chunked(dec_r, z, chunk_lat=min(CHUNK_LAT, dec_hi))
                     if dec_ch else decoder_decode(dec_r, z))
                return np.clip(o.float()[0].cpu().numpy() / 32767.0, -1, 1)

            for L in LS:
                rec = dict(model=model, prec=prec, chunked=chunked, L=L,
                           enc_scratch_mb=e_sc / 2**20, dec_scratch_mb=d_sc / 2**20)
                # ── latency ──
                if music is not None and L * SPL <= music.shape[1]:
                    x = torch.from_numpy(
                        np.ascontiguousarray(music[:, :L * SPL])).unsqueeze(0).cuda()
                else:
                    x = None
                if x is not None:
                    try:
                        if L > e_hi and not chunked:
                            raise ValueError(f"L>{e_hi} exceeds the single-shot encoder band")
                        ef = ((lambda: encode_chunked(e_r, x, chunk_lat=min(CHUNK_LAT, e_hi)))
                              if chunked else (lambda: encoder_encode(e_r, x)))
                        rec["enc_ms"] = fastest(ef); z = ef()
                    except Exception as ex:
                        rec["enc_err"] = f"{type(ex).__name__}: {ex}"[:70]; z = None
                    try:
                        if z is None:
                            raise ValueError("no latents")
                        if L > d_hi and not chunked:
                            raise ValueError(f"L>{d_hi} exceeds the single-shot decoder band")
                        zz = z.to(d_r.in_dtype["latent"])
                        df = ((lambda: decode_chunked(d_r, zz, chunk_lat=min(CHUNK_LAT, d_hi)))
                              if chunked else (lambda: decoder_decode(d_r, zz)))
                        rec["dec_ms"] = fastest(df)
                    except Exception as ex:
                        rec["dec_err"] = f"{type(ex).__name__}: {ex}"[:70]
                # ── accuracy ──
                stages = {"enc": (e_r, chunked, e_hi, pd_r, True, CHUNK_LAT),
                          "dec": (pe_r, True, CHUNK_LAT, d_r, chunked, d_hi)}
                if music is not None and a.accuracy in ("fixed", "both") \
                        and L <= FIXED_REGION and FIXED_REGION % L == 0:
                    ref = music[:, :FIXED_REGION * SPL].T
                    for which, args_ in stages.items():
                        try:
                            if (not args_[1] and L > args_[2]) or (not args_[4] and L > args_[5]):
                                raise ValueError("band ceiling")
                            y = np.concatenate([roundtrip(o, L, *args_)
                                                for o in range(0, FIXED_REGION, L)], axis=0)
                            rec[f"{which}_fix_db"] = snr_db(ref, y)
                        except Exception as ex:
                            rec[f"{which}_fix_err"] = str(ex)[:50]
                if music is not None and a.accuracy in ("absolute", "both") \
                        and L * SPL <= music.shape[1]:
                    ref = music[:, :L * SPL].T
                    for which, args_ in stages.items():
                        try:
                            if (not args_[1] and L > args_[2]) or (not args_[4] and L > args_[5]):
                                raise ValueError("band ceiling")
                            rec[f"{which}_rt_db"] = snr_db(ref, roundtrip(0, L, *args_))
                        except Exception as ex:
                            rec[f"{which}_rt_err"] = str(ex)[:50]
                rows.append(rec)
                print(f"  L={L:>5d}  enc {rec.get('enc_ms', float('nan')):8.2f}ms  "
                      f"dec {rec.get('dec_ms', float('nan')):8.2f}ms  "
                      f"fixed {rec.get('dec_fix_db', float('nan')):7.3f}dB", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
            for o in (e_r, d_r, pe_r, pd_r):
                del o
            del e_b, d_b, pe_b, pd_b
            gc.collect(); torch.cuda.empty_cache()
    print(f"\n  wrote {a.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
