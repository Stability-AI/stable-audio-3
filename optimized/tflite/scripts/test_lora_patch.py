"""Tests for the TFLite LoRA merge (lora_core.py + lora_patch.py).

Stdlib unittest, no torch / no mlx / no pytest (matches test_windows_compat.py).
Two tiers:
  * CI-safe: safetensors reader, per-type merge math (independent re-derivation),
    spec parser, quantized-precision refusal — no model files needed.
  * Model-dependent (skipped unless a local DiT .tflite is present): the real
    plini adapter's 169/169 name-mapping bijection on medium; w16a32 fp16
    buffer discovery; and a synthetic FULL-target-set adapter on sm-music that
    exercises every mapping tier + the clone/patch/read-back path.

Run:  python scripts/test_lora_patch.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lora_core as lc  # noqa: E402

# Local model paths (present on a dev box with the tflite bundle installed;
# absent in CI → the model-dependent tests skip). SA3_TFLITE_MODELS_DIR
# overrides the location (e.g. pointing a worktree's tests at the main checkout).
_CHECKOUT = os.environ.get(
    "SA3_TFLITE_MODELS_DIR",
    os.path.abspath(os.path.join(_HERE, "..", "models", "tflite")))
_MODELS = {
    ("sm-music", "fp32"): f"{_CHECKOUT}/sa3-sm-music/dit_fp32.tflite",
    ("sm-music", "w16a32"): f"{_CHECKOUT}/sa3-sm-music/dit_w16a32.tflite",
    ("medium", "fp32"): f"{_CHECKOUT}/sa3-m/dit_fp32.tflite",
    ("sm-music", "w8a32"): f"{_CHECKOUT}/sa3-sm-music/dit_w8a32.tflite",
}
_PLINI = "/Users/cj/clod/speed-metal/scripts/lora_bench/plini-sa3-380.safetensors"


def _have(key):
    p = _MODELS.get(key)
    return p and os.path.exists(p)


def _save_native_safetensors(path, layers, cfg):
    """Write a minimal SA3-native adapter: {layer.parametrizations.weight.0.param}."""
    tensors, blob, hdr = {}, bytearray(), {}
    for layer, params in layers.items():
        for pname, arr in params.items():
            tensors[f"{layer}.parametrizations.weight.0.{pname}"] = np.ascontiguousarray(arr, np.float32)
    for k, v in tensors.items():
        off = len(blob)
        blob += v.tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, len(blob)]}
    hdr["__metadata__"] = {"lora_config": json.dumps(cfg)}
    hbytes = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.write(blob)


# ── CI-safe unit tests ─────────────────────────────────────────────────────────

class TestSafetensorsReader(unittest.TestCase):
    def test_roundtrip_fp32_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            A = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
            _save_native_safetensors(p, {"foo": {"lora_A": A, "lora_B": np.zeros((6, 4), np.float32)}},
                                     {"adapter_type": "lora", "rank": 4, "alpha": 4})
            t, meta = lc._load_safetensors(p)
            self.assertIn("foo.parametrizations.weight.0.lora_A", t)
            np.testing.assert_array_equal(t["foo.parametrizations.weight.0.lora_A"], A)
            self.assertEqual(json.loads(meta["lora_config"])["adapter_type"], "lora")

    def test_pickle_refused(self):
        for bad in ("evil.ckpt", "evil.pt", "evil.bin", "x.npz"):
            with self.assertRaises(lc.LoraError):
                lc._load_safetensors(bad)


class TestMergeMath(unittest.TestCase):
    """Independent re-derivation of the four core types (guards the copy)."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.W0 = self.rng.standard_normal((6, 8)).astype(np.float32)

    def _p(self, t, r=3):
        p = {}
        if not t.endswith("-xs"):
            p["lora_A"] = self.rng.standard_normal((r, 8)).astype(np.float32)
            p["lora_B"] = self.rng.standard_normal((6, r)).astype(np.float32)
        else:
            p["M_xs"] = self.rng.standard_normal((r, r)).astype(np.float32)
        if "magnitude" in lc._PARAMS_FOR[t]:
            p["magnitude"] = np.linalg.norm(self.W0, axis=1 if "rows" in t else 0).astype(np.float32)
        if "magnitude_r" in lc._PARAMS_FOR[t]:
            p["magnitude_r"] = np.linalg.norm(self.W0, axis=1).astype(np.float32)
            p["magnitude_c"] = np.linalg.norm(self.W0, axis=0).astype(np.float32)
        return p

    def test_lora(self):
        p = self._p("lora")
        ref = self.W0 + 2.0 * (p["lora_B"] @ p["lora_A"])
        np.testing.assert_allclose(lc.merged_weight(self.W0, p, "lora", 2.0), ref, atol=1e-5)

    def test_dora_rows(self):
        p = self._p("dora-rows")
        V = self.W0 + 1.0 * (p["lora_B"] @ p["lora_A"])
        ref = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12) * p["magnitude"].reshape(-1, 1)
        np.testing.assert_allclose(lc.merged_weight(self.W0, p, "dora-rows", 1.0), ref, atol=1e-5)

    def test_dora_cols(self):
        p = self._p("dora-cols")
        V = self.W0 + 1.0 * (p["lora_B"] @ p["lora_A"])
        ref = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12) * p["magnitude"].reshape(1, -1)
        np.testing.assert_allclose(lc.merged_weight(self.W0, p, "dora-cols", 1.0), ref, atol=1e-5)

    def test_all_types_finite_and_shaped(self):
        for t in lc._PARAMS_FOR:
            p = self._p(t)
            lc.check_shapes("x", self.W0, p, t)
            W = lc.merged_weight(self.W0, p, t, 1.0)
            self.assertEqual(W.shape, self.W0.shape)
            self.assertTrue(np.isfinite(W).all(), t)

    def test_shape_mismatch_raises(self):
        p = {"lora_A": np.zeros((3, 99), np.float32), "lora_B": np.zeros((6, 3), np.float32)}
        with self.assertRaises(lc.LoraError):
            lc.check_shapes("x", self.W0, p, "lora")


class TestSpecParser(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(lc.parse_lora_spec(["a.safetensors"])["strength"], 1.0)
        self.assertEqual(lc.parse_lora_spec(["a.safetensors", "strength=0.7"])["strength"], 0.7)

    def test_steps_rejected_points_to_mlx(self):
        with self.assertRaises(lc.LoraError) as cm:
            lc.parse_lora_spec(["a.safetensors", "steps=2-8"])
        self.assertIn("MLX", str(cm.exception))

    def test_bad_tokens(self):
        for bad in (["a", "b.st"], ["a", "strength=x"], []):
            with self.assertRaises(lc.LoraError):
                lc.parse_lora_spec(bad)


class TestQuantizedRefusal(unittest.TestCase):
    def test_precision_refused_before_touching_files(self):
        import lora_patch as lp
        for prec in ("w8a32", "w8a8-dyn", "w4a32"):
            with self.assertRaises(lc.LoraError) as cm:
                lp.get_patched_dit("/nonexistent.tflite", [{"path": "x", "strength": 1.0}],
                                   family="medium", precision=prec, log=lambda _m: None)
            self.assertIn(prec, str(cm.exception))


# ── model-dependent tests ────────────────────────────────────────────────────

@unittest.skipUnless(_have(("medium", "fp32")) and os.path.exists(_PLINI),
                     "medium fp32 DiT + plini adapter not present")
class TestRealAdapterMapping(unittest.TestCase):
    def test_plini_169_bijection_on_medium(self):
        import mmap
        import lora_patch as lp
        f = open(_MODELS[("medium", "fp32")], "rb")
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            mapper = lp._Mapper(list(lp._fc_weight_buffers(buf)))
            _at, _sc, layers = lc.parse_adapter(_PLINI)
            offs = set()
            for layer, p in layers.items():
                want = (p["lora_B"].shape[0], p["lora_A"].shape[1])
                rec = mapper.resolve(layer, want)
                self.assertEqual(rec["shape"], want, layer)
                offs.add(rec["off"])
            self.assertEqual(len(layers), 169)
            self.assertEqual(len(offs), 169, "buffer offsets must be unique (no collisions)")
        finally:
            del mapper
            buf.close()
            f.close()


@unittest.skipUnless(_have(("sm-music", "w16a32")), "sm-music w16a32 DiT not present")
class TestW16A32Discovery(unittest.TestCase):
    def test_all_weights_fp16_via_dequant(self):
        import mmap
        import lora_patch as lp
        f = open(_MODELS[("sm-music", "w16a32")], "rb")
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            fcs = list(lp._fc_weight_buffers(buf))
            self.assertGreater(len(fcs), 100)
            self.assertTrue(all(np.dtype(r["np_dtype"]) == np.float16 for r in fcs))
            self.assertTrue(any(r["shape"] == (768, 256) for r in fcs), "seconds cond present")
        finally:
            del fcs
            buf.close()
            f.close()


@unittest.skipUnless(_have(("sm-music", "fp32")), "sm-music fp32 DiT not present")
class TestFullSetPatchRoundTrip(unittest.TestCase):
    """Synthetic all-target adapter (every mapping tier) → clone + patch +
    byte-exact read-back on the smaller sm-music fp32 model."""

    def test_map_patch_readback(self):
        import mmap
        import lora_patch as lp
        base = _MODELS[("sm-music", "fp32")]
        f = open(base, "rb")
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        fcs = list(lp._fc_weight_buffers(buf))
        mapper = lp._Mapper(fcs)

        # Inverse-classify each FC → a checkpoint layer name, so we can build a
        # synthetic adapter that targets every tier (block linears, local_embed,
        # singleton embedders, seconds cond).
        ckpt_for = {}  # off -> (ckpt_layer_name, (fan_out, fan_in))
        # tier 1
        rev_block = {("self_attn", "to_qkv", False): "self_attn.to_qkv",
                     ("self_attn", "to_out", False): "self_attn.to_out",
                     ("cross_attn", "to_q", False): "cross_attn.to_q",
                     ("cross_attn", "to_kv", False): "cross_attn.to_kv",
                     ("cross_attn", "to_out", False): "cross_attn.to_out",
                     ("ff", "proj", True): "ff.ff.0.proj", ("ff", "2", False): "ff.ff.2"}
        for (blk, mod, leaf, glu), rec in mapper._block.items():
            ckpt_for[rec["off"]] = (f"model.transformer.layers.{blk}.{rev_block[(mod, leaf, glu)]}", rec["shape"])
        # tier 2 (topological rank == block)
        for leaf, recs in mapper._local.items():
            for blk, rec in enumerate(recs):
                ckpt_for[rec["off"]] = (f"model.transformer.layers.{blk}.to_local_embed.seq.{leaf}", rec["shape"])
        # tier 3 singletons
        for ckpt, sub in lp._SINGLETON_CKPT.items():
            hit = [r for r in fcs if sub in r["name"]]
            if hit:
                ckpt_for[hit[0]["off"]] = (ckpt, hit[0]["shape"])
        # tier 4 seconds cond
        if mapper._seconds:
            ckpt_for[mapper._seconds[0]["off"]] = (lc.COND_SECONDS_LAYER, mapper._seconds[0]["shape"])

        rng = np.random.default_rng(7)
        layers = {}
        for off, (name, (fo, fi)) in ckpt_for.items():
            r = 4
            layers[name] = {
                "lora_A": (rng.standard_normal((r, fi)) * 0.01).astype(np.float32),
                "lora_B": (rng.standard_normal((fo, r)) * 0.01).astype(np.float32),
                "magnitude": np.linalg.norm(
                    rng.standard_normal((fo, fi)).astype(np.float32), axis=1).astype(np.float32),
            }
        buf.close()
        f.close()

        with tempfile.TemporaryDirectory() as d:
            adapter = os.path.join(d, "full.safetensors")
            _save_native_safetensors(adapter, layers,
                                     {"adapter_type": "dora-rows", "rank": 4, "alpha": 4})
            specs = [lc.parse_lora_spec([adapter, "strength=0.5"])]
            out = lp.get_patched_dit(base, specs, family="sm-music", precision="fp32",
                                     cache_dir=os.path.join(d, "cache"), log=lambda _m: None)
            self.assertTrue(out.exists())

            # read-back: every patched buffer == fp32(W0 + 0.5*(merged-W0))
            fb = open(base, "rb"); bb = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
            fo2 = open(out, "rb"); bo = mmap.mmap(fo2.fileno(), 0, access=mmap.ACCESS_READ)
            def owned(mm, off, size, dt):
                # copy out of the mmap so no view keeps it open at close time
                return np.array(np.frombuffer(mm, dt, count=size // np.dtype(dt).itemsize,
                                              offset=off))
            m2 = lp._Mapper(list(lp._fc_weight_buffers(bb)))
            checked = 0
            for name, p in layers.items():
                rec = m2.resolve(name, (p["lora_B"].shape[0], p["lora_A"].shape[1]))
                W0 = owned(bb, rec["off"], rec["size"], np.float32).reshape(rec["shape"])
                got = owned(bo, rec["off"], rec["size"], np.float32).reshape(rec["shape"])
                ref = (W0 + 0.5 * (lc.merged_weight(W0, p, "dora-rows", 1.0) - W0)).astype(np.float32)
                self.assertTrue(np.array_equal(got, ref), name)
                checked += 1
            self.assertEqual(checked, len(layers))
            # a non-target FC — any FC not in our set — must be unchanged
            untouched = [r for r in lp._fc_weight_buffers(bb) if r["off"] not in ckpt_for]
            if untouched:
                r = untouched[0]
                self.assertTrue(np.array_equal(owned(bb, r["off"], r["size"], np.uint8),
                                               owned(bo, r["off"], r["size"], np.uint8)),
                                "untouched tensor changed")
            del m2
            bb.close(); bo.close(); fb.close(); fo2.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
