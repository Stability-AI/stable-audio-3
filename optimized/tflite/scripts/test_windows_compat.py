"""Cross-platform (Windows/macOS/Linux) compatibility tests for the TFLite CLI.

Run directly — stdlib unittest only, no pytest, NO model downloads, no network:

    python scripts/test_windows_compat.py

Covers the three Windows-specific failure modes this release guards against:
1. weights.ensure_local materializes model files without symlink privileges
   (symlink -> hardlink -> copy fallback chain);
2. the CLI's banners/emoji print cleanly under a legacy console code page
   (the guarded sys.stdout.reconfigure(encoding="utf-8") at script top);
3. --play dispatches to the right playback backend per platform
   (afplay / winsound / aplay) without executing anything.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Keep this test's own output safe under legacy Windows console code pages
# (the subprocess tests below spawn FRESH interpreters, so this does not mask them).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SCRIPTS_DIR = Path(__file__).resolve().parent          # <project>/scripts
PROJECT_DIR = SCRIPTS_DIR.parent                       # <project>
sys.path.insert(0, str(SCRIPTS_DIR))                   # import weights / sa3_tflite / examples

import weights  # noqa: E402


class TestEnsureLocalLinkOrCopy(unittest.TestCase):
    """ensure_local must materialize the target on every OS, with or without
    symlink privileges (Windows without Developer Mode raises WinError 1314)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="sa3_citest_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.fake_cache = self.tmp / "hf_cache"
        self.fake_cache.mkdir()
        # Redirect the project root so nothing is written into the repo checkout.
        self._orig_script_dir = weights.SCRIPT_DIR
        weights.SCRIPT_DIR = self.tmp / "project"
        self.addCleanup(lambda: setattr(weights, "SCRIPT_DIR", self._orig_script_dir))
        # Register a fake manifest entry + suppress the (non-ASCII) HF login tip
        # so no in-process print depends on console encoding.
        self.rel = "models/tflite/_citest/fake_model.bin"
        weights.FLAT_MANIFEST[self.rel] = "tflite/_citest/fake_model.bin"
        self.addCleanup(weights.FLAT_MANIFEST.pop, self.rel, None)
        self._orig_tip = weights._LOGIN_TIP_SHOWN
        weights._LOGIN_TIP_SHOWN = True
        self.addCleanup(lambda: setattr(weights, "_LOGIN_TIP_SHOWN", self._orig_tip))
        self.payload = b"not a real tflite model \x00\x01\x02" * 100

    def _fake_hf_hub_download(self, repo_id, filename):
        self.assertEqual(repo_id, weights.REPO_ID)
        cached = self.fake_cache / filename.replace("/", "__")
        cached.write_bytes(self.payload)
        return str(cached)

    def test_ensure_local_materializes_target(self):
        """End-to-end: monkeypatched download -> target exists and is readable.
        Uses whatever link mode the OS grants (symlink where privileged,
        hardlink/copy otherwise) — all satisfy exists()/is_present()."""
        import huggingface_hub
        with mock.patch.object(huggingface_hub, "hf_hub_download",
                               self._fake_hf_hub_download):
            target = weights.ensure_local(self.rel, verbose=False)
        self.assertTrue(target.exists(), f"target not materialized: {target}")
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertTrue(weights.is_present(self.rel))

    def test_fallback_chain_needs_no_symlink_privilege(self):
        """Force symlink_to to fail (as on Windows without Developer Mode,
        WinError 1314) and verify the hardlink/copy fallback still materializes
        a readable, non-symlink target. Runs identically on every OS."""
        cached = self.fake_cache / "src.bin"
        cached.write_bytes(self.payload)
        target = self.tmp / "project" / "models" / "linked.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        boom = OSError(1314, "A required privilege is not held by the client")
        with mock.patch.object(type(target), "symlink_to", side_effect=boom):
            weights._link_or_copy(cached, target)
        self.assertTrue(target.exists())
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), self.payload)

    def test_copy_fallback_when_hardlink_also_fails(self):
        """Symlink AND hardlink refused (e.g. cross-volume) -> plain copy."""
        cached = self.fake_cache / "src2.bin"
        cached.write_bytes(self.payload)
        target = self.tmp / "project" / "models" / "copied.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(type(target), "symlink_to",
                               side_effect=OSError(1314, "no symlink privilege")), \
             mock.patch.object(weights, "_LINK_FALLBACK_SHOWN", True), \
             mock.patch("os.link", side_effect=OSError(17, "cross-device link")):
            weights._link_or_copy(cached, target)
        self.assertTrue(target.exists())
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), self.payload)


class TestConsoleEncoding(unittest.TestCase):
    """The banners use box-drawing + emoji. Run the scripts in FRESH interpreters
    with PYTHONUTF8/PYTHONIOENCODING stripped, so stdout is a pipe under the
    runner's native default encoding (cp1252 on Windows without the in-script
    reconfigure fix) — a UnicodeEncodeError would make these exit nonzero."""

    @staticmethod
    def _clean_env():
        env = os.environ.copy()
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        return env

    def test_sa3_tflite_help(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "sa3_tflite.py"), "--help"],
            capture_output=True, env=self._clean_env(), timeout=120)
        self.assertEqual(
            r.returncode, 0,
            f"--help failed\nstdout:\n{r.stdout.decode('utf-8', 'replace')}\n"
            f"stderr:\n{r.stderr.decode('utf-8', 'replace')}")
        out = r.stdout.decode("utf-8", "replace")
        self.assertIn("--prompt", out)
        self.assertIn("--play", out)

    def test_examples_block(self):
        # examples.py is a module (no __main__): render the full emoji block.
        r = subprocess.run(
            [sys.executable, "-c",
             "import examples; examples.print_example_commands()"],
            capture_output=True, env=self._clean_env(), cwd=str(SCRIPTS_DIR),
            timeout=120)
        self.assertEqual(
            r.returncode, 0,
            f"examples block failed\nstdout:\n{r.stdout.decode('utf-8', 'replace')}\n"
            f"stderr:\n{r.stderr.decode('utf-8', 'replace')}")
        self.assertIn("Generate audio", r.stdout.decode("utf-8", "replace"))


class TestPlayBackendDispatch(unittest.TestCase):
    """--play platform dispatch, without playing anything."""

    def _backend(self):
        import sa3_tflite
        return sa3_tflite._play_backend

    def test_darwin_uses_afplay(self):
        self.assertEqual(self._backend()("darwin"), ("subprocess", ["afplay"]))

    def test_windows_uses_winsound(self):
        kind, argv = self._backend()("win32")
        self.assertEqual(kind, "winsound")
        self.assertIsNone(argv)
        if sys.platform == "win32":
            import winsound  # stdlib on Windows — the module the dispatcher relies on
            self.assertTrue(hasattr(winsound, "PlaySound"))

    def test_linux_uses_aplay_or_prints_path(self):
        kind, argv = self._backend()("linux")
        if kind == "subprocess":
            self.assertEqual(argv[0], "aplay")
        else:
            self.assertEqual((kind, argv), ("none", None))

    def test_current_platform_resolves(self):
        kind, argv = self._backend()()
        self.assertIn(kind, ("subprocess", "winsound", "none"))
        if kind == "subprocess":
            self.assertIsInstance(argv, list)
        else:
            self.assertIsNone(argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
