"""ONNX Runtime session factory for the DirectML execution provider."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


class DirectMLUnavailable(RuntimeError):
    pass


def _english_system_messages() -> None:
    # ORT embeds Win32 error text in the ANSI code page, which its Python binding fails to
    # decode as UTF-8 on non-English Windows; English messages are plain ASCII.
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadUILanguage(0x0409)


@dataclass
class Placement:
    """Kernel launches per execution provider, collected from an ORT profile."""

    kernels: Counter = field(default_factory=Counter)
    cpu_ops: Counter = field(default_factory=Counter)

    def merge(self, other: Placement) -> None:
        self.kernels.update(other.kernels)
        self.cpu_ops.update(other.cpu_ops)

    def describe(self) -> str:
        parts = [f"{provider.replace('ExecutionProvider', '')}={count}"
                 for provider, count in sorted(self.kernels.items())]
        text = ", ".join(parts) if parts else "no kernels recorded"
        if self.cpu_ops:
            ops = ", ".join(f"{op}x{n}" for op, n in sorted(self.cpu_ops.items()))
            text += f" (CPU ops: {ops})"
        return text


@dataclass
class RuntimeConfig:
    provider: str = "dml"
    device_id: int = 0
    profile: bool = False
    log_level: int = 3

    def providers(self) -> list:
        if self.provider == "cpu":
            return [CPU]
        if DML not in ort.get_available_providers():
            raise DirectMLUnavailable(
                "DmlExecutionProvider is not available. Install onnxruntime-directml "
                "(and not the plain onnxruntime package) on Windows 10/11.")
        return [(DML, {"device_id": str(self.device_id)}), CPU]


class Session:
    """InferenceSession wrapper with static-shape binding and optional placement profiling.

    `initializers` replaces externally stored graph weights by name before graph optimization,
    so the graph file itself is never rewritten.
    """

    def __init__(self, path: Path | str, config: RuntimeConfig, dims: dict[str, int] | None = None,
                 name: str | None = None, initializers: dict[str, np.ndarray] | None = None):
        self.name = name or Path(path).stem
        options = ort.SessionOptions()
        options.log_severity_level = config.log_level
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if config.provider != "cpu":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        for dim, value in (dims or {}).items():
            options.add_free_dimension_override_by_name(dim, int(value))
        self._weights = [ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a))
                         for a in (initializers or {}).values()]
        if self._weights:
            options.add_external_initializers(list(initializers), self._weights)
        self._profile_dir = None
        if config.profile:
            self._profile_dir = tempfile.mkdtemp(prefix="sa3_dml_profile_")
            options.enable_profiling = True
            options.profile_file_prefix = os.path.join(self._profile_dir, self.name)
        _english_system_messages()
        self.session = ort.InferenceSession(str(path), options, providers=config.providers())
        if config.provider != "cpu" and DML not in self.session.get_providers():
            raise DirectMLUnavailable(f"{self.name}: ONNX Runtime could not create a DirectML session")
        self.session.disable_fallback()
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.placement = Placement()

    def input_dtype(self, name: str) -> np.dtype:
        return np.float16 if self.inputs[name].type == "tensor(float16)" else np.dtype(
            {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32}[
                self.inputs[name].type])

    def run(self, feeds: dict[str, np.ndarray], outputs: list[str] | None = None) -> list[np.ndarray]:
        typed = {k: np.ascontiguousarray(v, dtype=self.input_dtype(k)) for k, v in feeds.items()}
        return self.session.run(outputs or self.outputs, typed)

    def close(self) -> Placement:
        if self.session is not None and self._profile_dir is not None:
            profile = self.session.end_profiling()
            self.placement = read_placement(profile)
            try:
                os.remove(profile)
                os.rmdir(self._profile_dir)
            except OSError:
                pass
        self.session = None
        self.inputs = {}
        self._weights = []
        gc.collect()
        return self.placement


class SessionPool:
    """Keeps at most one session per role, reused while its graph, shape and weights are unchanged."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._sessions: dict[str, tuple[tuple, Session]] = {}
        self.placements: dict[str, Placement] = {}

    def get(self, role: str, path: Path | str, dims: dict[str, int] | None = None,
            initializers: dict[str, np.ndarray] | None = None, key: object = None) -> Session:
        ident = (str(path), tuple(sorted((dims or {}).items())), key)
        current = self._sessions.get(role)
        if current is not None and current[0] == ident:
            return current[1]
        self.release(role)
        session = Session(path, self.config, dims, name=role, initializers=initializers)
        self._sessions[role] = (ident, session)
        return session

    def release(self, role: str | None = None) -> None:
        for name in [role] if role is not None else list(self._sessions):
            entry = self._sessions.pop(name, None)
            if entry is not None:
                self.placements.setdefault(name, Placement()).merge(entry[1].close())


def read_placement(profile_path: str) -> Placement:
    placement = Placement()
    with open(profile_path, encoding="utf-8") as f:
        events = json.load(f)
    for event in events:
        args = event.get("args", {})
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        provider = args.get("provider")
        if provider is None:
            continue
        placement.kernels[provider] += 1
        if provider == CPU:
            placement.cpu_ops[args.get("op_name", "?")] += 1
    return placement
