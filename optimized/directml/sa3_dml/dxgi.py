"""DXGI adapter enumeration and per-process video memory queries (Windows only)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

_LOCAL, _NON_LOCAL = 0, 1
_SOFTWARE_FLAG = 0x2


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> _GUID:
        import uuid
        u = uuid.UUID(text)
        return cls(u.time_low, u.time_mid, u.time_hi_version,
                   (ctypes.c_ubyte * 8).from_buffer_copy(u.bytes[8:]))


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wintypes.UINT),
                ("DeviceId", wintypes.UINT), ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", _LUID), ("Flags", wintypes.UINT)]


class _MemoryInfo(ctypes.Structure):
    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64), ("CurrentReservation", ctypes.c_uint64)]


_IID_FACTORY1 = _GUID.parse("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_ADAPTER3 = _GUID.parse("645967a4-1392-4310-a798-8053ce3e93fd")


def _method(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _method(obj, 2, wintypes.ULONG)(obj)


@dataclass
class Adapter:
    index: int
    name: str
    vendor_id: int
    dedicated_bytes: int
    shared_bytes: int
    software: bool
    luid: tuple[int, int]


@dataclass
class MemoryUsage:
    local_usage: int
    local_budget: int
    non_local_usage: int
    non_local_budget: int


def _factory() -> ctypes.c_void_p | None:
    if sys.platform != "win32":
        return None
    factory = ctypes.c_void_p()
    hr = ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory))
    return factory if hr == 0 else None


def _enum(factory: ctypes.c_void_p):
    index = 0
    enum_adapters1 = _method(factory, 12, ctypes.HRESULT, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
    while True:
        adapter = ctypes.c_void_p()
        try:
            enum_adapters1(factory, index, ctypes.byref(adapter))
        except OSError:
            return
        yield index, adapter
        index += 1


def adapters() -> list[Adapter]:
    """DXGI adapters in enumeration order, which is the order of the DirectML `device_id`."""
    factory = _factory()
    if factory is None:
        return []
    found = []
    try:
        for index, adapter in _enum(factory):
            desc = _AdapterDesc1()
            _method(adapter, 10, ctypes.HRESULT, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc))
            found.append(Adapter(index, desc.Description, desc.VendorId, desc.DedicatedVideoMemory,
                                 desc.SharedSystemMemory, bool(desc.Flags & _SOFTWARE_FLAG),
                                 (desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart)))
            _release(adapter)
    finally:
        _release(factory)
    return found


def memory_usage(luid: tuple[int, int]) -> MemoryUsage | None:
    """This process's video memory usage and budget on the adapter with the given LUID."""
    factory = _factory()
    if factory is None:
        return None
    try:
        for _, adapter in _enum(factory):
            desc = _AdapterDesc1()
            _method(adapter, 10, ctypes.HRESULT, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc))
            if (desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart) != tuple(luid):
                _release(adapter)
                continue
            adapter3 = ctypes.c_void_p()
            _method(adapter, 0, ctypes.HRESULT, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))(
                adapter, ctypes.byref(_IID_ADAPTER3), ctypes.byref(adapter3))
            _release(adapter)
            query = _method(adapter3, 14, ctypes.HRESULT, wintypes.UINT, wintypes.UINT,
                            ctypes.POINTER(_MemoryInfo))
            local, non_local = _MemoryInfo(), _MemoryInfo()
            query(adapter3, 0, _LOCAL, ctypes.byref(local))
            query(adapter3, 0, _NON_LOCAL, ctypes.byref(non_local))
            _release(adapter3)
            return MemoryUsage(local.CurrentUsage, local.Budget, non_local.CurrentUsage, non_local.Budget)
    finally:
        _release(factory)
    return None
