"""
EFV · Backend Loader
====================
Loads the compiled native extensions (Rust + C++ GPU) and exposes them
under a single ``efv.backend`` namespace.  Falls back gracefully to the
pure-Python implementation when native extensions are unavailable.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Optional

__all__ = ["core", "gpu", "GPU_AVAILABLE", "BACKEND_NAME"]

# ── Paths ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

# ── 1. Load Rust core (PyO3 extension) ───────────────────────────────────

core: Optional[object] = None
try:
    import efv_core as core          # built by maturin
except ImportError:
    try:
        # Development: look next to the package dir
        sys.path.insert(0, str(_HERE.parent))
        import efv_core as core
    except ImportError:
        core = None

if core is None:
    import importlib.util, warnings
    warnings.warn(
        "[EFV] efv_core (Rust) not found — falling back to pure Python. "
        "Run `maturin develop` or `pip install efv` to enable the fast path.",
        RuntimeWarning,
        stacklevel=3,
    )

# ── 2. Load C++ GPU shared library (ctypes) ───────────────────────────────

def _find_gpu_lib() -> Optional[Path]:
    """Search common locations for efv_gpu shared library."""
    system = platform.system()
    names  = {
        "Windows": ["efv_gpu.dll"],
        "Darwin":  ["libefv_gpu.dylib", "efv_gpu.dylib"],
    }.get(system, ["libefv_gpu.so", "efv_gpu.so"])

    search = [
        _HERE,                          # installed next to Python package
        _HERE.parent,                   # repo root
        _HERE.parent / "build",
        _HERE.parent / "gpu" / "build",
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "bin",       # Windows conda
    ]

    for directory in search:
        for name in names:
            p = directory / name
            if p.exists():
                return p
    return None


class _GpuLib:
    """Thin ctypes wrapper around efv_gpu shared library."""

    def __init__(self, path: Path):
        self._lib = ctypes.CDLL(str(path))
        self._setup_signatures()
        ret = self._lib.efv_gpu_init()
        self.backend_name: str = self._lib.efv_gpu_backend_name().decode()
        self.available: bool   = bool(self._lib.efv_gpu_available())

    def _setup_signatures(self):
        lib = self._lib

        lib.efv_gpu_init.restype         = ctypes.c_int
        lib.efv_gpu_backend_name.restype = ctypes.c_char_p
        lib.efv_gpu_available.restype    = ctypes.c_int
        lib.efv_gpu_shutdown.restype     = None

        uint8_p = ctypes.POINTER(ctypes.c_uint8)
        int16_p = ctypes.POINTER(ctypes.c_int16)

        for fn_name in ("efv_gpu_resize_bilinear", "efv_gpu_resize_nearest"):
            fn = getattr(lib, fn_name)
            fn.restype  = ctypes.c_int
            fn.argtypes = [
                uint8_p, uint8_p,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
            ]

        lib.efv_gpu_composite_over.restype  = ctypes.c_int
        lib.efv_gpu_composite_over.argtypes = [
            uint8_p, ctypes.c_uint32, ctypes.c_uint32,
            uint8_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
        ]

        for fn_name in ("efv_gpu_brightness", "efv_gpu_contrast", "efv_gpu_saturation"):
            fn = getattr(lib, fn_name)
            fn.restype  = ctypes.c_int
            fn.argtypes = [uint8_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]

        for fn_name in ("efv_gpu_blur_horizontal", "efv_gpu_blur_vertical"):
            fn = getattr(lib, fn_name)
            fn.restype  = ctypes.c_int
            fn.argtypes = [
                uint8_p, uint8_p,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ]

        lib.efv_gpu_pcm_volume.restype  = None
        lib.efv_gpu_pcm_volume.argtypes = [int16_p, ctypes.c_size_t, ctypes.c_float]

        lib.efv_gpu_pcm_mix.restype  = None
        lib.efv_gpu_pcm_mix.argtypes = [
            int16_p, ctypes.c_size_t,
            int16_p, ctypes.c_size_t,
            int16_p,
            ctypes.c_float, ctypes.c_float,
        ]

    # ── Convenience wrappers ──────────────────────────────────────────────

    def resize_bilinear(self, rgba: bytes, sw: int, sh: int,
                        dw: int, dh: int) -> bytes:
        dst  = (ctypes.c_uint8 * (dw * dh * 4))()
        src_ = (ctypes.c_uint8 * len(rgba))(*rgba)
        self._lib.efv_gpu_resize_bilinear(src_, dst, sw, sh, dw, dh)
        return bytes(dst)

    def resize_nearest(self, rgba: bytes, sw: int, sh: int,
                       dw: int, dh: int) -> bytes:
        dst  = (ctypes.c_uint8 * (dw * dh * 4))()
        src_ = (ctypes.c_uint8 * len(rgba))(*rgba)
        self._lib.efv_gpu_resize_nearest(src_, dst, sw, sh, dw, dh)
        return bytes(dst)

    def composite_over(self, dst_rgba: bytearray,
                       dst_w: int, dst_h: int,
                       src_rgba: bytes,
                       src_w: int, src_h: int,
                       x_off: int, y_off: int,
                       opacity: float = 1.0) -> None:
        """Composite in-place into dst_rgba (bytearray)."""
        dst_ = (ctypes.c_uint8 * len(dst_rgba)).from_buffer(dst_rgba)
        src_ = (ctypes.c_uint8 * len(src_rgba))(*src_rgba)
        self._lib.efv_gpu_composite_over(
            dst_, dst_w, dst_h,
            src_, src_w, src_h,
            x_off, y_off, ctypes.c_float(opacity),
        )

    def brightness(self, rgba: bytearray, w: int, h: int, factor: float) -> None:
        buf = (ctypes.c_uint8 * len(rgba)).from_buffer(rgba)
        self._lib.efv_gpu_brightness(buf, w, h, ctypes.c_float(factor))

    def contrast(self, rgba: bytearray, w: int, h: int, factor: float) -> None:
        buf = (ctypes.c_uint8 * len(rgba)).from_buffer(rgba)
        self._lib.efv_gpu_contrast(buf, w, h, ctypes.c_float(factor))

    def saturation(self, rgba: bytearray, w: int, h: int, factor: float) -> None:
        buf = (ctypes.c_uint8 * len(rgba)).from_buffer(rgba)
        self._lib.efv_gpu_saturation(buf, w, h, ctypes.c_float(factor))

    def shutdown(self) -> None:
        self._lib.efv_gpu_shutdown()


gpu: Optional[_GpuLib] = None
BACKEND_NAME: str       = "CPU-Python"
GPU_AVAILABLE: bool     = False

_gpu_path = _find_gpu_lib()
if _gpu_path is not None:
    try:
        gpu           = _GpuLib(_gpu_path)
        BACKEND_NAME  = gpu.backend_name
        GPU_AVAILABLE = gpu.available
    except Exception as exc:
        import warnings
        warnings.warn(
            f"[EFV] Failed to load efv_gpu from {_gpu_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

if core is not None and gpu is None:
    BACKEND_NAME = "CPU-Rust"
elif core is None and gpu is None:
    BACKEND_NAME = "CPU-Python"
