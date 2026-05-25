"""EFV · Color Ops – brightness, contrast, saturation, greyscale, LUT (pure-Python)."""
from __future__ import annotations

from typing import Optional


def apply_brightness(frame: dict, factor: float) -> dict:
    fi  = int(factor * 256)
    src = frame["rgba"]
    out = bytearray(len(src))
    n   = len(src) // 4
    for i in range(n):
        out[i*4]   = max(0, min(255, (src[i*4]   * fi) >> 8))
        out[i*4+1] = max(0, min(255, (src[i*4+1] * fi) >> 8))
        out[i*4+2] = max(0, min(255, (src[i*4+2] * fi) >> 8))
        out[i*4+3] = src[i*4+3]
    return dict(rgba=out, width=frame["width"], height=frame["height"])


def apply_contrast(frame: dict, factor: float) -> dict:
    fi  = int(factor * 256)
    src = frame["rgba"]
    out = bytearray(len(src))
    n   = len(src) // 4
    for i in range(n):
        for c in range(3):
            v = (((int(src[i*4+c]) - 128) * fi) >> 8) + 128
            out[i*4+c] = max(0, min(255, v))
        out[i*4+3] = src[i*4+3]
    return dict(rgba=out, width=frame["width"], height=frame["height"])


def apply_saturation(frame: dict, factor: float) -> dict:
    fi  = int(factor * 256)
    src = frame["rgba"]
    out = bytearray(len(src))
    n   = len(src) // 4
    for i in range(n):
        r = src[i*4]; g = src[i*4+1]; b = src[i*4+2]
        luma = (r * 77 + g * 150 + b * 29) >> 8
        out[i*4]   = max(0, min(255, luma + (((r - luma) * fi) >> 8)))
        out[i*4+1] = max(0, min(255, luma + (((g - luma) * fi) >> 8)))
        out[i*4+2] = max(0, min(255, luma + (((b - luma) * fi) >> 8)))
        out[i*4+3] = src[i*4+3]
    return dict(rgba=out, width=frame["width"], height=frame["height"])


def apply_greyscale(frame: dict) -> dict:
    src = frame["rgba"]
    out = bytearray(len(src))
    n   = len(src) // 4
    for i in range(n):
        r = src[i*4]; g = src[i*4+1]; b = src[i*4+2]
        grey = (r * 77 + g * 150 + b * 29) >> 8
        out[i*4]   = grey
        out[i*4+1] = grey
        out[i*4+2] = grey
        out[i*4+3] = src[i*4+3]
    return dict(rgba=out, width=frame["width"], height=frame["height"])


def apply_lut(frame: dict, lut: bytes) -> dict:
    """Apply a 768-byte RGB LUT."""
    if len(lut) < 768:
        return frame
    src = frame["rgba"]
    out = bytearray(len(src))
    n   = len(src) // 4
    for i in range(n):
        out[i*4]   = lut[src[i*4]]
        out[i*4+1] = lut[256 + src[i*4+1]]
        out[i*4+2] = lut[512 + src[i*4+2]]
        out[i*4+3] = src[i*4+3]
    return dict(rgba=out, width=frame["width"], height=frame["height"])


# ── LUT builder ───────────────────────────────────────────────────────────

def build_gamma_lut(gamma_r: float = 1.0,
                    gamma_g: float = 1.0,
                    gamma_b: float = 1.0) -> bytes:
    """Build a 768-byte gamma LUT for use with ``apply_lut``."""
    lut = bytearray(768)
    for i in range(256):
        v = i / 255.0
        lut[i]       = max(0, min(255, int(v ** (1.0 / max(gamma_r, 1e-6)) * 255 + 0.5)))
        lut[256 + i] = max(0, min(255, int(v ** (1.0 / max(gamma_g, 1e-6)) * 255 + 0.5)))
        lut[512 + i] = max(0, min(255, int(v ** (1.0 / max(gamma_b, 1e-6)) * 255 + 0.5)))
    return bytes(lut)


# ── JPEG fallback ─────────────────────────────────────────────────────────

def encode_jpeg_py(rgb: bytes, width: int, height: int, quality: int = 85) -> bytes:
    """Encode RGB bytes to JPEG using Pillow or a minimal fallback."""
    try:
        from PIL import Image
        import io
        img = Image.frombytes("RGB", (width, height), rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        pass
    # Last resort: save as uncompressed BMP-in-AVI (RIFF DIB)
    return _rgb_to_bmp_bytes(rgb, width, height)


def _rgb_to_bmp_bytes(rgb: bytes, w: int, h: int) -> bytes:
    """Minimal 24-bit uncompressed BMP (used only as last resort)."""
    import struct
    row_size = (w * 3 + 3) & ~3
    data_size = row_size * h
    file_size = 54 + data_size
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib    = struct.pack("<IiiHHIIiiII", 40, w, -h, 1, 24, 0, data_size, 0, 0, 0, 0)
    rows   = bytearray()
    for y in range(h):
        row_start = y * w * 3
        row = bytearray()
        for x in range(w):
            r = rgb[row_start + x*3]
            g = rgb[row_start + x*3+1]
            b = rgb[row_start + x*3+2]
            row += bytes([b, g, r])
        row += b"\x00" * (row_size - w * 3)
        rows += row
    return header + dib + bytes(rows)
