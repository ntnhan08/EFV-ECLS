"""EFV · Filter Ops – blur, sharpen, denoise, vignette (pure-Python fallback)."""
from __future__ import annotations

import math


def apply_blur(frame: dict, radius: int) -> dict:
    """Simple box-blur fallback (very fast, lower quality than Gaussian)."""
    if radius <= 0:
        return frame
    w, h   = frame["width"], frame["height"]
    src    = frame["rgba"]
    temp   = _box_pass_h(src, w, h, radius)
    result = _box_pass_v(temp, w, h, radius)
    return dict(rgba=result, width=w, height=h)


def _box_pass_h(src: bytearray, w: int, h: int, r: int) -> bytearray:
    out = bytearray(len(src))
    for y in range(h):
        row_off = y * w
        for x in range(w):
            acc = [0, 0, 0, 0]
            cnt = 0
            for kx in range(max(0, x-r), min(w, x+r+1)):
                i = (row_off + kx) * 4
                for c in range(4): acc[c] += src[i+c]
                cnt += 1
            i = (row_off + x) * 4
            for c in range(4): out[i+c] = acc[c] // cnt
    return out


def _box_pass_v(src: bytearray, w: int, h: int, r: int) -> bytearray:
    out = bytearray(len(src))
    for x in range(w):
        for y in range(h):
            acc = [0, 0, 0, 0]
            cnt = 0
            for ky in range(max(0, y-r), min(h, y+r+1)):
                i = (ky * w + x) * 4
                for c in range(4): acc[c] += src[i+c]
                cnt += 1
            i = (y * w + x) * 4
            for c in range(4): out[i+c] = acc[c] // cnt
    return out


def apply_sharpen(frame: dict, strength: float) -> dict:
    """Unsharp mask: sharpen = original + strength * (original - blurred)."""
    w, h     = frame["width"], frame["height"]
    src      = frame["rgba"]
    blurred  = apply_blur(frame, 1)["rgba"]
    out      = bytearray(len(src))
    si       = int(strength * 256)

    n = w * h
    for i in range(n):
        for c in range(3):
            v = int(src[i*4+c]) + ((int(src[i*4+c]) - int(blurred[i*4+c])) * si >> 8)
            out[i*4+c] = max(0, min(255, v))
        out[i*4+3] = src[i*4+3]
    return dict(rgba=out, width=w, height=h)


def apply_denoise(frame: dict, strength: float) -> dict:
    """Median-3×3 denoising (simplified: uses local average for speed)."""
    # For simplicity use a slight blur as denoising
    radius = max(1, int(strength * 1.5))
    return apply_blur(frame, radius)


def apply_vignette(frame: dict, strength: float, radius: float) -> dict:
    """Dark oval vignette effect."""
    w, h  = frame["width"], frame["height"]
    src   = frame["rgba"]
    out   = bytearray(len(src))
    cx, cy = w / 2.0, h / 2.0
    max_d  = math.sqrt(cx * cx + cy * cy)
    r_pix  = radius * max_d

    for y in range(h):
        for x in range(w):
            dx = x - cx; dy = y - cy
            d  = math.sqrt(dx*dx + dy*dy)
            if d <= r_pix:
                alpha = 1.0
            else:
                alpha = max(0.0, 1.0 - (d - r_pix) / (max_d - r_pix + 1e-6))
            # Apply: mix pixel toward black
            factor = 1.0 - strength * (1.0 - alpha)
            idx    = (y * w + x) * 4
            out[idx]   = max(0, min(255, int(src[idx]   * factor)))
            out[idx+1] = max(0, min(255, int(src[idx+1] * factor)))
            out[idx+2] = max(0, min(255, int(src[idx+2] * factor)))
            out[idx+3] = src[idx+3]
    return dict(rgba=out, width=w, height=h)
