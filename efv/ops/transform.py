"""EFV · Transform Ops – crop, rotate, flip, resize (pure-Python fallback)."""
from __future__ import annotations

import math
import struct
from typing import Optional


# ── Crop ──────────────────────────────────────────────────────────────────

def apply_crop(frame: dict, cx: int, cy: int, cw: int, ch: int) -> dict:
    fw, fh = frame["width"], frame["height"]
    cx = max(0, min(cx, fw)); cy = max(0, min(cy, fh))
    cw = max(1, min(cw, fw - cx)); ch = max(1, min(ch, fh - cy))
    src  = frame["rgba"]
    out  = bytearray(cw * ch * 4)
    for y in range(ch):
        for x in range(cw):
            si = ((cy + y) * fw + (cx + x)) * 4
            di = (y * cw + x) * 4
            out[di:di+4] = src[si:si+4]
    return dict(rgba=out, width=cw, height=ch)


# ── Flip ──────────────────────────────────────────────────────────────────

def apply_flip(frame: dict, horizontal: bool, vertical: bool) -> dict:
    w, h = frame["width"], frame["height"]
    src  = frame["rgba"]
    out  = bytearray(len(src))
    for y in range(h):
        for x in range(w):
            sx = (w - 1 - x) if horizontal else x
            sy = (h - 1 - y) if vertical   else y
            si = (sy * w + sx) * 4
            di = (y  * w + x ) * 4
            out[di:di+4] = src[si:si+4]
    return dict(rgba=out, width=w, height=h)


# ── Rotate ────────────────────────────────────────────────────────────────

def apply_rotate(frame: dict, degrees: float, expand: bool) -> dict:
    # Normalise
    degrees = degrees % 360

    # Fast paths for 90° multiples
    if degrees == 0:
        return frame
    if degrees == 90:
        return _rotate_90(frame)
    if degrees == 180:
        return _rotate_180(frame)
    if degrees == 270:
        return _rotate_270(frame)

    # General rotation
    rad   = math.radians(degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    w, h  = frame["width"], frame["height"]

    if expand:
        corners = [(0,0),(w,0),(0,h),(w,h)]
        xs = [abs(cos_a)*cx + abs(sin_a)*cy for cx,cy in corners]
        ys = [abs(sin_a)*cx + abs(cos_a)*cy for cx,cy in corners]
        nw = int(max(xs)) + 1
        nh = int(max(ys)) + 1
    else:
        nw, nh = w, h

    cx_src = w  / 2.0
    cy_src = h  / 2.0
    cx_dst = nw / 2.0
    cy_dst = nh / 2.0

    src = frame["rgba"]
    out = bytearray(nw * nh * 4)

    for y in range(nh):
        for x in range(nw):
            dx = x - cx_dst
            dy = y - cy_dst
            sx =  cos_a * dx + sin_a * dy + cx_src
            sy = -sin_a * dx + cos_a * dy + cy_src
            sx_i = int(sx); sy_i = int(sy)
            if 0 <= sx_i < w and 0 <= sy_i < h:
                si = (sy_i * w + sx_i) * 4
                di = (y * nw + x) * 4
                out[di:di+4] = src[si:si+4]

    return dict(rgba=out, width=nw, height=nh)


def _rotate_90(frame: dict) -> dict:
    w, h = frame["width"], frame["height"]
    src  = frame["rgba"]
    out  = bytearray(w * h * 4)
    for y in range(h):
        for x in range(w):
            si = (y * w + x) * 4
            di = (x * h + (h - 1 - y)) * 4
            out[di:di+4] = src[si:si+4]
    return dict(rgba=out, width=h, height=w)


def _rotate_180(frame: dict) -> dict:
    src = frame["rgba"]
    return dict(rgba=bytearray(reversed(bytearray(
        src[i:i+4] for i in range(0, len(src), 4) for _ in [None]
    ))), width=frame["width"], height=frame["height"])


def _rotate_270(frame: dict) -> dict:
    w, h = frame["width"], frame["height"]
    src  = frame["rgba"]
    out  = bytearray(w * h * 4)
    for y in range(h):
        for x in range(w):
            si = (y * w + x) * 4
            di = ((w - 1 - x) * h + y) * 4
            out[di:di+4] = src[si:si+4]
    return dict(rgba=out, width=h, height=w)


# ── Resize (pure Python bilinear) ────────────────────────────────────────

def resize_bilinear_py(frame: dict, dw: int, dh: int) -> dict:
    sw, sh = frame["width"], frame["height"]
    src    = frame["rgba"]
    x_r    = sw / dw
    y_r    = sh / dh
    out    = bytearray(dw * dh * 4)

    for dy in range(dh):
        sy_f = (dy + 0.5) * y_r - 0.5
        sy0  = max(0, min(int(sy_f), sh - 1))
        sy1  = min(sy0 + 1, sh - 1)
        wy1  = sy_f - math.floor(sy_f)
        wy0  = 1.0 - wy1
        for dx in range(dw):
            sx_f = (dx + 0.5) * x_r - 0.5
            sx0  = max(0, min(int(sx_f), sw - 1))
            sx1  = min(sx0 + 1, sw - 1)
            wx1  = sx_f - math.floor(sx_f)
            wx0  = 1.0 - wx1
            i00  = (sy0 * sw + sx0) * 4
            i10  = (sy0 * sw + sx1) * 4
            i01  = (sy1 * sw + sx0) * 4
            i11  = (sy1 * sw + sx1) * 4
            di   = (dy * dw + dx) * 4
            for c in range(4):
                v = (src[i00+c] * wx0 * wy0 +
                     src[i10+c] * wx1 * wy0 +
                     src[i01+c] * wx0 * wy1 +
                     src[i11+c] * wx1 * wy1)
                out[di+c] = max(0, min(255, int(v + 0.5)))
    return dict(rgba=out, width=dw, height=dh)


# ── BGR (AVI raw) decode ──────────────────────────────────────────────────

def bgr_to_rgb(raw: bytes, w: int, h: int) -> bytes:
    """Convert bottom-up BGR to top-down RGB."""
    out = bytearray(w * h * 3)
    for y in range(h):
        src_row = (h - 1 - y) * w
        dst_row = y * w
        for x in range(w):
            si = (src_row + x) * 3
            di = (dst_row + x) * 3
            out[di]   = raw[si+2]
            out[di+1] = raw[si+1]
            out[di+2] = raw[si]
    return bytes(out)


def decode_frame_py(raw: bytes, ftype: str, w: int, h: int) -> dict:
    if ftype == "jpeg":
        rgb = _decode_jpeg_py(raw)
        if rgb is None:
            rgb = bytes(w * h * 3)
        fw, fh = w, h
    elif ftype == "png":
        rgba, fw, fh = _decode_png_py(raw)
        return dict(rgba=bytearray(rgba), width=fw, height=fh)
    else:
        rgb, fw, fh = bgr_to_rgb(raw, w, h), w, h

    n    = fw * fh
    rgba = bytearray(n * 4)
    for i in range(n):
        rgba[i*4]   = rgb[i*3]
        rgba[i*4+1] = rgb[i*3+1]
        rgba[i*4+2] = rgb[i*3+2]
        rgba[i*4+3] = 255
    return dict(rgba=rgba, width=fw, height=fh)


def _decode_jpeg_py(data: bytes) -> Optional[bytes]:
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img.tobytes()
    except Exception:
        return None


def _decode_png_py(data: bytes):
    try:
        from PIL import Image
        import io
        img  = Image.open(io.BytesIO(data)).convert("RGBA")
        w, h = img.size
        return img.tobytes(), w, h
    except Exception:
        return b"", 0, 0


def _pure_avi_meta(path: str) -> dict:
    """Read basic AVI header metadata without Rust."""
    try:
        from ..py_compat.avi_reader import PureAviReader
        r = PureAviReader(path)
        m = r.info()
        r.close()
        return m
    except Exception as exc:
        return {"error": str(exc)}
