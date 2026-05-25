"""EFV · Visual Overlay Ops – pure-Python fallback implementations."""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple


# ── Image loader ─────────────────────────────────────────────────────────

def load_image(path: str) -> Optional[dict]:
    """
    Load PNG or JPEG from *path* into a frame dict.
    Tries: efv_core Rust codec → Pillow → PPM/PBM fallback.
    """
    from ..backend import core

    path = str(path)
    if not os.path.exists(path):
        import warnings
        warnings.warn(f"[EFV] Image not found: {path!r}")
        return None

    ext = os.path.splitext(path)[1].lower()
    data = open(path, "rb").read()

    if core is not None:
        try:
            if ext in (".png",):
                rgba, w, h = core.py_decode_png(data)
                return dict(rgba=bytearray(rgba), width=w, height=h)
            elif ext in (".jpg", ".jpeg"):
                rgb, w, h = core.py_decode_jpeg(data)
                n    = w * h
                rgba = bytearray(n * 4)
                for i in range(n):
                    rgba[i*4]   = rgb[i*3]
                    rgba[i*4+1] = rgb[i*3+1]
                    rgba[i*4+2] = rgb[i*3+2]
                    rgba[i*4+3] = 255
                return dict(rgba=rgba, width=w, height=h)
        except Exception:
            pass

    # Pillow fallback
    try:
        from PIL import Image
        img  = Image.open(path).convert("RGBA")
        w, h = img.size
        return dict(rgba=bytearray(img.tobytes()), width=w, height=h)
    except ImportError:
        pass

    import warnings
    warnings.warn(f"[EFV] Could not load image {path!r}: install Pillow for broader format support.")
    return None


# ── Logo / overlay ────────────────────────────────────────────────────────

def apply_logo(frame: dict, logo: dict, params: dict) -> dict:
    """Composite a logo onto the video frame."""
    from ..backend import core, GPU_AVAILABLE, gpu

    scale   = params.get("scale",   1.0)
    opacity = params.get("opacity", 1.0)
    margin  = params.get("margin",  10)
    pos     = params.get("position", "topright")
    px      = params.get("x")
    py_     = params.get("y")

    ovl = logo
    if abs(scale - 1.0) > 1e-4:
        ovl = _scale_image(ovl, scale)

    if core is not None:
        vf  = core.Frame.from_rgba(bytes(frame["rgba"]), frame["width"], frame["height"])
        lf  = core.Frame.from_rgba(bytes(ovl["rgba"]),   ovl["width"],   ovl["height"])
        x_off, y_off = _calc_position(pos, frame["width"], frame["height"],
                                       ovl["width"], ovl["height"], margin, px, py_)
        result = core.py_overlay_image(vf, lf, pos, 1.0, opacity, margin,
                                        px if px is not None else -999999,
                                        py_ if py_ is not None else -999999)
        return dict(rgba=bytearray(result.to_rgba()),
                    width=frame["width"], height=frame["height"])

    x_off, y_off = _calc_position(pos, frame["width"], frame["height"],
                                   ovl["width"], ovl["height"], margin, px, py_)
    return _composite(frame, ovl, x_off, y_off, opacity)


def apply_watermark(frame: dict, params: dict) -> dict:
    """Render a text watermark diagonally across the frame."""
    text      = params["text"]
    opacity   = params.get("opacity",   0.3)
    font_size = params.get("font_size", 36)
    color     = params.get("color",     (255, 255, 255))

    w, h = frame["width"], frame["height"]
    # Draw text centred diagonally
    overlay = _make_text_frame(text, w, h, font_size, color,
                                angle=-30, opacity=opacity, centred=True)
    if overlay is None:
        return frame
    return _composite(frame, overlay, 0, 0, 1.0)


def apply_text(frame: dict, params: dict) -> dict:
    """Render static text onto the frame."""
    text      = params["text"]
    x         = params.get("x", 10)
    y         = params.get("y", 10)
    font_size = params.get("font_size", 24)
    color     = params.get("color",     (255, 255, 255))
    bg_color  = params.get("bg_color")

    overlay = _make_text_image(text, font_size, color, bg_color)
    if overlay is None:
        return frame
    return _composite(frame, overlay, x, y, 1.0)


def apply_subtitle(frame: dict, params: dict) -> dict:
    """Render a subtitle bar at the bottom (or top) of the frame."""
    text      = params["text"]
    font_size = params.get("font_size", 28)
    color     = params.get("color",     (255, 255, 255))
    bg_color  = params.get("bg_color",  (0, 0, 0, 160))
    position  = params.get("position",  "bottom")

    w, h    = frame["width"], frame["height"]
    bar_h   = font_size + 16
    bar_w   = w
    bar_rgba = bytearray(bar_w * bar_h * 4)

    # Fill background
    if len(bg_color) == 4:
        br, bg, bb, ba = bg_color
    else:
        br, bg, bb, ba = (*bg_color, 200)
    for i in range(bar_w * bar_h):
        bar_rgba[i*4]   = br
        bar_rgba[i*4+1] = bg
        bar_rgba[i*4+2] = bb
        bar_rgba[i*4+3] = ba

    bar = dict(rgba=bar_rgba, width=bar_w, height=bar_h)
    bar = _draw_text_on_frame(bar, text, 8, 8, font_size, color)

    y_off = h - bar_h - 4 if position == "bottom" else 4
    return _composite(frame, bar, 0, y_off, 1.0)


# ── Compositing helper ────────────────────────────────────────────────────

def _composite(dst: dict, src: dict, x_off: int, y_off: int, opacity: float) -> dict:
    dw, dh = dst["width"], dst["height"]
    sw, sh = src["width"], src["height"]
    d      = bytearray(dst["rgba"])
    s      = src["rgba"]

    x0 = max(0, x_off)
    y0 = max(0, y_off)
    x1 = min(dw, x_off + sw)
    y1 = min(dh, y_off + sh)

    op_i = int(opacity * 255)

    for dy in range(y0, y1):
        sy = dy - y_off
        for dx in range(x0, x1):
            sx = dx - x_off
            si = (sy * sw + sx) * 4
            di = (dy * dw + dx) * 4
            sa = min(255, (s[si+3] * op_i + 127) >> 8)
            if sa == 0:
                continue
            if sa >= 254:
                d[di]   = s[si]
                d[di+1] = s[si+1]
                d[di+2] = s[si+2]
                d[di+3] = 255
                continue
            inv = 255 - sa
            da  = d[di+3]
            for c in range(3):
                d[di+c] = (s[si+c] * sa + d[di+c] * inv) >> 8
            d[di+3] = min(255, sa + ((da * inv) >> 8))

    return dict(rgba=d, width=dw, height=dh)


# ── Text rendering helpers ────────────────────────────────────────────────

def _make_text_image(
    text: str, font_size: int,
    color: tuple, bg_color: Optional[tuple]
) -> Optional[dict]:
    """Create a small RGBA frame containing rendered text."""
    try:
        return _pillow_text_image(text, font_size, color, bg_color)
    except Exception:
        return _bitmap_text_image(text, font_size, color, bg_color)


def _pillow_text_image(text, font_size, color, bg_color):
    from PIL import Image, ImageDraw, ImageFont
    dummy = Image.new("RGBA", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0] + 8
    th   = bbox[3] - bbox[1] + 8
    if bg_color:
        bg = (*bg_color[:3], bg_color[3] if len(bg_color) > 3 else 200)
    else:
        bg = (0, 0, 0, 0)
    img  = Image.new("RGBA", (tw, th), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), text, font=font, fill=(*color[:3], 255))
    return dict(rgba=bytearray(img.tobytes()), width=tw, height=th)


def _bitmap_text_image(text, font_size, color, bg_color):
    """Extremely simple 8×8 pixel-font fallback."""
    FONT = _get_8x8_font()
    cw, ch = 8, 8
    scale  = max(1, font_size // 8)
    pad    = 4
    tw     = len(text) * cw * scale + pad * 2
    th     = ch * scale + pad * 2
    bg     = (0, 0, 0, 0) if bg_color is None else tuple(bg_color[:4]) + (200,) * (4 - len(bg_color))
    rgba   = bytearray(tw * th * 4)
    for i in range(tw * th):
        rgba[i*4]   = bg[0]
        rgba[i*4+1] = bg[1]
        rgba[i*4+2] = bg[2]
        rgba[i*4+3] = bg[3] if len(bg) > 3 else 0

    for ci, ch_code in enumerate(text):
        glyph = FONT.get(ch_code, FONT.get("?", [0]*8))
        for row in range(8):
            bits = glyph[row]
            for col in range(8):
                if (bits >> (7 - col)) & 1:
                    for sy in range(scale):
                        for sx in range(scale):
                            px = pad + ci * cw * scale + col * scale + sx
                            py = pad + row * scale + sy
                            idx = (py * tw + px) * 4
                            if 0 <= idx < len(rgba) - 3:
                                rgba[idx]   = color[0]
                                rgba[idx+1] = color[1]
                                rgba[idx+2] = color[2]
                                rgba[idx+3] = 255
    return dict(rgba=rgba, width=tw, height=th)


def _draw_text_on_frame(frame: dict, text: str, x: int, y: int,
                         font_size: int, color: tuple) -> dict:
    overlay = _make_text_image(text, font_size, color, None)
    if overlay is None:
        return frame
    return _composite(frame, overlay, x, y, 1.0)


def _make_text_frame(text, w, h, font_size, color,
                      angle=0, opacity=1.0, centred=False) -> Optional[dict]:
    """Full-frame text overlay (used for watermarks)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = (w - tw) // 2 if centred else 10
        ty = (h - th) // 2 if centred else 10
        draw.text((tx, ty), text, font=font,
                  fill=(*color[:3], int(opacity * 255)))
        return dict(rgba=bytearray(img.tobytes()), width=w, height=h)
    except Exception:
        return None


# ── Position helper ───────────────────────────────────────────────────────

def _calc_position(pos, vw, vh, ow, oh, margin, explicit_x, explicit_y):
    if explicit_x is not None and explicit_y is not None:
        return explicit_x, explicit_y
    return {
        "center":      ((vw-ow)//2,      (vh-oh)//2),
        "topright":    (vw-ow-margin,    margin),
        "topleft":     (margin,          margin),
        "bottomright": (vw-ow-margin,    vh-oh-margin),
        "bottomleft":  (margin,          vh-oh-margin),
        "top":         ((vw-ow)//2,      margin),
        "bottom":      ((vw-ow)//2,      vh-oh-margin),
        "left":        (margin,          (vh-oh)//2),
        "right":       (vw-ow-margin,    (vh-oh)//2),
    }.get(pos, (vw-ow-margin, margin))


def _scale_image(img: dict, factor: float) -> dict:
    """Bilinear scale an image dict."""
    sw, sh = img["width"], img["height"]
    dw = max(1, int(sw * factor))
    dh = max(1, int(sh * factor))
    from ..backend import core
    if core is not None:
        f   = core.Frame.from_rgba(bytes(img["rgba"]), sw, sh)
        f2  = core.py_resize_bilinear(f, dw, dh)
        return dict(rgba=bytearray(f2.to_rgba()), width=dw, height=dh)
    from .transform import resize_bilinear_py
    return resize_bilinear_py(img, dw, dh)


# ── Minimal 8×8 bitmap font ───────────────────────────────────────────────

def _get_8x8_font() -> dict:
    """Returns a very small subset of an 8×8 font for the pure-Python fallback."""
    # Each character is 8 rows of bits (msb = leftmost pixel)
    # Only printable ASCII included; missing chars fall back to '?'
    return {
        ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '?': [0x7E,0x81,0x81,0x02,0x08,0x00,0x08,0x00],
        'A': [0x18,0x24,0x42,0x7E,0x42,0x42,0x42,0x00],
        'B': [0x7C,0x42,0x42,0x7C,0x42,0x42,0x7C,0x00],
        'C': [0x3C,0x42,0x40,0x40,0x40,0x42,0x3C,0x00],
        'D': [0x7C,0x42,0x42,0x42,0x42,0x42,0x7C,0x00],
        'E': [0x7E,0x40,0x40,0x7C,0x40,0x40,0x7E,0x00],
        'F': [0x7E,0x40,0x40,0x7C,0x40,0x40,0x40,0x00],
        'V': [0x42,0x42,0x42,0x42,0x24,0x18,0x00,0x00],
        'a': [0x00,0x3C,0x02,0x3E,0x42,0x42,0x3E,0x00],
        'e': [0x00,0x3C,0x42,0x7E,0x40,0x42,0x3C,0x00],
        'f': [0x1C,0x20,0x20,0x7C,0x20,0x20,0x20,0x00],
        'v': [0x00,0x42,0x42,0x42,0x24,0x18,0x00,0x00],
        '0': [0x3C,0x42,0x46,0x4A,0x52,0x62,0x3C,0x00],
        '1': [0x08,0x18,0x28,0x08,0x08,0x08,0x3E,0x00],
        '2': [0x3C,0x42,0x02,0x0C,0x30,0x40,0x7E,0x00],
        '3': [0x3C,0x42,0x02,0x1C,0x02,0x42,0x3C,0x00],
        '.': [0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00],
        ':': [0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00],
        '-': [0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00],
        '_': [0x00,0x00,0x00,0x00,0x00,0x00,0x7E,0x00],
        '@': [0x3C,0x42,0x9A,0xAA,0xAE,0x40,0x3E,0x00],
    }
