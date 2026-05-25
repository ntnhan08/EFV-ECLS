"""
EFV · Pipeline
==============
Lightweight dataclasses describing each pending video operation.
The pipeline is a simple list appended to by Video's fluent methods
and consumed by the Executor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class OpKind(Enum):
    # ── Audio ──────────────────────────────────────────────────────────
    MUTE            = auto()
    ADD_AUDIO       = auto()
    CHANGE_VOLUME   = auto()
    FADE_AUDIO      = auto()

    # ── Visual overlays ────────────────────────────────────────────────
    ADD_LOGO        = auto()
    ADD_WATERMARK   = auto()
    ADD_TEXT        = auto()
    ADD_SUBTITLE    = auto()

    # ── Geometry ───────────────────────────────────────────────────────
    RESIZE          = auto()
    CROP            = auto()
    ZOOM            = auto()
    ROTATE          = auto()
    FLIP            = auto()

    # ── Color ──────────────────────────────────────────────────────────
    BRIGHTNESS      = auto()
    CONTRAST        = auto()
    SATURATION      = auto()
    APPLY_LUT       = auto()
    GREYSCALE       = auto()

    # ── Filters ────────────────────────────────────────────────────────
    BLUR            = auto()
    SHARPEN         = auto()
    DENOISE         = auto()
    VIGNETTE        = auto()

    # ── Timeline ───────────────────────────────────────────────────────
    CUT             = auto()
    SPEED           = auto()
    REVERSE         = auto()
    LOOP            = auto()
    TRIM            = auto()

    # ── Transitions ────────────────────────────────────────────────────
    FADE_IN         = auto()
    FADE_OUT        = auto()
    CROSSFADE       = auto()


@dataclass(frozen=True)
class Op:
    """Immutable descriptor for a single pipeline operation."""
    kind:    OpKind
    params:  dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:          # pragma: no cover
        kv = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"Op({self.kind.name}{', ' + kv if kv else ''})"


# ── Factory helpers (used internally by Video) ────────────────────────────

def op_mute()                                              -> Op: return Op(OpKind.MUTE)
def op_add_audio(path: str, volume: float = 1.0,
                 start_sec: float = 0.0)                  -> Op:
    return Op(OpKind.ADD_AUDIO, dict(path=path, volume=volume, start_sec=start_sec))

def op_change_volume(factor: float)                        -> Op:
    return Op(OpKind.CHANGE_VOLUME, dict(factor=factor))

def op_fade_audio(in_sec: float = 0.0,
                  out_sec: float = 0.0)                    -> Op:
    return Op(OpKind.FADE_AUDIO, dict(in_sec=in_sec, out_sec=out_sec))

def op_add_logo(path: str, position: str = "topright",
                scale: float = 1.0, opacity: float = 1.0,
                margin: int = 10,
                x: Optional[int] = None,
                y: Optional[int] = None)                   -> Op:
    return Op(OpKind.ADD_LOGO, dict(path=path, position=position,
              scale=scale, opacity=opacity, margin=margin, x=x, y=y))

def op_add_watermark(text: str, opacity: float = 0.3,
                     font_size: int = 36,
                     color: tuple = (255, 255, 255))       -> Op:
    return Op(OpKind.ADD_WATERMARK, dict(text=text, opacity=opacity,
              font_size=font_size, color=color))

def op_add_text(text: str, x: int = 10, y: int = 10,
                font_size: int = 24,
                color: tuple = (255, 255, 255),
                bg_color: Optional[tuple] = None,
                duration_sec: Optional[float] = None,
                start_sec: float = 0.0)                    -> Op:
    return Op(OpKind.ADD_TEXT, dict(text=text, x=x, y=y,
              font_size=font_size, color=color, bg_color=bg_color,
              duration_sec=duration_sec, start_sec=start_sec))

def op_add_subtitle(text: str, start_sec: float,
                    end_sec: float,
                    font_size: int = 28,
                    color: tuple = (255, 255, 255),
                    bg_color: tuple = (0, 0, 0, 160),
                    position: str = "bottom")              -> Op:
    return Op(OpKind.ADD_SUBTITLE, dict(text=text, start_sec=start_sec,
              end_sec=end_sec, font_size=font_size, color=color,
              bg_color=bg_color, position=position))

def op_resize(width: int, height: int,
              method: str = "bilinear")                    -> Op:
    return Op(OpKind.RESIZE, dict(width=width, height=height, method=method))

def op_crop(x: int, y: int,
            width: int, height: int)                       -> Op:
    return Op(OpKind.CROP, dict(x=x, y=y, width=width, height=height))

def op_zoom(factor: float, smooth: bool = True)            -> Op:
    return Op(OpKind.ZOOM, dict(factor=factor, smooth=smooth))

def op_rotate(degrees: float,
              expand: bool = True)                         -> Op:
    return Op(OpKind.ROTATE, dict(degrees=degrees, expand=expand))

def op_flip(horizontal: bool = True,
            vertical: bool = False)                        -> Op:
    return Op(OpKind.FLIP, dict(horizontal=horizontal, vertical=vertical))

def op_brightness(factor: float)                           -> Op:
    return Op(OpKind.BRIGHTNESS, dict(factor=factor))

def op_contrast(factor: float)                             -> Op:
    return Op(OpKind.CONTRAST, dict(factor=factor))

def op_saturation(factor: float)                           -> Op:
    return Op(OpKind.SATURATION, dict(factor=factor))

def op_apply_lut(lut: bytes)                               -> Op:
    return Op(OpKind.APPLY_LUT, dict(lut=lut))

def op_greyscale()                                         -> Op:
    return Op(OpKind.GREYSCALE)

def op_blur(radius: int = 2)                               -> Op:
    return Op(OpKind.BLUR, dict(radius=radius))

def op_sharpen(strength: float = 1.0)                      -> Op:
    return Op(OpKind.SHARPEN, dict(strength=strength))

def op_denoise(strength: float = 1.0)                      -> Op:
    return Op(OpKind.DENOISE, dict(strength=strength))

def op_vignette(strength: float = 0.5,
                radius: float = 0.7)                       -> Op:
    return Op(OpKind.VIGNETTE, dict(strength=strength, radius=radius))

def op_cut(start_sec: float, end_sec: float)               -> Op:
    return Op(OpKind.CUT, dict(start_sec=start_sec, end_sec=end_sec))

def op_speed(factor: float)                                -> Op:
    return Op(OpKind.SPEED, dict(factor=factor))

def op_reverse()                                           -> Op: return Op(OpKind.REVERSE)

def op_loop(times: int = 2)                                -> Op:
    return Op(OpKind.LOOP, dict(times=times))

def op_trim(start_sec: float = 0.0,
            end_sec: Optional[float] = None)               -> Op:
    return Op(OpKind.TRIM, dict(start_sec=start_sec, end_sec=end_sec))

def op_fade_in(duration_sec: float = 1.0)                  -> Op:
    return Op(OpKind.FADE_IN, dict(duration_sec=duration_sec))

def op_fade_out(duration_sec: float = 1.0)                 -> Op:
    return Op(OpKind.FADE_OUT, dict(duration_sec=duration_sec))

def op_crossfade(other_path: str,
                 duration_sec: float = 1.0)                -> Op:
    return Op(OpKind.CROSSFADE, dict(other_path=other_path,
              duration_sec=duration_sec))
