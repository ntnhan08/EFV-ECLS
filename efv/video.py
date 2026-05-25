"""
EFV · Video
===========
Main public class.  Every method returns ``self`` for chaining.
Call ``.save(path)`` to execute the pipeline and write the output.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from . import pipeline as _pl
from .pipeline import Op
from .__version__ import __version__


class Video:
    """
    Fluent video-processing interface.

    Example
    -------
    >>> import efv
    >>> (efv.Video("input.avi")
    ...     .mute()
    ...     .add_logo("logo.png", position="topright", opacity=0.8)
    ...     .resize(1280, 720)
    ...     .brightness(1.1)
    ...     .fade_in(0.5)
    ...     .fade_out(0.5)
    ...     .save("output.avi"))
    """

    def __init__(self, path: str, quality: int = 85):
        self._path    = str(path)
        self._ops: List[Op] = []
        self._quality = quality

    # ═══════════════════════════════════════════════════════════════════
    #  AUDIO OPS
    # ═══════════════════════════════════════════════════════════════════

    def mute(self) -> "Video":
        """Remove all audio from the video."""
        return self._add(_pl.op_mute())

    def add_audio(
        self,
        path:      str,
        volume:    float = 1.0,
        start_sec: float = 0.0,
    ) -> "Video":
        """
        Replace or mix in an external audio track.

        Parameters
        ----------
        path      : WAV file to add.
        volume    : Mix volume (1.0 = full).
        start_sec : Start offset inside the added audio.
        """
        return self._add(_pl.op_add_audio(path, volume, start_sec))

    def volume(self, factor: float) -> "Video":
        """
        Scale the existing audio volume.

        Parameters
        ----------
        factor : 0.0 = silence, 1.0 = unchanged, 2.0 = double.
        """
        return self._add(_pl.op_change_volume(factor))

    def fade_audio(
        self,
        in_sec:  float = 0.0,
        out_sec: float = 0.0,
    ) -> "Video":
        """Apply linear audio fade-in / fade-out."""
        return self._add(_pl.op_fade_audio(in_sec, out_sec))

    # ═══════════════════════════════════════════════════════════════════
    #  VISUAL OVERLAYS
    # ═══════════════════════════════════════════════════════════════════

    def add_logo(
        self,
        path:     str,
        position: str   = "topright",
        scale:    float = 1.0,
        opacity:  float = 1.0,
        margin:   int   = 10,
        x:        Optional[int] = None,
        y:        Optional[int] = None,
    ) -> "Video":
        """
        Overlay a PNG/JPEG logo on every frame.

        Parameters
        ----------
        path     : Image file (PNG with alpha recommended).
        position : One of ``topright`` ``topleft`` ``bottomright``
                   ``bottomleft`` ``center`` ``top`` ``bottom``
                   ``left`` ``right``.
        scale    : Resize logo by this factor before compositing.
        opacity  : Alpha multiplier (0.0–1.0).
        margin   : Pixel gap from the edge.
        x, y     : Explicit pixel coordinates (overrides *position*).
        """
        return self._add(_pl.op_add_logo(path, position, scale, opacity, margin, x, y))

    def add_watermark(
        self,
        text:      str,
        opacity:   float         = 0.3,
        font_size: int           = 36,
        color:     Tuple         = (255, 255, 255),
    ) -> "Video":
        """Render a semi-transparent text watermark across every frame."""
        return self._add(_pl.op_add_watermark(text, opacity, font_size, color))

    def add_text(
        self,
        text:         str,
        x:            int            = 10,
        y:            int            = 10,
        font_size:    int            = 24,
        color:        Tuple          = (255, 255, 255),
        bg_color:     Optional[Tuple] = None,
        duration_sec: Optional[float] = None,
        start_sec:    float           = 0.0,
    ) -> "Video":
        """
        Burn text onto frames.

        Parameters
        ----------
        duration_sec : Show for this many seconds (None = entire video).
        start_sec    : Show text starting from this timestamp.
        """
        return self._add(_pl.op_add_text(
            text, x, y, font_size, color, bg_color, duration_sec, start_sec
        ))

    def add_subtitle(
        self,
        text:      str,
        start_sec: float,
        end_sec:   float,
        font_size: int           = 28,
        color:     Tuple         = (255, 255, 255),
        bg_color:  Tuple         = (0, 0, 0, 160),
        position:  str           = "bottom",
    ) -> "Video":
        """Burn a timed subtitle onto frames between *start_sec* and *end_sec*."""
        return self._add(_pl.op_add_subtitle(
            text, start_sec, end_sec, font_size, color, bg_color, position
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  GEOMETRY
    # ═══════════════════════════════════════════════════════════════════

    def resize(
        self,
        width:  int,
        height: int,
        method: str = "bilinear",
    ) -> "Video":
        """
        Resize every frame.

        Parameters
        ----------
        method : ``"bilinear"`` (default, high quality) or ``"nearest"``
                 (fast preview).
        """
        return self._add(_pl.op_resize(width, height, method))

    def crop(self, x: int, y: int, width: int, height: int) -> "Video":
        """Crop every frame to the rectangle starting at (x, y)."""
        return self._add(_pl.op_crop(x, y, width, height))

    def zoom(self, factor: float, smooth: bool = True) -> "Video":
        """
        Zoom (scale) every frame.

        Parameters
        ----------
        factor : 1.0 = no change, 2.0 = double size, 0.5 = half size.
        smooth : Use bilinear interpolation (True) or nearest-neighbour (False).
        """
        return self._add(_pl.op_zoom(factor, smooth))

    def rotate(self, degrees: float, expand: bool = True) -> "Video":
        """
        Rotate every frame clockwise.

        Parameters
        ----------
        degrees : Rotation angle (e.g. 90, 180, 270 or any float).
        expand  : Grow the canvas so no pixels are clipped.
        """
        return self._add(_pl.op_rotate(degrees, expand))

    def flip(
        self,
        horizontal: bool = True,
        vertical:   bool = False,
    ) -> "Video":
        """Mirror every frame horizontally and/or vertically."""
        return self._add(_pl.op_flip(horizontal, vertical))

    # ═══════════════════════════════════════════════════════════════════
    #  COLOR CORRECTION
    # ═══════════════════════════════════════════════════════════════════

    def brightness(self, factor: float) -> "Video":
        """
        Adjust brightness.

        Parameters
        ----------
        factor : 0.0 = black, 1.0 = unchanged, 2.0 = double brightness.
        """
        return self._add(_pl.op_brightness(factor))

    def contrast(self, factor: float) -> "Video":
        """
        Adjust contrast around mid-grey.

        Parameters
        ----------
        factor : 0.0 = flat grey, 1.0 = unchanged, 2.0 = high contrast.
        """
        return self._add(_pl.op_contrast(factor))

    def saturation(self, factor: float) -> "Video":
        """
        Adjust colour saturation.

        Parameters
        ----------
        factor : 0.0 = greyscale, 1.0 = unchanged, 2.0 = vivid.
        """
        return self._add(_pl.op_saturation(factor))

    def greyscale(self) -> "Video":
        """Convert every frame to greyscale."""
        return self._add(_pl.op_greyscale())

    def apply_lut(self, lut: bytes) -> "Video":
        """
        Apply a 768-byte Look-Up Table (3 × 256 per channel R/G/B).
        Build one with ``efv.build_gamma_lut()``.
        """
        return self._add(_pl.op_apply_lut(lut))

    # ═══════════════════════════════════════════════════════════════════
    #  FILTERS
    # ═══════════════════════════════════════════════════════════════════

    def blur(self, radius: int = 2) -> "Video":
        """Apply a Gaussian blur. *radius* is the half-kernel size (1–64)."""
        return self._add(_pl.op_blur(radius))

    def sharpen(self, strength: float = 1.0) -> "Video":
        """Apply an unsharp-mask sharpening filter."""
        return self._add(_pl.op_sharpen(strength))

    def denoise(self, strength: float = 1.0) -> "Video":
        """Apply a lightweight denoising filter."""
        return self._add(_pl.op_denoise(strength))

    def vignette(self, strength: float = 0.5, radius: float = 0.7) -> "Video":
        """
        Add a vignette (dark border) effect.

        Parameters
        ----------
        strength : Darkness of the vignette (0.0–1.0).
        radius   : Normalised radius at which darkening begins (0.0–1.0).
        """
        return self._add(_pl.op_vignette(strength, radius))

    # ═══════════════════════════════════════════════════════════════════
    #  TIMELINE
    # ═══════════════════════════════════════════════════════════════════

    def trim(
        self,
        start_sec: float = 0.0,
        end_sec:   Optional[float] = None,
    ) -> "Video":
        """Keep only frames between *start_sec* and *end_sec*."""
        return self._add(_pl.op_trim(start_sec, end_sec))

    def cut(self, start_sec: float, end_sec: float) -> "Video":
        """Remove frames between *start_sec* and *end_sec*."""
        return self._add(_pl.op_cut(start_sec, end_sec))

    def speed(self, factor: float) -> "Video":
        """
        Change playback speed.

        Parameters
        ----------
        factor : 2.0 = double speed (half the frames kept),
                 0.5 = half speed (frames duplicated).
        """
        return self._add(_pl.op_speed(factor))

    def reverse(self) -> "Video":
        """Reverse the video (play backwards)."""
        return self._add(_pl.op_reverse())

    def loop(self, times: int = 2) -> "Video":
        """Repeat the video *times* times."""
        return self._add(_pl.op_loop(times))

    # ═══════════════════════════════════════════════════════════════════
    #  TRANSITIONS
    # ═══════════════════════════════════════════════════════════════════

    def fade_in(self, duration_sec: float = 1.0) -> "Video":
        """Fade from black at the start."""
        return self._add(_pl.op_fade_in(duration_sec))

    def fade_out(self, duration_sec: float = 1.0) -> "Video":
        """Fade to black at the end."""
        return self._add(_pl.op_fade_out(duration_sec))

    def crossfade(self, other_path: str, duration_sec: float = 1.0) -> "Video":
        """Crossfade (dissolve) into another video at the end."""
        return self._add(_pl.op_crossfade(other_path, duration_sec))

    # ═══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ═══════════════════════════════════════════════════════════════════

    def set_quality(self, quality: int) -> "Video":
        """Set JPEG output quality (1–100). Default: 85."""
        self._quality = max(1, min(100, quality))
        return self

    # ═══════════════════════════════════════════════════════════════════
    #  EXECUTE
    # ═══════════════════════════════════════════════════════════════════

    def save(
        self,
        output_path:   str,
        show_progress: bool = True,
    ) -> "Video":
        """
        Execute the entire pipeline and write to *output_path*.

        This is the only method that performs any actual work.
        All previous calls only enqueue operations.

        Parameters
        ----------
        output_path   : Destination AVI file path.
        show_progress : Print a live progress bar.

        Returns
        -------
        self : allows further chaining if desired.
        """
        from .executor import Executor
        Executor(
            src_path      = self._path,
            ops           = list(self._ops),
            quality       = self._quality,
            show_progress = show_progress,
        ).run(output_path)
        return self

    # ═══════════════════════════════════════════════════════════════════
    #  INTROSPECTION
    # ═══════════════════════════════════════════════════════════════════

    def info(self) -> dict:
        """Return a dict of video metadata without processing."""
        from .backend import core
        from .ops.transform import _pure_avi_meta
        if core is not None:
            reader = core.AviReader(self._path)
            m      = reader.info
            return {
                "path":         self._path,
                "width":        m.video.width,
                "height":       m.video.height,
                "fps":          m.video.fps,
                "frame_count":  m.video.frame_count,
                "duration_sec": m.duration_sec,
                "codec":        m.video.codec,
                "has_audio":    m.audio.has_audio,
                "sample_rate":  m.audio.sample_rate,
                "channels":     m.audio.channels,
            }
        return _pure_avi_meta(self._path)

    def pending_ops(self) -> List[Op]:
        """Return a copy of the pending operation list."""
        return list(self._ops)

    def reset(self) -> "Video":
        """Clear all pending operations."""
        self._ops.clear()
        return self

    # ── Internal ─────────────────────────────────────────────────────────

    def _add(self, op: Op) -> "Video":
        self._ops.append(op)
        return self

    def __repr__(self) -> str:
        return (
            f"<efv.Video path={self._path!r} "
            f"ops={len(self._ops)} quality={self._quality}>"
        )
