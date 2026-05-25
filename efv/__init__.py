"""
EFV – Extreme Fast Video
========================
Ultra-fast, ultra-light Python video processing library.

* Rust core      – JPEG codec, resize, compositing, color ops, PCM (via maturin / PyO3)
* C++ GPU layer  – CUDA / OpenCL / Metal / CPU-SIMD (via ctypes)
* Pure-Python    – full fallback, zero native dependencies required

Quickstart
----------
>>> import efv
>>> (efv.Video("input.avi")
...     .mute()
...     .add_logo("logo.png", position="topright", opacity=0.8)
...     .resize(1280, 720)
...     .brightness(1.1)
...     .contrast(1.05)
...     .fade_in(0.5)
...     .fade_out(0.5)
...     .save("output.avi"))

Standalone functions
--------------------
>>> efv.mute("in.avi", "out.avi")
>>> efv.add_logo("in.avi", "out.avi", logo_path="logo.png")
>>> efv.resize("in.avi", "out.avi", 1280, 720)
>>> efv.add_text("in.avi", "out.avi", "Hello!", x=20, y=20)
>>> efv.trim("in.avi", "out.avi", start_sec=5, end_sec=15)
>>> efv.speed("in.avi", "out.avi", factor=2.0)
>>> efv.reverse("in.avi", "out.avi")

Backend info
------------
>>> print(efv.backend_name())   # "CUDA" / "OpenCL" / "Metal" / "CPU-Rust" / "CPU-Python"
>>> print(efv.gpu_available())  # True / False
"""
from __future__ import annotations

from .__version__ import __version__, __author__, __license__
from .video      import Video
from .backend    import BACKEND_NAME, GPU_AVAILABLE
from .ops.color  import build_gamma_lut


__all__ = [
    # Core class
    "Video",
    # Standalone one-liner functions
    "mute",
    "add_audio",
    "volume",
    "fade_audio",
    "add_logo",
    "add_watermark",
    "add_text",
    "add_subtitle",
    "resize",
    "crop",
    "zoom",
    "rotate",
    "flip",
    "brightness",
    "contrast",
    "saturation",
    "greyscale",
    "apply_lut",
    "blur",
    "sharpen",
    "denoise",
    "vignette",
    "trim",
    "cut",
    "speed",
    "reverse",
    "loop",
    "fade_in",
    "fade_out",
    "crossfade",
    # LUT helper
    "build_gamma_lut",
    # Backend helpers
    "backend_name",
    "gpu_available",
    "info",
    # Meta
    "__version__",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone convenience functions
#  Each is a thin wrapper: Video(src).op(...).save(dst)
# ═══════════════════════════════════════════════════════════════════════════

def mute(src: str, dst: str, **kw) -> None:
    """Remove all audio.  ``efv.mute("in.avi", "out.avi")``"""
    Video(src, **_quality(kw)).mute().save(dst, **_progress(kw))

def add_audio(src: str, dst: str, audio_path: str,
              volume: float = 1.0, start_sec: float = 0.0, **kw) -> None:
    """Mix in an external audio track."""
    Video(src, **_quality(kw)).add_audio(audio_path, volume, start_sec).save(dst, **_progress(kw))

def volume(src: str, dst: str, factor: float, **kw) -> None:
    """Scale audio volume."""
    Video(src, **_quality(kw)).volume(factor).save(dst, **_progress(kw))

def fade_audio(src: str, dst: str,
               in_sec: float = 0.0, out_sec: float = 0.0, **kw) -> None:
    """Apply audio fade-in / fade-out."""
    Video(src, **_quality(kw)).fade_audio(in_sec, out_sec).save(dst, **_progress(kw))

def add_logo(src: str, dst: str, logo_path: str,
             position: str = "topright", scale: float = 1.0,
             opacity: float = 1.0, margin: int = 10,
             x: int = None, y: int = None, **kw) -> None:
    """Overlay a PNG/JPEG logo on every frame."""
    Video(src, **_quality(kw)) \
        .add_logo(logo_path, position, scale, opacity, margin, x, y) \
        .save(dst, **_progress(kw))

def add_watermark(src: str, dst: str, text: str,
                  opacity: float = 0.3, font_size: int = 36,
                  color: tuple = (255, 255, 255), **kw) -> None:
    """Burn a semi-transparent text watermark."""
    Video(src, **_quality(kw)) \
        .add_watermark(text, opacity, font_size, color) \
        .save(dst, **_progress(kw))

def add_text(src: str, dst: str, text: str,
             x: int = 10, y: int = 10,
             font_size: int = 24,
             color: tuple = (255, 255, 255),
             bg_color=None,
             duration_sec=None, start_sec: float = 0.0, **kw) -> None:
    """Burn text onto every frame (or a timed range)."""
    Video(src, **_quality(kw)) \
        .add_text(text, x, y, font_size, color, bg_color, duration_sec, start_sec) \
        .save(dst, **_progress(kw))

def add_subtitle(src: str, dst: str, text: str,
                 start_sec: float, end_sec: float,
                 font_size: int = 28,
                 color: tuple = (255, 255, 255),
                 bg_color: tuple = (0, 0, 0, 160),
                 position: str = "bottom", **kw) -> None:
    """Burn a timed subtitle."""
    Video(src, **_quality(kw)) \
        .add_subtitle(text, start_sec, end_sec, font_size, color, bg_color, position) \
        .save(dst, **_progress(kw))

def resize(src: str, dst: str, width: int, height: int,
           method: str = "bilinear", **kw) -> None:
    """Resize every frame."""
    Video(src, **_quality(kw)).resize(width, height, method).save(dst, **_progress(kw))

def crop(src: str, dst: str, x: int, y: int,
         width: int, height: int, **kw) -> None:
    """Crop every frame."""
    Video(src, **_quality(kw)).crop(x, y, width, height).save(dst, **_progress(kw))

def zoom(src: str, dst: str, factor: float,
         smooth: bool = True, **kw) -> None:
    """Zoom (scale) every frame."""
    Video(src, **_quality(kw)).zoom(factor, smooth).save(dst, **_progress(kw))

def rotate(src: str, dst: str, degrees: float,
           expand: bool = True, **kw) -> None:
    """Rotate every frame."""
    Video(src, **_quality(kw)).rotate(degrees, expand).save(dst, **_progress(kw))

def flip(src: str, dst: str,
         horizontal: bool = True, vertical: bool = False, **kw) -> None:
    """Mirror every frame."""
    Video(src, **_quality(kw)).flip(horizontal, vertical).save(dst, **_progress(kw))

def brightness(src: str, dst: str, factor: float, **kw) -> None:
    """Adjust brightness."""
    Video(src, **_quality(kw)).brightness(factor).save(dst, **_progress(kw))

def contrast(src: str, dst: str, factor: float, **kw) -> None:
    """Adjust contrast."""
    Video(src, **_quality(kw)).contrast(factor).save(dst, **_progress(kw))

def saturation(src: str, dst: str, factor: float, **kw) -> None:
    """Adjust saturation."""
    Video(src, **_quality(kw)).saturation(factor).save(dst, **_progress(kw))

def greyscale(src: str, dst: str, **kw) -> None:
    """Convert to greyscale."""
    Video(src, **_quality(kw)).greyscale().save(dst, **_progress(kw))

def apply_lut(src: str, dst: str, lut: bytes, **kw) -> None:
    """Apply a 768-byte RGB LUT."""
    Video(src, **_quality(kw)).apply_lut(lut).save(dst, **_progress(kw))

def blur(src: str, dst: str, radius: int = 2, **kw) -> None:
    """Apply Gaussian blur."""
    Video(src, **_quality(kw)).blur(radius).save(dst, **_progress(kw))

def sharpen(src: str, dst: str, strength: float = 1.0, **kw) -> None:
    """Apply unsharp-mask sharpening."""
    Video(src, **_quality(kw)).sharpen(strength).save(dst, **_progress(kw))

def denoise(src: str, dst: str, strength: float = 1.0, **kw) -> None:
    """Apply denoising filter."""
    Video(src, **_quality(kw)).denoise(strength).save(dst, **_progress(kw))

def vignette(src: str, dst: str,
             strength: float = 0.5, radius: float = 0.7, **kw) -> None:
    """Add a vignette effect."""
    Video(src, **_quality(kw)).vignette(strength, radius).save(dst, **_progress(kw))

def trim(src: str, dst: str,
         start_sec: float = 0.0, end_sec: float = None, **kw) -> None:
    """Trim to [start_sec, end_sec]."""
    Video(src, **_quality(kw)).trim(start_sec, end_sec).save(dst, **_progress(kw))

def cut(src: str, dst: str,
        start_sec: float, end_sec: float, **kw) -> None:
    """Remove segment [start_sec, end_sec]."""
    Video(src, **_quality(kw)).cut(start_sec, end_sec).save(dst, **_progress(kw))

def speed(src: str, dst: str, factor: float, **kw) -> None:
    """Change playback speed."""
    Video(src, **_quality(kw)).speed(factor).save(dst, **_progress(kw))

def reverse(src: str, dst: str, **kw) -> None:
    """Reverse video."""
    Video(src, **_quality(kw)).reverse().save(dst, **_progress(kw))

def loop(src: str, dst: str, times: int = 2, **kw) -> None:
    """Loop video N times."""
    Video(src, **_quality(kw)).loop(times).save(dst, **_progress(kw))

def fade_in(src: str, dst: str, duration_sec: float = 1.0, **kw) -> None:
    """Fade from black at the start."""
    Video(src, **_quality(kw)).fade_in(duration_sec).save(dst, **_progress(kw))

def fade_out(src: str, dst: str, duration_sec: float = 1.0, **kw) -> None:
    """Fade to black at the end."""
    Video(src, **_quality(kw)).fade_out(duration_sec).save(dst, **_progress(kw))

def crossfade(src: str, dst: str, other_path: str,
              duration_sec: float = 1.0, **kw) -> None:
    """Crossfade into another video at the end."""
    Video(src, **_quality(kw)).crossfade(other_path, duration_sec).save(dst, **_progress(kw))

# ── Backend helpers ───────────────────────────────────────────────────────

def backend_name() -> str:
    """Return the active compute backend name."""
    return BACKEND_NAME

def gpu_available() -> bool:
    """Return True if a GPU backend (CUDA / OpenCL / Metal) is active."""
    return GPU_AVAILABLE

def info(path: str) -> dict:
    """Return metadata dict for a video file without processing it."""
    return Video(path).info()

# ── Internal helpers ──────────────────────────────────────────────────────

def _quality(kw: dict) -> dict:
    return {"quality": kw.pop("quality", 85)}

def _progress(kw: dict) -> dict:
    return {"show_progress": kw.pop("show_progress", True)}
