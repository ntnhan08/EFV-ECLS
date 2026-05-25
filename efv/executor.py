"""
EFV · Executor
==============
Reads an AVI, applies every Op in the pipeline frame-by-frame,
and writes the result to a new AVI file.

Priority:
  1. Rust core  (efv_core)   – JPEG codec, resize, composite, blur, color, PCM
  2. GPU lib    (efv_gpu)    – same operations but dispatched to CUDA/OpenCL/Metal
  3. Pure Python             – pure-Python fallback (slower, no native deps)
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
from pathlib import Path
from typing import List, Optional

from .backend import core, gpu, GPU_AVAILABLE
from .pipeline import Op, OpKind
from .utils.progress import ProgressBar

# ── Pure-Python fallback imports ──────────────────────────────────────────
from .ops import audio as _audio_py
from .ops import visual as _visual_py
from .ops import transform as _transform_py
from .ops import color as _color_py
from .ops import time_ as _time_py
from .ops import filter_ as _filter_py


# ═══════════════════════════════════════════════════════════════════════════
#  Executor
# ═══════════════════════════════════════════════════════════════════════════

class Executor:
    """
    Applies a list of Ops to a video file and writes the result.

    Parameters
    ----------
    src_path  : input AVI file path
    ops       : ordered list of Op descriptors
    quality   : JPEG quality for output frames (1–100)
    show_progress : display a progress bar
    """

    def __init__(
        self,
        src_path:      str,
        ops:           List[Op],
        quality:       int  = 85,
        show_progress: bool = True,
    ):
        self.src_path      = src_path
        self.ops           = ops
        self.quality       = quality
        self.show_progress = show_progress

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, dst_path: str) -> None:
        """Process video and save to *dst_path*."""
        # ── 1. Open source ───────────────────────────────────────────────
        reader = self._open_reader(self.src_path)
        meta   = reader.info if hasattr(reader, "info") else reader.info()

        # ── 2. Apply timeline ops (cut / trim / speed / reverse / loop) ──
        frame_indices, audio_pcm, fps, sample_rate, channels = \
            self._apply_timeline_ops(reader, meta)

        # ── 3. Open writer ───────────────────────────────────────────────
        writer = self._open_writer(
            dst_path,
            width=meta.video.width,
            height=meta.video.height,
            fps=fps,
            has_audio=meta.audio.has_audio,
            sample_rate=sample_rate,
            channels=channels,
        )

        # ── 4. Pre-load overlay / logo / LUT assets ──────────────────────
        assets = self._preload_assets()

        # ── 5. Per-frame loop ─────────────────────────────────────────────
        n = len(frame_indices)
        bar = ProgressBar(n, "EFV Processing") if self.show_progress else None

        for seq, (raw, ftype) in enumerate(frame_indices):
            frame = self._decode_frame(raw, ftype, meta.video.width, meta.video.height)
            frame = self._apply_frame_ops(frame, seq, n, fps, meta, assets)
            jpeg  = self._encode_frame(frame, meta.video.width, meta.video.height)
            writer.write_video_frame(jpeg)

            if bar:
                bar.update(seq + 1)

        # ── 6. Audio ──────────────────────────────────────────────────────
        if meta.audio.has_audio or self._has_add_audio():
            audio_out = self._apply_audio_ops(
                audio_pcm, fps, n, sample_rate, channels, meta
            )
            if audio_out:
                writer.write_audio_chunk(audio_out)

        writer.close()
        if bar:
            bar.finish()

    # ── Reader / Writer ───────────────────────────────────────────────────

    def _open_reader(self, path: str):
        if core is not None:
            return core.AviReader(path)
        from .py_compat.avi_reader import PureAviReader
        return PureAviReader(path)

    def _open_writer(self, path, width, height, fps,
                     has_audio, sample_rate, channels):
        if core is not None:
            return core.AviWriter(
                path, width, height, fps, has_audio, sample_rate, channels
            )
        from .py_compat.avi_writer import PureAviWriter
        return PureAviWriter(path, width, height, fps, has_audio,
                             sample_rate, channels)

    # ── Timeline ──────────────────────────────────────────────────────────

    def _apply_timeline_ops(self, reader, meta):
        """Build frame_indices list and processed audio PCM."""
        fps         = meta.video.fps
        sample_rate = meta.audio.sample_rate if meta.audio.has_audio else 44100
        channels    = meta.audio.channels    if meta.audio.has_audio else 2

        raw_frames  = reader.iter_frames()          # list of (data, ftype)
        audio_pcm   = bytes(reader.read_audio()) if meta.audio.has_audio else b""

        # Collect timeline ops in order
        for op in self.ops:
            if op.kind == OpKind.TRIM:
                start = op.params["start_sec"]
                end   = op.params.get("end_sec")
                raw_frames, audio_pcm, fps = _time_py.apply_trim(
                    raw_frames, audio_pcm, fps, sample_rate, channels, start, end
                )
            elif op.kind == OpKind.CUT:
                s, e = op.params["start_sec"], op.params["end_sec"]
                raw_frames, audio_pcm = _time_py.apply_cut(
                    raw_frames, audio_pcm, fps, sample_rate, channels, s, e
                )
            elif op.kind == OpKind.SPEED:
                raw_frames, audio_pcm, fps = _time_py.apply_speed(
                    raw_frames, audio_pcm, fps, sample_rate, channels,
                    op.params["factor"]
                )
            elif op.kind == OpKind.REVERSE:
                raw_frames, audio_pcm = _time_py.apply_reverse(
                    raw_frames, audio_pcm
                )
            elif op.kind == OpKind.LOOP:
                raw_frames, audio_pcm = _time_py.apply_loop(
                    raw_frames, audio_pcm, op.params["times"]
                )

        return raw_frames, audio_pcm, fps, sample_rate, channels

    # ── Frame decode / encode ─────────────────────────────────────────────

    def _decode_frame(self, raw: bytes, ftype: str, w: int, h: int) -> dict:
        """Return a dict with keys: rgba (bytearray), width, height."""
        if core is not None:
            if ftype == "jpeg":
                rgb, fw, fh = core.py_decode_jpeg(raw)
            elif ftype == "png":
                rgba, fw, fh = core.py_decode_png(raw)
                return dict(rgba=bytearray(rgba), width=fw, height=fh)
            else:                                  # raw BGR bottom-up
                rgb, fw, fh = _transform_py.bgr_to_rgb(raw, w, h), w, h
            # RGB → RGBA
            n    = fw * fh
            rgba = bytearray(n * 4)
            for i in range(n):
                rgba[i*4]     = rgb[i*3]
                rgba[i*4 + 1] = rgb[i*3 + 1]
                rgba[i*4 + 2] = rgb[i*3 + 2]
                rgba[i*4 + 3] = 255
            return dict(rgba=rgba, width=fw, height=fh)
        else:
            return _transform_py.decode_frame_py(raw, ftype, w, h)

    def _encode_frame(self, frame: dict, w: int, h: int) -> bytes:
        rgba  = frame["rgba"]
        fw, fh = frame.get("width", w), frame.get("height", h)
        # RGBA → RGB
        n   = fw * fh
        rgb = bytearray(n * 3)
        for i in range(n):
            rgb[i*3]     = rgba[i*4]
            rgb[i*3 + 1] = rgba[i*4 + 1]
            rgb[i*3 + 2] = rgba[i*4 + 2]
        if core is not None:
            return bytes(core.py_encode_jpeg(bytes(rgb), fw, fh, self.quality))
        return _color_py.encode_jpeg_py(bytes(rgb), fw, fh, self.quality)

    # ── Per-frame ops ─────────────────────────────────────────────────────

    def _apply_frame_ops(
        self, frame: dict, seq: int, total: int, fps: float, meta, assets: dict
    ) -> dict:
        t_sec = seq / fps if fps > 0 else 0.0

        for op in self.ops:
            k = op.kind
            p = op.params

            # ── Geometry ──────────────────────────────────────────────
            if k == OpKind.RESIZE:
                frame = self._resize(frame, p["width"], p["height"], p.get("method","bilinear"))

            elif k == OpKind.CROP:
                frame = _transform_py.apply_crop(frame, p["x"], p["y"], p["width"], p["height"])

            elif k == OpKind.ZOOM:
                factor = p["factor"]
                w, h   = frame["width"], frame["height"]
                nw     = max(2, int(w * factor))
                nh     = max(2, int(h * factor))
                nw    += nw % 2; nh += nh % 2
                frame  = self._resize(frame, nw, nh, "bilinear" if p.get("smooth", True) else "nearest")

            elif k == OpKind.ROTATE:
                frame = _transform_py.apply_rotate(frame, p["degrees"], p.get("expand", True))

            elif k == OpKind.FLIP:
                frame = _transform_py.apply_flip(frame, p.get("horizontal", True), p.get("vertical", False))

            # ── Color ─────────────────────────────────────────────────
            elif k == OpKind.BRIGHTNESS:
                frame = self._brightness(frame, p["factor"])

            elif k == OpKind.CONTRAST:
                frame = self._contrast(frame, p["factor"])

            elif k == OpKind.SATURATION:
                frame = self._saturation(frame, p["factor"])

            elif k == OpKind.GREYSCALE:
                frame = _color_py.apply_greyscale(frame)

            elif k == OpKind.APPLY_LUT:
                frame = _color_py.apply_lut(frame, p["lut"])

            # ── Filters ───────────────────────────────────────────────
            elif k == OpKind.BLUR:
                frame = self._blur(frame, p.get("radius", 2))

            elif k == OpKind.SHARPEN:
                frame = _filter_py.apply_sharpen(frame, p.get("strength", 1.0))

            elif k == OpKind.DENOISE:
                frame = _filter_py.apply_denoise(frame, p.get("strength", 1.0))

            elif k == OpKind.VIGNETTE:
                frame = _filter_py.apply_vignette(frame, p.get("strength", 0.5), p.get("radius", 0.7))

            # ── Overlays ──────────────────────────────────────────────
            elif k == OpKind.ADD_LOGO:
                key  = ("logo", p["path"])
                logo = assets.get(key)
                if logo is not None:
                    frame = _visual_py.apply_logo(frame, logo, p)

            elif k == OpKind.ADD_WATERMARK:
                frame = _visual_py.apply_watermark(frame, p)

            elif k == OpKind.ADD_TEXT:
                # Respect duration/start constraints
                start = p.get("start_sec", 0.0)
                dur   = p.get("duration_sec")
                if t_sec >= start and (dur is None or t_sec < start + dur):
                    frame = _visual_py.apply_text(frame, p)

            elif k == OpKind.ADD_SUBTITLE:
                if p["start_sec"] <= t_sec < p["end_sec"]:
                    frame = _visual_py.apply_subtitle(frame, p)

            # ── Transitions ───────────────────────────────────────────
            elif k == OpKind.FADE_IN:
                dur = p["duration_sec"]
                if t_sec < dur:
                    alpha = t_sec / dur
                    frame = _color_py.apply_brightness(frame, alpha)

            elif k == OpKind.FADE_OUT:
                dur      = p["duration_sec"]
                end_t    = total / fps if fps > 0 else 0.0
                remaining = end_t - t_sec
                if remaining < dur:
                    alpha = max(0.0, remaining / dur)
                    frame = _color_py.apply_brightness(frame, alpha)

        return frame

    # ── Audio ops ─────────────────────────────────────────────────────────

    def _apply_audio_ops(
        self, pcm: bytes, fps: float, n_frames: int,
        sample_rate: int, channels: int, meta
    ) -> Optional[bytes]:
        for op in self.ops:
            k, p = op.kind, op.params
            if k == OpKind.MUTE:
                pcm = _audio_py.apply_mute(pcm, n_frames, fps, sample_rate, channels)
            elif k == OpKind.CHANGE_VOLUME:
                pcm = _audio_py.apply_volume(pcm, p["factor"])
            elif k == OpKind.ADD_AUDIO:
                pcm = _audio_py.apply_add_audio(
                    pcm, p["path"], sample_rate, channels,
                    p.get("volume", 1.0), p.get("start_sec", 0.0),
                    n_frames / fps if fps > 0 else 0.0
                )
            elif k == OpKind.FADE_AUDIO:
                dur = n_frames / fps if fps > 0 else 0.0
                pcm = _audio_py.apply_fade_audio(
                    pcm, sample_rate, channels, dur,
                    p.get("in_sec", 0.0), p.get("out_sec", 0.0)
                )
        target_bytes = int(n_frames / fps * sample_rate * channels * 2) if fps > 0 else 0
        return _audio_py.pad_or_trim(pcm, target_bytes)

    # ── Preload assets (logos, LUTs …) ────────────────────────────────────

    def _preload_assets(self) -> dict:
        assets = {}
        for op in self.ops:
            if op.kind == OpKind.ADD_LOGO:
                path = op.params["path"]
                key  = ("logo", path)
                if key not in assets:
                    assets[key] = _visual_py.load_image(path)
        return assets

    def _has_add_audio(self) -> bool:
        return any(op.kind == OpKind.ADD_AUDIO for op in self.ops)

    # ── Backend-dispatched operations ─────────────────────────────────────

    def _resize(self, frame: dict, w: int, h: int, method: str) -> dict:
        rgba = frame["rgba"]
        fw, fh = frame["width"], frame["height"]
        if w == fw and h == fh:
            return frame
        if core is not None:
            src_f = core.Frame.from_rgba(bytes(rgba), fw, fh)
            if method == "nearest":
                dst_f = core.py_resize_nearest(src_f, w, h)
            else:
                dst_f = core.py_resize_bilinear(src_f, w, h)
            return dict(rgba=bytearray(dst_f.to_rgba()), width=w, height=h)
        if GPU_AVAILABLE and gpu is not None:
            new_rgba = gpu.resize_bilinear(bytes(rgba), fw, fh, w, h) \
                       if method != "nearest" else \
                       gpu.resize_nearest(bytes(rgba), fw, fh, w, h)
            return dict(rgba=bytearray(new_rgba), width=w, height=h)
        return _transform_py.resize_bilinear_py(frame, w, h)

    def _brightness(self, frame: dict, factor: float) -> dict:
        if core is not None:
            f  = core.Frame.from_rgba(bytes(frame["rgba"]), frame["width"], frame["height"])
            f2 = core.py_adjust_brightness(f, factor)
            return dict(rgba=bytearray(f2.to_rgba()), width=frame["width"], height=frame["height"])
        if GPU_AVAILABLE and gpu is not None:
            buf = bytearray(frame["rgba"])
            gpu.brightness(buf, frame["width"], frame["height"], factor)
            return dict(rgba=buf, width=frame["width"], height=frame["height"])
        return _color_py.apply_brightness(frame, factor)

    def _contrast(self, frame: dict, factor: float) -> dict:
        if core is not None:
            f  = core.Frame.from_rgba(bytes(frame["rgba"]), frame["width"], frame["height"])
            f2 = core.py_adjust_contrast(f, factor)
            return dict(rgba=bytearray(f2.to_rgba()), width=frame["width"], height=frame["height"])
        if GPU_AVAILABLE and gpu is not None:
            buf = bytearray(frame["rgba"])
            gpu.contrast(buf, frame["width"], frame["height"], factor)
            return dict(rgba=buf, width=frame["width"], height=frame["height"])
        return _color_py.apply_contrast(frame, factor)

    def _saturation(self, frame: dict, factor: float) -> dict:
        if core is not None:
            f  = core.Frame.from_rgba(bytes(frame["rgba"]), frame["width"], frame["height"])
            f2 = core.py_adjust_saturation(f, factor)
            return dict(rgba=bytearray(f2.to_rgba()), width=frame["width"], height=frame["height"])
        if GPU_AVAILABLE and gpu is not None:
            buf = bytearray(frame["rgba"])
            gpu.saturation(buf, frame["width"], frame["height"], factor)
            return dict(rgba=buf, width=frame["width"], height=frame["height"])
        return _color_py.apply_saturation(frame, factor)

    def _blur(self, frame: dict, radius: int) -> dict:
        if core is not None:
            f  = core.Frame.from_rgba(bytes(frame["rgba"]), frame["width"], frame["height"])
            f2 = core.py_gaussian_blur(f, radius)
            return dict(rgba=bytearray(f2.to_rgba()), width=frame["width"], height=frame["height"])
        return _filter_py.apply_blur(frame, radius)
