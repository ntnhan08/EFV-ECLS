"""
Pure-Python AVI RIFF reader.
Used only when the Rust efv_core extension is not compiled.
Mirrors the public API of efv_core.AviReader.
"""
from __future__ import annotations

import os
import struct
from typing import List, Optional, Tuple


class _Meta:
    class Video:
        width = height = frame_count = 0
        fps   = 25.0
        codec = "MJPG"
    class Audio:
        has_audio   = False
        sample_rate = 44100
        channels    = 2
        bit_depth   = 16
    video        = Video()
    audio        = Audio()
    duration_sec = 0.0


class PureAviReader:
    def __init__(self, path: str):
        self._path = path
        self._meta = _Meta()
        self._frames: List[Tuple[int, int, bool]] = []  # (offset, size, is_video)
        self._audio_chunks: List[Tuple[int, int]] = []
        self._parse()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def info(self):
        return self._meta

    def read_audio(self) -> bytes:
        if not self._audio_chunks:
            return b""
        out = bytearray()
        with open(self._path, "rb") as f:
            for off, sz in self._audio_chunks:
                f.seek(off)
                out += f.read(sz)
        return bytes(out)

    def iter_frames(self) -> List[Tuple[bytes, str]]:
        codec = self._meta.video.codec.strip()
        ftype = "jpeg" if ("MJPG" in codec or "JPEG" in codec) else "rgb24"
        result = []
        with open(self._path, "rb") as f:
            for off, sz, is_video in self._frames:
                if not is_video:
                    continue
                f.seek(off)
                result.append((f.read(sz), ftype))
        return result

    def close(self):
        pass

    def info(self) -> dict:
        m = self._meta
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

    # ── Parser ────────────────────────────────────────────────────────────

    def _parse(self):
        with open(self._path, "rb") as f:
            hdr = f.read(12)
            if hdr[:4] != b"RIFF" or hdr[8:12] != b"AVI ":
                raise ValueError("Not a valid AVI file")
            file_size = os.path.getsize(self._path)
            self._parse_chunks(f, 12, file_size)
        m = self._meta
        if m.video.fps > 0:
            m.duration_sec = m.video.frame_count / m.video.fps

    def _parse_chunks(self, f, start: int, file_size: int):
        pos = start
        m   = self._meta
        while pos + 8 <= file_size:
            f.seek(pos)
            tag = f.read(4)
            if len(tag) < 4:
                break
            sz_raw = f.read(4)
            if len(sz_raw) < 4:
                break
            chunk_size = struct.unpack_from("<I", sz_raw)[0]

            if tag == b"LIST":
                list_type = f.read(4)
                self._parse_chunks(f, pos + 12, min(pos + 8 + chunk_size, file_size))

            elif tag == b"avih":
                d = f.read(min(chunk_size, 64))
                if len(d) >= 40:
                    micro           = struct.unpack_from("<I", d, 0)[0]
                    m.video.fps     = 1_000_000 / max(micro, 1)
                    m.video.frame_count = struct.unpack_from("<I", d, 24)[0]
                    m.video.width   = struct.unpack_from("<I", d, 32)[0]
                    m.video.height  = struct.unpack_from("<I", d, 36)[0]

            elif tag == b"strh":
                d = f.read(min(chunk_size, 64))
                if len(d) >= 8:
                    if d[:4] == b"vids":
                        m.video.codec = d[4:8].decode("ascii", errors="replace").strip()
                    elif d[:4] == b"auds":
                        m.audio.has_audio = True

            elif tag == b"strf":
                d = f.read(min(chunk_size, 32))
                if m.audio.has_audio and len(d) >= 16:
                    m.audio.channels    = struct.unpack_from("<H", d, 2)[0]
                    m.audio.sample_rate = struct.unpack_from("<I", d, 4)[0]
                    if len(d) >= 16:
                        m.audio.bit_depth = struct.unpack_from("<H", d, 14)[0]

            elif tag in (b"00dc", b"00db"):
                self._frames.append((pos + 8, chunk_size, True))

            elif tag == b"01wb":
                self._audio_chunks.append((pos + 8, chunk_size))
                self._frames.append((pos + 8, chunk_size, False))

            pad  = chunk_size % 2
            pos += 8 + chunk_size + pad
