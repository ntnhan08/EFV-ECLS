"""
Pure-Python MJPEG-AVI writer.
Used only when the Rust efv_core extension is not compiled.
"""
from __future__ import annotations

import struct


class PureAviWriter:
    def __init__(self, path: str, width: int, height: int, fps: float,
                 has_audio: bool = False, sample_rate: int = 44100,
                 channels: int = 2):
        self._path   = path
        self._width  = width
        self._height = height
        self._fps    = fps
        self._has_audio   = has_audio
        self._sample_rate = sample_rate
        self._channels    = channels

        self._f      = open(path, "wb")
        self._frames = 0
        self._index  = []        # list of (fourcc, flags, offset, size)

        self._write_headers()
        self._movi_start = self._f.tell()
        self._f.write(b"LIST")
        self._f.write(b"\x00\x00\x00\x00")   # size placeholder
        self._f.write(b"movi")

    # ── Public API ────────────────────────────────────────────────────────

    def write_video_frame(self, data: bytes):
        off  = self._f.tell()
        size = len(data)
        self._f.write(b"00dc")
        self._f.write(struct.pack("<I", size))
        self._f.write(data)
        if size % 2:
            self._f.write(b"\x00")
        self._index.append((b"00dc", 0x10, off, size))
        self._frames += 1

    def write_audio_chunk(self, data: bytes):
        off  = self._f.tell()
        size = len(data)
        self._f.write(b"01wb")
        self._f.write(struct.pack("<I", size))
        self._f.write(data)
        if size % 2:
            self._f.write(b"\x00")
        self._index.append((b"01wb", 0, off, size))

    def close(self):
        if self._f.closed:
            return

        movi_end      = self._f.tell()
        movi_data_sz  = movi_end - self._movi_start - 8

        # Patch movi LIST size
        self._f.seek(self._movi_start + 4)
        self._f.write(struct.pack("<I", movi_data_sz))

        # Write idx1
        self._f.seek(movi_end)
        idx_data = b"".join(
            e[0] + struct.pack("<III", e[1], e[2], e[3])
            for e in self._index
        )
        self._f.write(b"idx1")
        self._f.write(struct.pack("<I", len(idx_data)))
        self._f.write(idx_data)

        # Patch RIFF size
        total = self._f.tell()
        self._f.seek(4)
        self._f.write(struct.pack("<I", total - 8))

        # Patch avih frame count (offset 32 from file start)
        self._f.seek(32)
        self._f.write(struct.pack("<I", self._frames))

        self._f.flush()
        self._f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Header writing ────────────────────────────────────────────────────

    def _write_headers(self):
        f    = self._f
        w, h = self._width, self._height
        fps  = max(1, self._fps)
        micro = int(1_000_000 / fps)
        sr   = self._sample_rate
        ch   = self._channels
        bd   = 16
        bpf  = ch * (bd // 8)
        bps  = sr * bpf

        n_streams = 2 if self._has_audio else 1

        # RIFF AVI
        f.write(b"RIFF"); f.write(b"\x00\x00\x00\x00"); f.write(b"AVI ")

        # LIST hdrl
        hdrl_off = f.tell()
        f.write(b"LIST"); f.write(b"\x00\x00\x00\x00"); f.write(b"hdrl")

        # avih (56 bytes)
        f.write(b"avih"); f.write(struct.pack("<I", 56))
        f.write(struct.pack("<I", micro))          # micro_per_frame
        f.write(struct.pack("<I", 0))              # max_bytes_per_sec
        f.write(struct.pack("<I", 0))              # padding_granularity
        f.write(struct.pack("<I", 0x10))           # flags AVIF_HASINDEX
        f.write(struct.pack("<I", 0))              # total_frames ← patched in close()
        f.write(struct.pack("<I", 0))              # initial_frames
        f.write(struct.pack("<I", n_streams))
        f.write(struct.pack("<I", 0))              # suggested_buffer_size
        f.write(struct.pack("<I", w))
        f.write(struct.pack("<I", h))
        f.write(b"\x00" * 16)

        # Video strl
        self._write_video_strl(w, h, fps)
        if self._has_audio:
            self._write_audio_strl(sr, ch, bd, bpf, bps)

        # Patch hdrl size
        hdrl_end = f.tell()
        f.seek(hdrl_off + 4)
        f.write(struct.pack("<I", hdrl_end - hdrl_off - 8))
        f.seek(hdrl_end)

    def _write_video_strl(self, w, h, fps):
        f = self._f
        strl_off = f.tell()
        f.write(b"LIST"); f.write(b"\x00\x00\x00\x00"); f.write(b"strl")

        # strh video
        f.write(b"strh"); f.write(struct.pack("<I", 56))
        f.write(b"vids"); f.write(b"MJPG")         # type + handler
        f.write(b"\x00" * 8)                       # flags…initial_frames
        f.write(struct.pack("<I", 1))               # scale
        f.write(struct.pack("<I", int(fps)))        # rate
        f.write(b"\x00" * 16)                      # start..quality
        f.write(struct.pack("<I", 0))               # sample_size
        f.write(b"\x00" * 8)                       # rcFrame

        # strf BITMAPINFOHEADER
        f.write(b"strf"); f.write(struct.pack("<I", 40))
        f.write(struct.pack("<I", 40))              # biSize
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", h))
        f.write(struct.pack("<HH", 1, 24))          # biPlanes, biBitCount
        f.write(b"MJPG")                            # biCompression
        f.write(struct.pack("<I", w * h * 3))       # biSizeImage
        f.write(b"\x00" * 16)

        end = f.tell()
        f.seek(strl_off + 4); f.write(struct.pack("<I", end - strl_off - 8))
        f.seek(end)

    def _write_audio_strl(self, sr, ch, bd, bpf, bps):
        f = self._f
        strl_off = f.tell()
        f.write(b"LIST"); f.write(b"\x00\x00\x00\x00"); f.write(b"strl")

        # strh audio
        f.write(b"strh"); f.write(struct.pack("<I", 56))
        f.write(b"auds"); f.write(b"\x00" * 4)     # no handler for PCM
        f.write(b"\x00" * 8)
        f.write(struct.pack("<I", bpf))             # scale
        f.write(struct.pack("<I", bps))             # rate
        f.write(b"\x00" * 20)
        f.write(struct.pack("<I", bpf))             # sample_size
        f.write(b"\x00" * 8)

        # strf WAVEFORMATEX
        f.write(b"strf"); f.write(struct.pack("<I", 18))
        f.write(struct.pack("<H", 1))               # PCM
        f.write(struct.pack("<H", ch))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", bps))
        f.write(struct.pack("<H", bpf))
        f.write(struct.pack("<H", bd))
        f.write(struct.pack("<H", 0))               # cbSize

        end = f.tell()
        f.seek(strl_off + 4); f.write(struct.pack("<I", end - strl_off - 8))
        f.seek(end)
