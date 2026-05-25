"""EFV · Timeline Ops – trim, cut, speed, reverse, loop (pure-Python)."""
from __future__ import annotations

import struct
from typing import List, Tuple


# Type alias
FrameList = List[Tuple[bytes, str]]


def apply_trim(
    frames: FrameList, audio: bytes,
    fps: float, sr: int, ch: int,
    start_sec: float, end_sec: float | None,
) -> tuple:
    """Keep frames in [start_sec, end_sec]."""
    s = max(0, int(start_sec * fps))
    e = len(frames) if end_sec is None else min(len(frames), int(end_sec * fps))
    out_frames = frames[s:e]
    out_audio  = _trim_pcm(audio, sr, ch, start_sec, end_sec)
    return out_frames, out_audio, fps


def apply_cut(
    frames: FrameList, audio: bytes,
    fps: float, sr: int, ch: int,
    start_sec: float, end_sec: float,
) -> tuple:
    """Remove frames in [start_sec, end_sec]."""
    s = max(0, int(start_sec * fps))
    e = min(len(frames), int(end_sec * fps))
    out_frames = frames[:s] + frames[e:]
    # Splice audio
    frame_bytes = ch * 2
    sb = s * frame_bytes * sr // max(1, int(fps))
    eb = e * frame_bytes * sr // max(1, int(fps))
    out_audio = audio[:sb] + audio[eb:]
    return out_frames, out_audio


def apply_speed(
    frames: FrameList, audio: bytes,
    fps: float, sr: int, ch: int,
    factor: float,
) -> tuple:
    """Change playback speed by *factor* (> 1 = faster, < 1 = slower)."""
    factor = max(0.1, factor)
    new_fps = fps * factor

    if factor > 1.0:
        # Drop frames
        step = factor
        out_frames = [frames[int(i)] for i in _frange(0, len(frames), step)]
    else:
        # Duplicate frames
        out_frames = []
        for f in frames:
            reps = max(1, round(1.0 / factor))
            out_frames.extend([f] * reps)

    out_audio = _resample_pcm_speed(audio, sr, ch, factor)
    return out_frames, out_audio, new_fps


def apply_reverse(frames: FrameList, audio: bytes) -> tuple:
    """Reverse both video and audio."""
    out_frames = list(reversed(frames))
    out_audio  = _reverse_pcm(audio)
    return out_frames, out_audio


def apply_loop(frames: FrameList, audio: bytes, times: int) -> tuple:
    """Repeat video and audio *times* times."""
    times      = max(1, times)
    out_frames = frames * times
    out_audio  = audio  * times
    return out_frames, out_audio


# ── PCM helpers ───────────────────────────────────────────────────────────

def _trim_pcm(pcm: bytes, sr: int, ch: int,
              start_sec: float, end_sec: float | None) -> bytes:
    frame_bytes = ch * 2
    start_b = int(start_sec * sr) * frame_bytes
    end_b   = len(pcm) if end_sec is None else int(end_sec * sr) * frame_bytes
    return pcm[start_b : min(end_b, len(pcm))]


def _reverse_pcm(pcm: bytes) -> bytes:
    """Reverse 16-bit PCM sample by sample."""
    n   = len(pcm) // 2
    out = bytearray(len(pcm))
    for i in range(n):
        src_i = (n - 1 - i) * 2
        out[i*2]   = pcm[src_i]
        out[i*2+1] = pcm[src_i+1]
    return bytes(out)


def _resample_pcm_speed(pcm: bytes, sr: int, ch: int, factor: float) -> bytes:
    """Very fast linear resampling for speed changes."""
    frame_bytes = ch * 2
    n_src = len(pcm) // frame_bytes
    n_dst = max(1, int(n_src / factor))
    out   = bytearray(n_dst * frame_bytes)

    for i in range(n_dst):
        src_f = i * factor
        i0    = min(int(src_f), n_src - 1)
        i1    = min(i0 + 1, n_src - 1)
        t     = src_f - i0
        for c in range(ch):
            s0 = struct.unpack_from("<h", pcm, (i0 * ch + c) * 2)[0]
            s1 = struct.unpack_from("<h", pcm, (i1 * ch + c) * 2)[0]
            v  = int(s0 + (s1 - s0) * t)
            v  = max(-32768, min(32767, v))
            struct.pack_into("<h", out, (i * ch + c) * 2, v)

    return bytes(out)


def _frange(start: float, stop: float, step: float):
    v = start
    while v < stop:
        yield v
        v += step
