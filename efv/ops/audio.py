"""EFV · Audio Ops – pure-Python fallback implementations."""
from __future__ import annotations

import struct
import wave
import io
from typing import Optional


def apply_mute(pcm: bytes, n_frames: int, fps: float,
               sample_rate: int, channels: int) -> bytes:
    """Return silence matching the video duration."""
    n_samples = int(n_frames / max(fps, 1) * sample_rate) * channels
    return b"\x00" * (n_samples * 2)


def apply_volume(pcm: bytes, factor: float) -> bytes:
    """Scale 16-bit PCM amplitude in-place."""
    fi  = int(factor * 256)
    n   = len(pcm) // 2
    out = bytearray(len(pcm))
    for i in range(n):
        s = struct.unpack_from("<h", pcm, i * 2)[0]
        v = max(-32768, min(32767, (s * fi) >> 8))
        struct.pack_into("<h", out, i * 2, v)
    return bytes(out)


def apply_add_audio(
    existing_pcm: bytes,
    audio_path:   str,
    sample_rate:  int,
    channels:     int,
    volume:       float,
    start_sec:    float,
    duration_sec: float,
) -> bytes:
    """Read WAV from *audio_path* and mix it with *existing_pcm*."""
    try:
        new_pcm = _load_wav_as_pcm(audio_path, sample_rate, channels)
    except Exception as exc:
        import warnings
        warnings.warn(f"[EFV] add_audio: cannot load {audio_path!r}: {exc}")
        return existing_pcm

    # Trim to start_sec
    start_bytes = int(start_sec * sample_rate * channels * 2)
    new_pcm = new_pcm[start_bytes:]

    # Pad / trim new_pcm to match duration
    target = int(duration_sec * sample_rate * channels * 2)
    if len(new_pcm) < target:
        new_pcm = new_pcm + b"\x00" * (target - len(new_pcm))
    new_pcm = new_pcm[:target]

    if volume != 1.0:
        new_pcm = apply_volume(new_pcm, volume)

    return _mix(existing_pcm, new_pcm)


def apply_fade_audio(
    pcm: bytes, sample_rate: int, channels: int,
    duration_sec: float, in_sec: float, out_sec: float
) -> bytes:
    frame_bytes = channels * 2
    total       = len(pcm) // frame_bytes
    out         = bytearray(pcm)

    fade_in_frames  = int(in_sec  * sample_rate)
    fade_out_frames = int(out_sec * sample_rate)

    for i in range(min(fade_in_frames, total)):
        alpha = i / max(fade_in_frames, 1)
        fi    = int(alpha * 256)
        for ch in range(channels):
            off = (i * channels + ch) * 2
            s   = struct.unpack_from("<h", out, off)[0]
            struct.pack_into("<h", out, off, max(-32768, min(32767, (s * fi) >> 8)))

    for i in range(min(fade_out_frames, total)):
        frame_idx = total - 1 - i
        alpha     = i / max(fade_out_frames, 1)
        fi        = int(alpha * 256)
        for ch in range(channels):
            off = (frame_idx * channels + ch) * 2
            s   = struct.unpack_from("<h", out, off)[0]
            struct.pack_into("<h", out, off, max(-32768, min(32767, (s * fi) >> 8)))

    return bytes(out)


def pad_or_trim(pcm: bytes, target_bytes: int) -> Optional[bytes]:
    if not pcm and target_bytes <= 0:
        return None
    if not pcm:
        return b"\x00" * target_bytes
    if len(pcm) >= target_bytes:
        return pcm[:target_bytes]
    return pcm + b"\x00" * (target_bytes - len(pcm))


# ── Helpers ───────────────────────────────────────────────────────────────

def _mix(a: bytes, b: bytes) -> bytes:
    na = len(a) // 2
    nb = len(b) // 2
    n  = max(na, nb)
    out = bytearray(n * 2)
    for i in range(n):
        sa = struct.unpack_from("<h", a, i * 2)[0] if i < na else 0
        sb = struct.unpack_from("<h", b, i * 2)[0] if i < nb else 0
        v  = max(-32768, min(32767, sa + sb))
        struct.pack_into("<h", out, i * 2, v)
    return bytes(out)


def _load_wav_as_pcm(path: str, target_sr: int, target_ch: int) -> bytes:
    """Load a WAV file and resample/remix to target_sr / target_ch (basic)."""
    with wave.open(path, "rb") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        bps = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # Convert to 16-bit
    if bps == 1:
        raw = bytes(((b - 128) * 256) & 0xFFFF for b in raw)
        raw = struct.pack(f"<{len(raw) // 2}h", *[
            struct.unpack_from("<h", raw, i * 2)[0] for i in range(len(raw) // 2)
        ])
    elif bps == 3:
        n   = len(raw) // 3
        raw = struct.pack(f"<{n}h", *[
            struct.unpack_from("<i", raw + b"\x00", i * 3)[0] >> 8 for i in range(n)
        ])
    elif bps == 4:
        n   = len(raw) // 4
        raw = struct.pack(f"<{n}h", *[
            struct.unpack_from("<i", raw, i * 4)[0] >> 16 for i in range(n)
        ])

    # Basic channel conversion (stereo ↔ mono only)
    if ch != target_ch:
        if ch == 1 and target_ch == 2:
            n   = len(raw) // 2
            out = bytearray(n * 4)
            for i in range(n):
                s = raw[i*2 : i*2+2]
                out[i*4 : i*4+2] = s
                out[i*4+2 : i*4+4] = s
            raw = bytes(out)
        elif ch == 2 and target_ch == 1:
            n   = len(raw) // 4
            out = bytearray(n * 2)
            for i in range(n):
                l = struct.unpack_from("<h", raw, i * 4)[0]
                r = struct.unpack_from("<h", raw, i * 4 + 2)[0]
                struct.pack_into("<h", out, i * 2, (l + r) // 2)
            raw = bytes(out)

    # Basic linear resampling
    if sr != target_sr:
        raw = _resample_pcm(raw, sr, target_sr, target_ch)

    return raw


def _resample_pcm(pcm: bytes, src_sr: int, dst_sr: int, ch: int) -> bytes:
    """Naive linear resampling."""
    ratio    = src_sr / dst_sr
    n_src    = len(pcm) // (ch * 2)
    n_dst    = int(n_src / ratio)
    out      = bytearray(n_dst * ch * 2)

    for i in range(n_dst):
        src_f = i * ratio
        i0    = min(int(src_f), n_src - 1)
        i1    = min(i0 + 1, n_src - 1)
        t     = src_f - i0
        for c in range(ch):
            s0 = struct.unpack_from("<h", pcm, (i0 * ch + c) * 2)[0]
            s1 = struct.unpack_from("<h", pcm, (i1 * ch + c) * 2)[0]
            v  = int(s0 + (s1 - s0) * t)
            struct.pack_into("<h", out, (i * ch + c) * 2, max(-32768, min(32767, v)))

    return bytes(out)
