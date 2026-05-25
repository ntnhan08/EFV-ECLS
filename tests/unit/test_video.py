"""
EFV · Unit Tests – Video class and ops
======================================
Run with: pytest tests/ -v
"""
import os
import struct
import tempfile
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────

def make_rgb_frame(w: int, h: int, r=128, g=64, b=32) -> bytes:
    return bytes([r, g, b] * w * h)


def make_rgba_frame(w: int, h: int, r=100, g=150, b=200, a=255) -> bytes:
    return bytes([r, g, b, a] * w * h)


def make_silent_pcm(n_samples: int, channels: int = 2) -> bytes:
    return b"\x00" * n_samples * channels * 2


def make_tone_pcm(n_samples: int, channels: int = 2,
                  amplitude: int = 16000) -> bytes:
    import math
    out = bytearray(n_samples * channels * 2)
    for i in range(n_samples):
        v = int(amplitude * math.sin(2 * math.pi * 440 * i / 44100))
        v = max(-32768, min(32767, v))
        for ch in range(channels):
            struct.pack_into("<h", out, (i * channels + ch) * 2, v)
    return bytes(out)


# ── Backend availability ───────────────────────────────────────────────────

class TestBackend:
    def test_import(self):
        import efv
        assert hasattr(efv, "backend_name")
        assert isinstance(efv.backend_name(), str)
        assert isinstance(efv.gpu_available(), bool)

    def test_backend_name_non_empty(self):
        import efv
        assert len(efv.backend_name()) > 0

    def test_version(self):
        import efv
        assert efv.__version__ == "1.0.0"


# ── Frame (Rust core) ─────────────────────────────────────────────────────

class TestRustFrame:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def test_frame_create(self):
        import efv_core as c
        f = c.Frame(64, 48)
        assert f.width  == 64
        assert f.height == 48

    def test_frame_from_rgb(self):
        import efv_core as c
        rgb = make_rgb_frame(10, 10)
        f   = c.Frame.from_rgb(rgb, 10, 10)
        assert f.width  == 10
        assert f.height == 10
        px = f.get_pixel(0, 0)
        assert px == (128, 64, 32, 255)

    def test_frame_from_rgba(self):
        import efv_core as c
        rgba = make_rgba_frame(8, 8)
        f    = c.Frame.from_rgba(rgba, 8, 8)
        px   = f.get_pixel(0, 0)
        assert px == (100, 150, 200, 255)

    def test_frame_set_get_pixel(self):
        import efv_core as c
        f = c.Frame(4, 4)
        f.set_pixel(2, 1, 10, 20, 30, 200)
        assert f.get_pixel(2, 1) == (10, 20, 30, 200)

    def test_frame_to_rgb_roundtrip(self):
        import efv_core as c
        rgb = make_rgb_frame(16, 16, 200, 100, 50)
        f   = c.Frame.from_rgb(rgb, 16, 16)
        out = bytes(f.to_rgb())
        assert out == rgb

    def test_frame_fill_rect(self):
        import efv_core as c
        f = c.Frame(10, 10, 0, 0, 0, 255)
        f.fill_rect(2, 2, 5, 5, 255, 0, 0, 255)
        assert f.get_pixel(3, 3) == (255, 0, 0, 255)
        assert f.get_pixel(0, 0) == (0, 0, 0, 255)

    def test_frame_copy(self):
        import efv_core as c
        f  = c.Frame.from_rgb(make_rgb_frame(8, 8), 8, 8)
        f2 = f.copy()
        f2.set_pixel(0, 0, 1, 2, 3, 4)
        # Original unchanged
        assert f.get_pixel(0, 0) == (128, 64, 32, 255)


# ── JPEG codec ────────────────────────────────────────────────────────────

class TestJpegCodec:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def test_encode_decode_roundtrip(self):
        import efv_core as c
        rgb = make_rgb_frame(64, 64, 180, 90, 45)
        enc = c.py_encode_jpeg(rgb, 64, 64, 95)
        assert isinstance(enc, bytes)
        assert len(enc) > 0
        # JPEG magic bytes
        assert enc[:2] == b"\xff\xd8"

        dec, w, h = c.py_decode_jpeg(enc)
        assert w == 64
        assert h == 64
        assert len(dec) == 64 * 64 * 3

    def test_encode_quality_range(self):
        import efv_core as c
        rgb    = make_rgb_frame(32, 32)
        enc_lo = c.py_encode_jpeg(rgb, 32, 32, 1)
        enc_hi = c.py_encode_jpeg(rgb, 32, 32, 100)
        # Higher quality = larger file
        assert len(enc_hi) > len(enc_lo)

    def test_decode_returns_correct_size(self):
        import efv_core as c
        rgb = make_rgb_frame(100, 80)
        enc = c.py_encode_jpeg(rgb, 100, 80, 85)
        dec, w, h = c.py_decode_jpeg(enc)
        assert w == 100 and h == 80
        assert len(dec) == 100 * 80 * 3


# ── Resize ────────────────────────────────────────────────────────────────

class TestResize:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def _frame(self, w=64, h=48):
        import efv_core as c
        return c.Frame.from_rgb(make_rgb_frame(w, h), w, h)

    def test_bilinear_dimensions(self):
        import efv_core as c
        f  = self._frame(64, 48)
        f2 = c.py_resize_bilinear(f, 128, 96)
        assert f2.width == 128 and f2.height == 96

    def test_nearest_dimensions(self):
        import efv_core as c
        f  = self._frame(64, 48)
        f2 = c.py_resize_nearest(f, 32, 24)
        assert f2.width == 32 and f2.height == 24

    def test_zoom_factor(self):
        import efv_core as c
        f  = self._frame(100, 100)
        f2 = c.py_zoom_frame(f, 2.0, True)
        assert f2.width == 200 and f2.height == 200

    def test_downscale_preserves_colour(self):
        import efv_core as c
        # Solid colour frame – downscale should keep same colour
        f  = c.Frame(100, 100, 200, 100, 50, 255)
        f2 = c.py_resize_bilinear(f, 10, 10)
        px = f2.get_pixel(5, 5)
        for i, expected in enumerate([200, 100, 50, 255]):
            assert abs(px[i] - expected) <= 2, f"channel {i}: {px[i]} != {expected}"


# ── Color ops ─────────────────────────────────────────────────────────────

class TestColorOps:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def _grey_frame(self, v=128):
        import efv_core as c
        return c.Frame(32, 32, v, v, v, 255)

    def test_brightness_increase(self):
        import efv_core as c
        f  = self._grey_frame(100)
        f2 = c.py_adjust_brightness(f, 2.0)
        px = f2.get_pixel(0, 0)
        assert px[0] >= 198  # allow small rounding

    def test_brightness_zero(self):
        import efv_core as c
        f  = self._grey_frame(200)
        f2 = c.py_adjust_brightness(f, 0.0)
        px = f2.get_pixel(0, 0)
        assert px[0] == 0

    def test_contrast_amplifies(self):
        import efv_core as c
        f  = c.Frame(32, 32, 200, 200, 200, 255)
        f2 = c.py_adjust_contrast(f, 2.0)
        px = f2.get_pixel(0, 0)
        assert px[0] > 200  # pushed further from 128

    def test_saturation_zero_gives_grey(self):
        import efv_core as c
        f  = c.Frame(4, 4, 200, 50, 50, 255)
        f2 = c.py_adjust_saturation(f, 0.0)
        px = f2.get_pixel(0, 0)
        # All channels equal (greyscale)
        assert abs(px[0] - px[1]) <= 2
        assert abs(px[1] - px[2]) <= 2


# ── Gaussian blur ─────────────────────────────────────────────────────────

class TestBlur:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def test_blur_output_size(self):
        import efv_core as c
        f  = c.Frame.from_rgb(make_rgb_frame(64, 64), 64, 64)
        f2 = c.py_gaussian_blur(f, 2)
        assert f2.width == 64 and f2.height == 64

    def test_blur_zero_radius(self):
        import efv_core as c
        f  = c.Frame.from_rgb(make_rgb_frame(16, 16), 16, 16)
        f2 = c.py_gaussian_blur(f, 0)
        # No change
        assert f.to_rgb() == f2.to_rgb()


# ── Compositing ───────────────────────────────────────────────────────────

class TestComposite:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def test_composite_opaque_overwrites(self):
        import efv_core as c
        dst = c.Frame(10, 10, 0, 0, 0, 255)
        src = c.Frame(4, 4, 255, 0, 0, 255)
        c.py_composite_over(dst, src, 3, 3, 1.0)
        assert dst.get_pixel(4, 4) == (255, 0, 0, 255)  # inside src
        assert dst.get_pixel(0, 0) == (0, 0, 0, 255)    # outside src

    def test_blend_frames_midpoint(self):
        import efv_core as c
        a = c.Frame(4, 4, 0,   0,   0,   255)
        b = c.Frame(4, 4, 200, 200, 200, 255)
        m = c.py_blend_frames(a, b, 0.5)
        px = m.get_pixel(0, 0)
        assert 95 <= px[0] <= 105

    def test_blend_t0_equals_a(self):
        import efv_core as c
        a = c.Frame(4, 4, 80,  80,  80,  255)
        b = c.Frame(4, 4, 200, 200, 200, 255)
        m = c.py_blend_frames(a, b, 0.0)
        px = m.get_pixel(0, 0)
        assert px[0] == 80

    def test_blend_mismatched_sizes_raises(self):
        import efv_core as c
        a = c.Frame(4, 4)
        b = c.Frame(8, 8)
        with pytest.raises(Exception):
            c.py_blend_frames(a, b, 0.5)


# ── PCM Audio ─────────────────────────────────────────────────────────────

class TestPcmAudio:
    pytest.importorskip("efv_core", reason="efv_core not compiled")

    def test_mute_returns_silence(self):
        import efv_core as c
        pcm = make_tone_pcm(4410, 2)
        out = c.py_mute_pcm(4410, 2)
        assert all(b == 0 for b in out)

    def test_volume_zero(self):
        import efv_core as c
        pcm = make_tone_pcm(1000, 2)
        out = c.py_change_volume(pcm, 0.0, 16)
        assert all(b == 0 for b in out)

    def test_volume_identity(self):
        import efv_core as c
        pcm = make_tone_pcm(1000, 2)
        out = c.py_change_volume(pcm, 1.0, 16)
        assert out == pcm

    def test_trim_length(self):
        import efv_core as c
        sr  = 44100; ch = 2
        pcm = make_tone_pcm(sr * 5, ch)  # 5 seconds
        out = c.py_trim_pcm(pcm, sr, ch, 1.0, 3.0)  # keep seconds 1-3
        expected = sr * 2 * ch * 2
        assert abs(len(out) - expected) <= ch * 2  # allow 1 frame off

    def test_mix_doubles_silence(self):
        import efv_core as c
        a = make_silent_pcm(1000)
        b = make_silent_pcm(1000)
        m = c.py_mix_pcm(a, b, 1.0, 1.0)
        assert all(b == 0 for b in m)

    def test_pad_or_trim_extend(self):
        import efv_core as c
        pcm = make_silent_pcm(1000)
        out = c.py_pad_or_trim_pcm(pcm, 44100, 2, 2.0)
        assert len(out) == 44100 * 2 * 2 * 2  # 2 sec * sr * ch * bytes

    def test_pad_or_trim_truncate(self):
        import efv_core as c
        pcm = make_silent_pcm(44100 * 5)  # 5 seconds
        out = c.py_pad_or_trim_pcm(pcm, 44100, 2, 2.0)
        assert len(out) == 44100 * 2 * 2 * 2


# ── Pure Python ops ───────────────────────────────────────────────────────

class TestPurePythonOps:
    """These must work without the Rust extension."""

    def _frame(self, w=32, h=32, r=128, g=64, b=32):
        return dict(rgba=bytearray([r, g, b, 255] * w * h), width=w, height=h)

    def test_brightness(self):
        from efv.ops.color import apply_brightness
        f  = self._frame(r=100)
        f2 = apply_brightness(f, 2.0)
        assert f2["rgba"][0] == min(255, 200)

    def test_contrast(self):
        from efv.ops.color import apply_contrast
        f  = self._frame(r=200)
        f2 = apply_contrast(f, 2.0)
        assert f2["rgba"][0] >= 200  # pushed away from 128

    def test_saturation_zero(self):
        from efv.ops.color import apply_saturation
        f  = self._frame(r=200, g=50, b=50)
        f2 = apply_saturation(f, 0.0)
        # All RGB channels equal
        assert abs(f2["rgba"][0] - f2["rgba"][1]) <= 2

    def test_greyscale(self):
        from efv.ops.color import apply_greyscale
        f  = self._frame(r=200, g=100, b=50)
        f2 = apply_greyscale(f)
        assert f2["rgba"][0] == f2["rgba"][1] == f2["rgba"][2]

    def test_lut_identity(self):
        from efv.ops.color import apply_lut, build_gamma_lut
        lut = build_gamma_lut(1.0, 1.0, 1.0)
        f   = self._frame()
        f2  = apply_lut(f, lut)
        assert bytes(f2["rgba"]) == bytes(f["rgba"])

    def test_crop(self):
        from efv.ops.transform import apply_crop
        f  = self._frame(64, 64)
        f2 = apply_crop(f, 10, 10, 20, 20)
        assert f2["width"] == 20 and f2["height"] == 20

    def test_flip_horizontal(self):
        from efv.ops.transform import apply_flip
        rgba = bytearray([255, 0, 0, 255] + [0, 0, 255, 255])  # 2×1
        f    = dict(rgba=rgba, width=2, height=1)
        f2   = apply_flip(f, horizontal=True, vertical=False)
        assert f2["rgba"][0] == 0   # blue moved to left
        assert f2["rgba"][4] == 255 # red moved to right

    def test_resize_py(self):
        from efv.ops.transform import resize_bilinear_py
        f  = self._frame(32, 32)
        f2 = resize_bilinear_py(f, 16, 16)
        assert f2["width"] == 16 and f2["height"] == 16

    def test_blur(self):
        from efv.ops.filter_ import apply_blur
        f  = self._frame(32, 32)
        f2 = apply_blur(f, 2)
        assert len(f2["rgba"]) == len(f["rgba"])

    def test_vignette(self):
        from efv.ops.filter_ import apply_vignette
        f  = self._frame(64, 64, r=255, g=255, b=255)
        f2 = apply_vignette(f, strength=1.0, radius=0.0)
        # Corners should be darker than centre
        corner = f2["rgba"][0]
        centre_idx = (32 * 64 + 32) * 4
        centre = f2["rgba"][centre_idx]
        assert corner <= centre

    def test_sharpen(self):
        from efv.ops.filter_ import apply_sharpen
        f  = self._frame()
        f2 = apply_sharpen(f, 1.0)
        assert len(f2["rgba"]) == len(f["rgba"])


# ── Timeline ops ──────────────────────────────────────────────────────────

class TestTimelineOps:
    def _frames(self, n):
        return [(bytes(i % 256), "rgb24") for i in range(n)]

    def _pcm(self, secs, sr=44100, ch=2):
        return b"\x00" * int(secs * sr * ch * 2)

    def test_trim(self):
        from efv.ops.time_ import apply_trim
        frames = self._frames(100)
        pcm    = self._pcm(10)
        out_f, out_a, fps = apply_trim(frames, pcm, 10.0, 44100, 2, 2.0, 5.0)
        assert len(out_f) == 30  # frames 20..50

    def test_cut(self):
        from efv.ops.time_ import apply_cut
        frames = self._frames(100)
        pcm    = self._pcm(10)
        out_f, out_a = apply_cut(frames, pcm, 10.0, 44100, 2, 2.0, 4.0)
        assert len(out_f) == 80  # removed 20 frames

    def test_reverse(self):
        from efv.ops.time_ import apply_reverse
        frames = self._frames(10)
        out_f, out_a = apply_reverse(frames, b"\x00" * 10)
        assert out_f[0] == frames[-1]
        assert out_f[-1] == frames[0]

    def test_loop(self):
        from efv.ops.time_ import apply_loop
        frames = self._frames(10)
        pcm    = self._pcm(1)
        out_f, out_a = apply_loop(frames, pcm, 3)
        assert len(out_f) == 30
        assert len(out_a) == len(pcm) * 3

    def test_speed_2x(self):
        from efv.ops.time_ import apply_speed
        frames = self._frames(60)
        pcm    = self._pcm(1)
        out_f, out_a, new_fps = apply_speed(frames, pcm, 30.0, 44100, 2, 2.0)
        assert len(out_f) == pytest.approx(30, abs=3)
        assert new_fps == pytest.approx(60.0, rel=0.05)


# ── Audio ops ─────────────────────────────────────────────────────────────

class TestAudioOps:
    def _pcm(self, secs=2, sr=44100, ch=2, amp=16000):
        import math
        n   = int(secs * sr)
        out = bytearray(n * ch * 2)
        for i in range(n):
            v = int(amp * math.sin(2 * math.pi * 440 * i / sr))
            v = max(-32768, min(32767, v))
            for c in range(ch):
                struct.pack_into("<h", out, (i * ch + c) * 2, v)
        return bytes(out)

    def test_mute(self):
        from efv.ops.audio import apply_mute
        out = apply_mute(b"\xff"*1000, 30, 30.0, 44100, 2)
        assert all(b == 0 for b in out)

    def test_volume_zero(self):
        from efv.ops.audio import apply_volume
        pcm = self._pcm()
        out = apply_volume(pcm, 0.0)
        assert all(b == 0 for b in out)

    def test_volume_preserves_length(self):
        from efv.ops.audio import apply_volume
        pcm = self._pcm()
        out = apply_volume(pcm, 1.5)
        assert len(out) == len(pcm)

    def test_pad_or_trim_extend(self):
        from efv.ops.audio import pad_or_trim
        pcm = self._pcm(1)
        out = pad_or_trim(pcm, len(pcm) * 2)
        assert len(out) == len(pcm) * 2

    def test_pad_or_trim_truncate(self):
        from efv.ops.audio import pad_or_trim
        pcm = self._pcm(2)
        out = pad_or_trim(pcm, len(pcm) // 2)
        assert len(out) == len(pcm) // 2


# ── Video.info() ──────────────────────────────────────────────────────────

class TestVideoClass:
    def test_fluent_chain_builds_ops(self):
        import efv
        v = (efv.Video("dummy.avi")
                 .mute()
                 .resize(1280, 720)
                 .brightness(1.1)
                 .blur(2))
        ops = v.pending_ops()
        assert len(ops) == 4

    def test_reset(self):
        import efv
        v = efv.Video("dummy.avi").mute().resize(640, 480)
        assert len(v.pending_ops()) == 2
        v.reset()
        assert len(v.pending_ops()) == 0

    def test_set_quality(self):
        import efv
        v = efv.Video("dummy.avi").set_quality(50)
        assert v._quality == 50

    def test_repr(self):
        import efv
        r = repr(efv.Video("test.avi"))
        assert "test.avi" in r
