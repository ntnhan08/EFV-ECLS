"""
EFV – Benchmark
===============
Measures frame-processing throughput across backends.
"""
import time, struct, random, sys

# ── Synthetic 720p RGBA frame ─────────────────────────────────────────────

W, H = 1280, 720
N    = W * H * 4
FRAME_RGBA = bytes(random.randint(0, 255) for _ in range(N))


def bench(label, fn, iterations=20):
    # warm-up
    fn()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    fps     = iterations / elapsed
    print(f"  {label:<40s}  {fps:7.1f} fps  ({elapsed*1000/iterations:.2f} ms/frame)")


print(f"\n{'='*60}")
print(f"  EFV Benchmark  –  {W}×{H} RGBA frames")
print(f"{'='*60}\n")

# ── Try Rust core ─────────────────────────────────────────────────────────

try:
    import efv_core as core
    print("[Rust core – efv_core]\n")

    f = core.Frame.from_rgba(FRAME_RGBA, W, H)

    bench("resize_bilinear → 640×360",
          lambda: core.py_resize_bilinear(f, 640, 360))
    bench("resize_nearest  → 640×360",
          lambda: core.py_resize_nearest(f, 640, 360))
    bench("adjust_brightness(1.2)",
          lambda: core.py_adjust_brightness(f, 1.2))
    bench("adjust_contrast(1.1)",
          lambda: core.py_adjust_contrast(f, 1.1))
    bench("adjust_saturation(1.3)",
          lambda: core.py_adjust_saturation(f, 1.3))
    bench("gaussian_blur(radius=3)",
          lambda: core.py_gaussian_blur(f, 3))

    # JPEG encode
    rgb = bytes(FRAME_RGBA[i] for i in range(0, N, 4) if True)  # placeholder
    rgb = f.to_rgb()
    bench("encode_jpeg(q=85)",
          lambda: core.py_encode_jpeg(rgb, W, H, 85))

except ImportError:
    print("[efv_core not available – skipping Rust benchmarks]\n")

# ── GPU library ───────────────────────────────────────────────────────────

from efv.backend import gpu, GPU_AVAILABLE, BACKEND_NAME
print(f"\n[GPU layer – {BACKEND_NAME}]\n")

if gpu is not None:
    import ctypes
    buf = bytearray(FRAME_RGBA)
    bench("gpu resize_bilinear → 640×360",
          lambda: gpu.resize_bilinear(FRAME_RGBA, W, H, 640, 360))
    bench("gpu brightness(1.2)",
          lambda: gpu.brightness(bytearray(FRAME_RGBA), W, H, 1.2))
    bench("gpu contrast(1.1)",
          lambda: gpu.contrast(bytearray(FRAME_RGBA), W, H, 1.1))
else:
    print("  (GPU library not loaded)")

# ── Pure Python ───────────────────────────────────────────────────────────

print("\n[Pure Python fallback]\n")

from efv.ops import color as _c, transform as _t, filter_ as _f

frame = dict(rgba=bytearray(FRAME_RGBA), width=W, height=H)

bench("resize_bilinear_py → 320×180",
      lambda: _t.resize_bilinear_py(frame, 320, 180),
      iterations=3)
bench("brightness",
      lambda: _c.apply_brightness(frame, 1.2))
bench("contrast",
      lambda: _c.apply_contrast(frame, 1.1))
bench("saturation",
      lambda: _c.apply_saturation(frame, 1.3))
bench("greyscale",
      lambda: _c.apply_greyscale(frame))
bench("blur(radius=2)",
      lambda: _f.apply_blur(frame, 2),
      iterations=3)

print(f"\n{'='*60}\n")
