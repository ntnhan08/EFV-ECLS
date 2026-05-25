# EFV — Extreme Fast Video

**Ultra-fast · Ultra-light · Cross-platform · GPU-accelerated Python video processing**

```
┌─────────────────────────────────────────────────────────────┐
│  Python API  →  Rust Core (PyO3/rayon)  →  C++ GPU Layer   │
│              ↘  Pure-Python fallback  ↗                     │
│                                                             │
│  Backends: CUDA · OpenCL · Apple Metal · CPU SIMD           │
│  OS:       Windows · Linux · macOS (Intel + Apple Silicon)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

| Category | Functions |
|---|---|
| **Audio** | `mute` · `add_audio` · `volume` · `fade_audio` |
| **Overlays** | `add_logo` · `add_watermark` · `add_text` · `add_subtitle` |
| **Geometry** | `resize` · `crop` · `zoom` · `rotate` · `flip` |
| **Color** | `brightness` · `contrast` · `saturation` · `greyscale` · `apply_lut` |
| **Filters** | `blur` · `sharpen` · `denoise` · `vignette` |
| **Timeline** | `trim` · `cut` · `speed` · `reverse` · `loop` |
| **Transitions** | `fade_in` · `fade_out` · `crossfade` |

---

## Installation

### Prerequisites

| Tool | Purpose | Required? |
|------|---------|-----------|
| Python ≥ 3.8 | Runtime | ✅ |
| Rust + maturin | Compile Rust core | For fast mode |
| CMake ≥ 3.18 | Build GPU library | For GPU mode |
| CUDA Toolkit | NVIDIA GPU | Optional |
| OpenCL SDK | AMD/Intel GPU | Optional |
| Xcode | Apple Metal (macOS) | Optional |
| Pillow | Text/PNG rendering fallback | Recommended |

### Quick install (development)

```bash
# 1. Clone
git clone https://github.com/efv-io/efv && cd efv

# 2. Install Rust (if not present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. Build everything + install
make dev          # builds Rust + GPU (auto-detect) + installs Python package

# OR step by step:
make rust         # Rust/PyO3 extension only
make gpu          # C++ GPU library (auto-detects CUDA/OpenCL/Metal)
pip install -e ".[pillow]"
```

### GPU-specific builds

```bash
make gpu-cuda    # Force NVIDIA CUDA
make gpu-opencl  # Force OpenCL (AMD/Intel/cross-platform)
make gpu-metal   # Force Apple Metal (macOS only)
make gpu-cpu     # CPU SIMD only (no GPU required)
```

### From wheel (end users)

```bash
pip install efv          # pure Python (no native deps)
pip install efv[pillow]  # + text/PNG rendering
```

---

## Usage

### Fluent (chainable) API

```python
import efv

(efv.Video("input.avi", quality=90)

    # Timeline
    .trim(start_sec=5, end_sec=60)
    .cut(start_sec=30, end_sec=33)   # remove a glitch

    # Geometry
    .resize(1920, 1080)
    .zoom(1.05)                       # slight digital zoom
    .rotate(0)

    # Audio
    .volume(1.2)
    .add_audio("music.wav", volume=0.4)
    .fade_audio(in_sec=1.0, out_sec=2.0)

    # Color correction
    .brightness(1.05)
    .contrast(1.1)
    .saturation(1.15)

    # LUT colour grade (warm tone)
    .apply_lut(efv.build_gamma_lut(gamma_r=1.1, gamma_g=1.0, gamma_b=0.9))

    # Filters
    .sharpen(0.8)
    .vignette(0.25, radius=0.8)

    # Overlays
    .add_logo("logo.png", position="topright", scale=0.12, opacity=0.9)
    .add_watermark("© My Channel", opacity=0.2)
    .add_subtitle("Welcome!", start_sec=2.0, end_sec=5.0)

    # Transitions
    .fade_in(0.5)
    .fade_out(1.0)

    .save("output.avi")
)
```

### Standalone one-liner functions

```python
import efv

efv.mute("in.avi", "out.avi")
efv.add_logo("in.avi", "out.avi", logo_path="logo.png", position="bottomright")
efv.resize("in.avi", "out.avi", 1280, 720)
efv.brightness("in.avi", "out.avi", factor=1.2)
efv.trim("in.avi", "out.avi", start_sec=10, end_sec=30)
efv.speed("in.avi", "out.avi", factor=2.0)
efv.reverse("in.avi", "out.avi")
efv.blur("in.avi", "out.avi", radius=3)
efv.add_text("in.avi", "out.avi", "Hello!", x=20, y=20, font_size=40)
efv.greyscale("in.avi", "out.avi")
efv.crossfade("in.avi", "out.avi", other_path="in2.avi", duration_sec=1.5)
```

### Video info

```python
meta = efv.info("input.avi")
print(meta["width"], meta["height"], meta["fps"], meta["duration_sec"])
```

### Backend detection

```python
print(efv.backend_name())   # "CUDA" / "OpenCL" / "Metal" / "CPU-Rust" / "CPU-Python"
print(efv.gpu_available())  # True / False
```

---

## Architecture

```
efv/                          Python package
├── __init__.py               Public API (Video class + standalone functions)
├── video.py                  Fluent Video class
├── pipeline.py               Op descriptors (immutable dataclasses)
├── executor.py               Pipeline executor (decode → ops → encode)
├── backend.py                Load Rust core + GPU lib at import time
├── ops/
│   ├── audio.py              Audio ops (pure-Python fallback)
│   ├── visual.py             Logo/text/subtitle overlays
│   ├── transform.py          Crop/rotate/flip/resize
│   ├── color.py              Brightness/contrast/saturation/LUT
│   ├── time_.py              Trim/cut/speed/reverse/loop
│   └── filter_.py            Blur/sharpen/denoise/vignette
└── py_compat/
    ├── avi_reader.py         Pure-Python AVI reader fallback
    └── avi_writer.py         Pure-Python MJPEG-AVI writer fallback

src/                          Rust source (compiled by maturin → efv_core.so)
├── lib.rs                    PyO3 module registration
├── frame.rs                  RGBA frame buffer (pyclass)
├── codec/jpeg.rs             JPEG encode/decode (jpeg-encoder/decoder crates)
├── codec/png.rs              PNG decode (png crate)
├── filter/resize.rs          Bilinear + nearest resize (rayon parallel)
├── filter/composite.rs       Porter-Duff alpha compositing (rayon parallel)
├── filter/color.rs           Color ops (rayon parallel)
├── filter/blur.rs            Separable Gaussian blur (rayon parallel)
├── audio/pcm.rs              PCM manipulation
└── container/avi.rs          AVI RIFF reader + MJPEG writer

gpu/                          C++ GPU shared library (libefv_gpu.so)
├── include/efv_gpu.h         Public C API (ctypes-compatible)
├── src/device.cpp            Backend auto-detection (CUDA/OpenCL/Metal/CPU)
├── src/resize.cpp            Resize dispatcher + CPU SIMD implementation
├── src/composite.cpp         Composite/color/PCM CPU SIMD + dispatcher
├── src/resize.cu             CUDA resize kernels (compiled if CUDA available)
├── opencl/resize.cl          OpenCL resize + composite kernels
└── metal/kernels.metal       Apple Metal compute shaders
```

### Performance tiers

| Operation | CPU-Python | CPU-Rust | CUDA (RTX 3080) |
|-----------|-----------|---------|----------------|
| 1080p resize | ~0.8 fps | ~45 fps | ~820 fps |
| Gaussian blur r=3 | ~0.5 fps | ~30 fps | ~600 fps |
| Brightness | ~4 fps | ~180 fps | ~2400 fps |
| JPEG encode q=85 | ~12 fps | ~120 fps | N/A (CPU codec) |

---

## Building & Testing

```bash
make all          # Build Rust + GPU
make test         # Run unit tests
make bench        # Run benchmarks
make clean        # Remove build artifacts
make help         # Show all targets
```

---

## LUT Colour Grading

```python
import efv

# Warm grade: boost reds, neutral greens, cooler blues
warm_lut = efv.build_gamma_lut(gamma_r=1.1, gamma_g=1.0, gamma_b=0.88)
efv.apply_lut("in.avi", "warm.avi", lut=warm_lut)

# Cool/cinematic look
cool_lut = efv.build_gamma_lut(gamma_r=0.9, gamma_g=0.95, gamma_b=1.15)
efv.apply_lut("in.avi", "cool.avi", lut=cool_lut)
```

---

## License

MIT © EFV Team
