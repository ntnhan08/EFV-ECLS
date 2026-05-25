/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · GPU Acceleration Layer  –  C API header
 *  Used by Python (ctypes) and Rust (via FFI).
 *  Compiled backends: CUDA · OpenCL · Metal · CPU-SIMD (fallback)
 * ══════════════════════════════════════════════════════════════════════════
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ── Backend identifiers ─────────────────────────────────────────────── */

typedef enum EfvBackend {
    EFV_BACKEND_CPU    = 0,
    EFV_BACKEND_CUDA   = 1,
    EFV_BACKEND_OPENCL = 2,
    EFV_BACKEND_METAL  = 3,
} EfvBackend;

/* ── Init / teardown ─────────────────────────────────────────────────── */

/**
 * Initialise the GPU subsystem and auto-detect the best available backend.
 * Returns the selected backend.  Call once before any other efv_gpu_* fn.
 */
EfvBackend efv_gpu_init(void);

/** Release all GPU resources (call on shutdown). */
void efv_gpu_shutdown(void);

/** Return the name string of the active backend. */
const char* efv_gpu_backend_name(void);

/** Return 1 if GPU acceleration is available, 0 for CPU fallback. */
int efv_gpu_available(void);

/* ── Frame operations ────────────────────────────────────────────────── */

/**
 * Bilinear resize  (RGBA → RGBA).
 *
 * src      : input  RGBA buffer (src_w * src_h * 4 bytes)
 * dst      : output RGBA buffer (dst_w * dst_h * 4 bytes, caller-allocated)
 * src_w/h  : source dimensions
 * dst_w/h  : destination dimensions
 * Returns 0 on success, non-zero on error.
 */
int efv_gpu_resize_bilinear(
    const uint8_t* src, uint8_t* dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h
);

/**
 * Nearest-neighbour resize (fast preview quality).
 */
int efv_gpu_resize_nearest(
    const uint8_t* src, uint8_t* dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h
);

/**
 * Porter-Duff "over" composite: dst_rgba[i] = src_rgba[i] over dst_rgba[i].
 *
 * dst      : destination RGBA buffer (modified in-place)
 * src      : source     RGBA buffer
 * dst_w/h  : destination frame dimensions
 * src_w/h  : source overlay dimensions
 * x_off    : pixel X offset of overlay (may be negative)
 * y_off    : pixel Y offset of overlay (may be negative)
 * opacity  : source alpha multiplier in [0.0, 1.0]
 */
int efv_gpu_composite_over(
    uint8_t*       dst, uint32_t dst_w, uint32_t dst_h,
    const uint8_t* src, uint32_t src_w, uint32_t src_h,
    int32_t x_off, int32_t y_off,
    float opacity
);

/**
 * Brightness adjustment: multiply RGB channels by `factor`.
 * Operates in-place.
 */
int efv_gpu_brightness(uint8_t* rgba, uint32_t width, uint32_t height, float factor);

/**
 * Contrast adjustment about mid-grey (128): (v-128)*factor+128.
 * Operates in-place.
 */
int efv_gpu_contrast(uint8_t* rgba, uint32_t width, uint32_t height, float factor);

/**
 * Saturation adjustment: 0 = greyscale, 1 = no change, 2 = double.
 * Operates in-place.
 */
int efv_gpu_saturation(uint8_t* rgba, uint32_t width, uint32_t height, float factor);

/**
 * Separable horizontal Gaussian blur pass (radius ≤ 64).
 *
 * src  : input  RGBA buffer
 * dst  : output RGBA buffer (caller-allocated)
 */
int efv_gpu_blur_horizontal(
    const uint8_t* src, uint8_t* dst,
    uint32_t width, uint32_t height,
    uint32_t radius
);

/** Vertical Gaussian blur pass. */
int efv_gpu_blur_vertical(
    const uint8_t* src, uint8_t* dst,
    uint32_t width, uint32_t height,
    uint32_t radius
);

/* ── PCM audio ───────────────────────────────────────────────────────── */

/**
 * Scale 16-bit signed LE PCM amplitude.
 * factor 1.0 = no change.
 * Operates in-place.
 */
void efv_gpu_pcm_volume(int16_t* pcm, size_t n_samples, float factor);

/**
 * Mix two 16-bit PCM streams into `dst`.
 * dst_n = max(n_a, n_b).
 */
void efv_gpu_pcm_mix(
    const int16_t* a, size_t n_a,
    const int16_t* b, size_t n_b,
    int16_t*       dst,
    float vol_a, float vol_b
);

#ifdef __cplusplus
}  /* extern "C" */
#endif
