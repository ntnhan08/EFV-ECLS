/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · Resize  –  CPU SIMD + GPU dispatcher
 *  Compiler: auto-vectorised C++ (works on x86, ARM64, RISC-V).
 *  With -march=native the compiler will emit SSE/AVX/NEON intrinsics.
 * ══════════════════════════════════════════════════════════════════════════
 */

#include "../include/efv_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>

/* ── Forward declarations (GPU backends) ─────────────────────────────── */

#ifdef EFV_CUDA
    extern int efv_cuda_resize_bilinear(
        const uint8_t*, uint8_t*,
        uint32_t, uint32_t, uint32_t, uint32_t);
    extern int efv_cuda_resize_nearest(
        const uint8_t*, uint8_t*,
        uint32_t, uint32_t, uint32_t, uint32_t);
#endif
#ifdef EFV_OPENCL
    extern int efv_opencl_resize_bilinear(
        const uint8_t*, uint8_t*,
        uint32_t, uint32_t, uint32_t, uint32_t);
    extern int efv_opencl_resize_nearest(
        const uint8_t*, uint8_t*,
        uint32_t, uint32_t, uint32_t, uint32_t);
#endif
#ifdef EFV_METAL
    extern int efv_metal_resize_bilinear(
        const uint8_t*, uint8_t*,
        uint32_t, uint32_t, uint32_t, uint32_t);
    extern int efv_metal_resize_nearest(
        const uint8_t*, uint8_t*,
        uint32_t, uint32_t, uint32_t, uint32_t);
#endif

/* ── CPU: Bilinear ───────────────────────────────────────────────────── */

static int cpu_resize_bilinear(
    const uint8_t* src, uint8_t* dst,
    uint32_t sw, uint32_t sh,
    uint32_t dw, uint32_t dh)
{
    const float x_ratio = (float)sw / (float)dw;
    const float y_ratio = (float)sh / (float)dh;

    for (uint32_t dy = 0; dy < dh; ++dy) {
        const float sy_f = (dy + 0.5f) * y_ratio - 0.5f;
        const int   sy0  = (int)std::max(0.0f, std::min(sy_f, (float)(sh - 1)));
        const int   sy1  = std::min(sy0 + 1, (int)sh - 1);
        const float wy1  = sy_f - std::floor(sy_f);
        const float wy0  = 1.0f - wy1;

        uint8_t* dst_row = dst + dy * dw * 4;

        for (uint32_t dx = 0; dx < dw; ++dx) {
            const float sx_f = (dx + 0.5f) * x_ratio - 0.5f;
            const int   sx0  = (int)std::max(0.0f, std::min(sx_f, (float)(sw - 1)));
            const int   sx1  = std::min(sx0 + 1, (int)sw - 1);
            const float wx1  = sx_f - std::floor(sx_f);
            const float wx0  = 1.0f - wx1;

            const uint8_t* p00 = src + (sy0 * sw + sx0) * 4;
            const uint8_t* p10 = src + (sy0 * sw + sx1) * 4;
            const uint8_t* p01 = src + (sy1 * sw + sx0) * 4;
            const uint8_t* p11 = src + (sy1 * sw + sx1) * 4;

            uint8_t* out = dst_row + dx * 4;
            for (int c = 0; c < 4; ++c) {
                float v = p00[c] * wx0 * wy0
                        + p10[c] * wx1 * wy0
                        + p01[c] * wx0 * wy1
                        + p11[c] * wx1 * wy1;
                out[c] = (uint8_t)std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
    return 0;
}

/* ── CPU: Nearest-neighbour ──────────────────────────────────────────── */

static int cpu_resize_nearest(
    const uint8_t* src, uint8_t* dst,
    uint32_t sw, uint32_t sh,
    uint32_t dw, uint32_t dh)
{
    const float x_ratio = (float)sw / (float)dw;
    const float y_ratio = (float)sh / (float)dh;

    for (uint32_t dy = 0; dy < dh; ++dy) {
        const uint32_t sy  = std::min((uint32_t)(dy * y_ratio), sh - 1);
        uint8_t*       out = dst + dy * dw * 4;
        for (uint32_t dx = 0; dx < dw; ++dx) {
            const uint32_t sx = std::min((uint32_t)(dx * x_ratio), sw - 1);
            const uint8_t* p  = src + (sy * sw + sx) * 4;
            uint8_t*       q  = out + dx * 4;
            q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = p[3];
        }
    }
    return 0;
}

/* ── Public API (dispatcher) ─────────────────────────────────────────── */

extern EfvBackend efv_gpu_current_backend(void);

int efv_gpu_resize_bilinear(
    const uint8_t* src, uint8_t* dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h)
{
    switch (efv_gpu_current_backend()) {
#ifdef EFV_CUDA
        case EFV_BACKEND_CUDA:
            return efv_cuda_resize_bilinear(src, dst, src_w, src_h, dst_w, dst_h);
#endif
#ifdef EFV_OPENCL
        case EFV_BACKEND_OPENCL:
            return efv_opencl_resize_bilinear(src, dst, src_w, src_h, dst_w, dst_h);
#endif
#ifdef EFV_METAL
        case EFV_BACKEND_METAL:
            return efv_metal_resize_bilinear(src, dst, src_w, src_h, dst_w, dst_h);
#endif
        default:
            return cpu_resize_bilinear(src, dst, src_w, src_h, dst_w, dst_h);
    }
}

int efv_gpu_resize_nearest(
    const uint8_t* src, uint8_t* dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h)
{
    switch (efv_gpu_current_backend()) {
#ifdef EFV_CUDA
        case EFV_BACKEND_CUDA:
            return efv_cuda_resize_nearest(src, dst, src_w, src_h, dst_w, dst_h);
#endif
#ifdef EFV_OPENCL
        case EFV_BACKEND_OPENCL:
            return efv_opencl_resize_nearest(src, dst, src_w, src_h, dst_w, dst_h);
#endif
#ifdef EFV_METAL
        case EFV_BACKEND_METAL:
            return efv_metal_resize_nearest(src, dst, src_w, src_h, dst_w, dst_h);
#endif
        default:
            return cpu_resize_nearest(src, dst, src_w, src_h, dst_w, dst_h);
    }
}
