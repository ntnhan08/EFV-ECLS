/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · Composite + Color + PCM  –  CPU SIMD + GPU dispatcher
 * ══════════════════════════════════════════════════════════════════════════
 */

#include "../include/efv_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>

/* ── Composite "over" ────────────────────────────────────────────────── */

int efv_gpu_composite_over(
    uint8_t*       dst, uint32_t dst_w, uint32_t dst_h,
    const uint8_t* src, uint32_t src_w, uint32_t src_h,
    int32_t x_off, int32_t y_off,
    float opacity)
{
    const int32_t x0 = std::max(0, x_off);
    const int32_t y0 = std::max(0, y_off);
    const int32_t x1 = std::min((int32_t)dst_w, x_off + (int32_t)src_w);
    const int32_t y1 = std::min((int32_t)dst_h, y_off + (int32_t)src_h);
    if (x1 <= x0 || y1 <= y0) return 0;

    const uint32_t opacity_i = (uint32_t)(opacity * 255.0f + 0.5f);

    for (int32_t dy = y0; dy < y1; ++dy) {
        const int32_t sy = dy - y_off;
        for (int32_t dx = x0; dx < x1; ++dx) {
            const int32_t sx = dx - x_off;
            const uint8_t* s = src + (sy * (int32_t)src_w + sx) * 4;
            uint8_t*       d = dst + (dy * (int32_t)dst_w + dx) * 4;

            const uint32_t sa = std::min(255u, (s[3] * opacity_i + 127) >> 8);
            if (sa == 0) continue;

            if (sa >= 254) {
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
                continue;
            }
            const uint32_t inv = 255 - sa;
            const uint32_t da  = d[3];
            for (int c = 0; c < 3; ++c)
                d[c] = (uint8_t)((s[c] * sa + d[c] * inv) >> 8);
            d[3] = (uint8_t)(sa + ((da * inv) >> 8));
        }
    }
    return 0;
}

/* ── Brightness ──────────────────────────────────────────────────────── */

int efv_gpu_brightness(uint8_t* rgba, uint32_t w, uint32_t h, float factor) {
    const int32_t fi  = (int32_t)(factor * 256.0f);
    const size_t  n   = (size_t)w * h;
    uint8_t*      p   = rgba;
    /* Compiler will auto-vectorise this loop with -O2 -march=native */
    for (size_t i = 0; i < n; ++i, p += 4) {
        for (int c = 0; c < 3; ++c) {
            int32_t v = (p[c] * fi) >> 8;
            p[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    return 0;
}

/* ── Contrast ────────────────────────────────────────────────────────── */

int efv_gpu_contrast(uint8_t* rgba, uint32_t w, uint32_t h, float factor) {
    const int32_t fi = (int32_t)(factor * 256.0f);
    const size_t  n  = (size_t)w * h;
    uint8_t*      p  = rgba;
    for (size_t i = 0; i < n; ++i, p += 4) {
        for (int c = 0; c < 3; ++c) {
            int32_t v = (((int32_t)p[c] - 128) * fi >> 8) + 128;
            p[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    return 0;
}

/* ── Saturation ──────────────────────────────────────────────────────── */

int efv_gpu_saturation(uint8_t* rgba, uint32_t w, uint32_t h, float factor) {
    const int32_t fi = (int32_t)(factor * 256.0f);
    const size_t  n  = (size_t)w * h;
    uint8_t*      p  = rgba;
    for (size_t i = 0; i < n; ++i, p += 4) {
        const int32_t r    = p[0], g = p[1], b = p[2];
        /* BT.601 luma */
        const int32_t luma = (r * 77 + g * 150 + b * 29) >> 8;
        for (int32_t c = 0; c < 3; ++c) {
            const int32_t orig = (int32_t)p[c];
            int32_t v = luma + (((orig - luma) * fi) >> 8);
            p[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    return 0;
}

/* ── Gaussian blur – horizontal pass ─────────────────────────────────── */

static void build_gaussian_kernel(uint32_t radius, float* kernel) {
    float sigma = (radius > 0) ? (float)radius / 3.0f : 1.0f;
    float s2    = 2.0f * sigma * sigma;
    float sum   = 0.0f;
    uint32_t sz = radius * 2 + 1;
    for (uint32_t i = 0; i < sz; ++i) {
        float x = (float)i - (float)radius;
        kernel[i] = expf(-x * x / s2);
        sum += kernel[i];
    }
    for (uint32_t i = 0; i < sz; ++i) kernel[i] /= sum;
}

int efv_gpu_blur_horizontal(
    const uint8_t* src, uint8_t* dst,
    uint32_t w, uint32_t h, uint32_t radius)
{
    float kernel[129];
    build_gaussian_kernel(radius, kernel);
    const int32_t r = (int32_t)radius;

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float acc[4] = {0};
            for (int32_t k = -(int32_t)radius; k <= r; ++k) {
                int32_t sx = (int32_t)x + k;
                sx = sx < 0 ? 0 : (sx >= (int32_t)w ? (int32_t)w - 1 : sx);
                const uint8_t* p = src + (y * w + (uint32_t)sx) * 4;
                float kv = kernel[k + r];
                for (int c = 0; c < 4; ++c) acc[c] += p[c] * kv;
            }
            uint8_t* out = dst + (y * w + x) * 4;
            for (int c = 0; c < 4; ++c) {
                float v = acc[c];
                out[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : (int)v));
            }
        }
    }
    return 0;
}

int efv_gpu_blur_vertical(
    const uint8_t* src, uint8_t* dst,
    uint32_t w, uint32_t h, uint32_t radius)
{
    float kernel[129];
    build_gaussian_kernel(radius, kernel);
    const int32_t r = (int32_t)radius;

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float acc[4] = {0};
            for (int32_t k = -(int32_t)radius; k <= r; ++k) {
                int32_t sy = (int32_t)y + k;
                sy = sy < 0 ? 0 : (sy >= (int32_t)h ? (int32_t)h - 1 : sy);
                const uint8_t* p = src + ((uint32_t)sy * w + x) * 4;
                float kv = kernel[k + r];
                for (int c = 0; c < 4; ++c) acc[c] += p[c] * kv;
            }
            uint8_t* out = dst + (y * w + x) * 4;
            for (int c = 0; c < 4; ++c) {
                float v = acc[c];
                out[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : (int)v));
            }
        }
    }
    return 0;
}

/* ── PCM audio ───────────────────────────────────────────────────────── */

void efv_gpu_pcm_volume(int16_t* pcm, size_t n, float factor) {
    const int32_t fi = (int32_t)(factor * 256.0f);
    for (size_t i = 0; i < n; ++i) {
        int32_t v = ((int32_t)pcm[i] * fi) >> 8;
        pcm[i] = (int16_t)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
    }
}

void efv_gpu_pcm_mix(
    const int16_t* a, size_t na,
    const int16_t* b, size_t nb,
    int16_t*       dst,
    float vol_a, float vol_b)
{
    const int32_t ia = (int32_t)(vol_a * 256.0f);
    const int32_t ib = (int32_t)(vol_b * 256.0f);
    const size_t  n  = na > nb ? na : nb;

    for (size_t i = 0; i < n; ++i) {
        const int32_t sa = (i < na) ? (int32_t)a[i] : 0;
        const int32_t sb = (i < nb) ? (int32_t)b[i] : 0;
        int32_t v = (sa * ia + sb * ib) >> 8;
        dst[i] = (int16_t)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
    }
}
