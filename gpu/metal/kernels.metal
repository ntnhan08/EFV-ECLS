/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · Apple Metal Compute Shaders
 *  Targets: macOS 10.14+, iOS 12+, Apple Silicon (M1/M2/M3)
 *  Compile: xcrun -sdk macosx metal -O3 -o kernels.metallib kernels.metal
 * ══════════════════════════════════════════════════════════════════════════
 */
#include <metal_stdlib>
using namespace metal;

// ── Bilinear resize ───────────────────────────────────────────────────────

kernel void resize_bilinear(
    device const uchar* src        [[ buffer(0) ]],
    device       uchar* dst        [[ buffer(1) ]],
    constant uint& src_w           [[ buffer(2) ]],
    constant uint& src_h           [[ buffer(3) ]],
    constant uint& dst_w           [[ buffer(4) ]],
    constant uint& dst_h           [[ buffer(5) ]],
    uint2 gid                      [[ thread_position_in_grid ]])
{
    if (gid.x >= dst_w || gid.y >= dst_h) return;

    float x_ratio = float(src_w) / float(dst_w);
    float y_ratio = float(src_h) / float(dst_h);

    float sx_f = (float(gid.x) + 0.5f) * x_ratio - 0.5f;
    float sy_f = (float(gid.y) + 0.5f) * y_ratio - 0.5f;

    int sx0 = clamp(int(sx_f),      0, int(src_w) - 1);
    int sx1 = clamp(int(sx_f) + 1,  0, int(src_w) - 1);
    int sy0 = clamp(int(sy_f),      0, int(src_h) - 1);
    int sy1 = clamp(int(sy_f) + 1,  0, int(src_h) - 1);

    float wx1 = sx_f - floor(sx_f);
    float wx0 = 1.0f - wx1;
    float wy1 = sy_f - floor(sy_f);
    float wy0 = 1.0f - wy1;

    device const uchar* p00 = src + (sy0 * int(src_w) + sx0) * 4;
    device const uchar* p10 = src + (sy0 * int(src_w) + sx1) * 4;
    device const uchar* p01 = src + (sy1 * int(src_w) + sx0) * 4;
    device const uchar* p11 = src + (sy1 * int(src_w) + sx1) * 4;

    device uchar* out = dst + (gid.y * dst_w + gid.x) * 4;
    for (int c = 0; c < 4; ++c) {
        float v = float(p00[c]) * wx0 * wy0
                + float(p10[c]) * wx1 * wy0
                + float(p01[c]) * wx0 * wy1
                + float(p11[c]) * wx1 * wy1;
        out[c] = uchar(clamp(int(v + 0.5f), 0, 255));
    }
}

// ── Nearest-neighbour resize ──────────────────────────────────────────────

kernel void resize_nearest(
    device const uchar* src        [[ buffer(0) ]],
    device       uchar* dst        [[ buffer(1) ]],
    constant uint& src_w           [[ buffer(2) ]],
    constant uint& src_h           [[ buffer(3) ]],
    constant uint& dst_w           [[ buffer(4) ]],
    constant uint& dst_h           [[ buffer(5) ]],
    uint2 gid                      [[ thread_position_in_grid ]])
{
    if (gid.x >= dst_w || gid.y >= dst_h) return;

    uint sx = clamp(uint(float(gid.x) / float(dst_w) * float(src_w)), 0u, src_w - 1);
    uint sy = clamp(uint(float(gid.y) / float(dst_h) * float(src_h)), 0u, src_h - 1);

    device const uchar* p   = src + (sy * src_w + sx) * 4;
    device       uchar* out = dst + (gid.y * dst_w + gid.x) * 4;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}

// ── Porter-Duff composite "over" ──────────────────────────────────────────

kernel void composite_over(
    device       uchar* dst        [[ buffer(0) ]],
    constant uint& dst_w           [[ buffer(1) ]],
    device const uchar* src        [[ buffer(2) ]],
    constant uint& src_w           [[ buffer(3) ]],
    constant uint& src_h           [[ buffer(4) ]],
    constant int&  x_off           [[ buffer(5) ]],
    constant int&  y_off           [[ buffer(6) ]],
    constant float& opacity        [[ buffer(7) ]],
    uint2 gid                      [[ thread_position_in_grid ]])
{
    int sx = int(gid.x);
    int sy = int(gid.y);
    if (sx >= int(src_w) || sy >= int(src_h)) return;

    int dx = sx + x_off;
    int dy = sy + y_off;
    if (dx < 0 || dy < 0) return;

    device const uchar* s = src + (sy * int(src_w) + sx) * 4;
    device       uchar* d = dst + (dy * int(dst_w) + dx) * 4;

    uint sa = uint(clamp(float(s[3]) * opacity, 0.0f, 255.0f));
    if (sa == 0) return;

    if (sa >= 254) {
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
        return;
    }
    uint inv = 255 - sa;
    uint da  = d[3];
    for (int c = 0; c < 3; ++c)
        d[c] = uchar((uint(s[c]) * sa + uint(d[c]) * inv) >> 8);
    d[3] = uchar(sa + ((da * inv) >> 8));
}

// ── Brightness ────────────────────────────────────────────────────────────

kernel void adjust_brightness(
    device uchar* rgba             [[ buffer(0) ]],
    constant float& factor         [[ buffer(1) ]],
    uint gid                       [[ thread_position_in_grid ]])
{
    device uchar* p = rgba + gid * 4;
    for (int c = 0; c < 3; ++c) {
        int v = int(float(p[c]) * factor + 0.5f);
        p[c] = uchar(clamp(v, 0, 255));
    }
}

// ── Contrast ──────────────────────────────────────────────────────────────

kernel void adjust_contrast(
    device uchar* rgba             [[ buffer(0) ]],
    constant float& factor         [[ buffer(1) ]],
    uint gid                       [[ thread_position_in_grid ]])
{
    device uchar* p = rgba + gid * 4;
    for (int c = 0; c < 3; ++c) {
        int v = int((float(p[c]) - 128.0f) * factor + 128.0f + 0.5f);
        p[c] = uchar(clamp(v, 0, 255));
    }
}

// ── Saturation ────────────────────────────────────────────────────────────

kernel void adjust_saturation(
    device uchar* rgba             [[ buffer(0) ]],
    constant float& factor         [[ buffer(1) ]],
    uint gid                       [[ thread_position_in_grid ]])
{
    device uchar* p = rgba + gid * 4;
    float r = float(p[0]), g = float(p[1]), b = float(p[2]);
    float luma = (r * 77.0f + g * 150.0f + b * 29.0f) / 256.0f;
    p[0] = uchar(clamp(int(luma + (r - luma) * factor + 0.5f), 0, 255));
    p[1] = uchar(clamp(int(luma + (g - luma) * factor + 0.5f), 0, 255));
    p[2] = uchar(clamp(int(luma + (b - luma) * factor + 0.5f), 0, 255));
}
