/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · OpenCL Resize Kernels
 *  Works on AMD, Intel, NVIDIA, Apple Silicon (via OpenCL 1.2+).
 * ══════════════════════════════════════════════════════════════════════════
 */

__kernel void resize_bilinear(
    __global const uchar* src,
    __global       uchar* dst,
    const uint src_w, const uint src_h,
    const uint dst_w, const uint dst_h)
{
    const uint dx = get_global_id(0);
    const uint dy = get_global_id(1);
    if (dx >= dst_w || dy >= dst_h) return;

    const float x_ratio = (float)src_w / (float)dst_w;
    const float y_ratio = (float)src_h / (float)dst_h;

    const float sx_f = (dx + 0.5f) * x_ratio - 0.5f;
    const float sy_f = (dy + 0.5f) * y_ratio - 0.5f;

    const int sx0 = clamp((int)sx_f,      0, (int)src_w - 1);
    const int sx1 = clamp((int)sx_f + 1,  0, (int)src_w - 1);
    const int sy0 = clamp((int)sy_f,      0, (int)src_h - 1);
    const int sy1 = clamp((int)sy_f + 1,  0, (int)src_h - 1);

    const float wx1 = sx_f - floor(sx_f);
    const float wx0 = 1.0f - wx1;
    const float wy1 = sy_f - floor(sy_f);
    const float wy0 = 1.0f - wy1;

    __global const uchar* p00 = src + (sy0 * src_w + sx0) * 4;
    __global const uchar* p10 = src + (sy0 * src_w + sx1) * 4;
    __global const uchar* p01 = src + (sy1 * src_w + sx0) * 4;
    __global const uchar* p11 = src + (sy1 * src_w + sx1) * 4;

    __global uchar* out = dst + (dy * dst_w + dx) * 4;
    for (int c = 0; c < 4; ++c) {
        float v = p00[c] * wx0 * wy0
                + p10[c] * wx1 * wy0
                + p01[c] * wx0 * wy1
                + p11[c] * wx1 * wy1;
        out[c] = (uchar)clamp((int)(v + 0.5f), 0, 255);
    }
}

__kernel void resize_nearest(
    __global const uchar* src,
    __global       uchar* dst,
    const uint src_w, const uint src_h,
    const uint dst_w, const uint dst_h)
{
    const uint dx = get_global_id(0);
    const uint dy = get_global_id(1);
    if (dx >= dst_w || dy >= dst_h) return;

    const uint sx = clamp((uint)((float)dx / dst_w * src_w), 0u, src_w - 1);
    const uint sy = clamp((uint)((float)dy / dst_h * src_h), 0u, src_h - 1);

    __global const uchar* p   = src + (sy * src_w + sx) * 4;
    __global       uchar* out = dst + (dy * dst_w + dx) * 4;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}

__kernel void composite_over(
    __global       uchar* dst,  const uint dst_w, const uint dst_h,
    __global const uchar* src,  const uint src_w, const uint src_h,
    const int x_off, const int y_off,
    const float opacity)
{
    const int dx = (int)get_global_id(0) + (x_off > 0 ? x_off : 0);
    const int dy = (int)get_global_id(1) + (y_off > 0 ? y_off : 0);
    if (dx < 0 || dy < 0) return;
    if ((uint)dx >= dst_w || (uint)dy >= dst_h) return;

    const int sx = dx - x_off;
    const int sy = dy - y_off;
    if (sx < 0 || sy < 0 || (uint)sx >= src_w || (uint)sy >= src_h) return;

    __global const uchar* s = src + (sy * src_w + sx) * 4;
    __global       uchar* d = dst + (dy * dst_w + dx) * 4;

    const uint sa = clamp((uint)((s[3] * opacity * 255.0f + 127.0f) / 255.0f), 0u, 255u);
    if (sa == 0) return;

    if (sa >= 254) {
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
        return;
    }
    const uint inv = 255 - sa;
    const uint da  = d[3];
    for (int c = 0; c < 3; ++c)
        d[c] = (uchar)((s[c] * sa + d[c] * inv) >> 8);
    d[3] = (uchar)(sa + ((da * inv) >> 8));
}
