/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · CUDA Resize Kernels
 *  Compiled with: nvcc -arch=sm_60 -O3 -DEFV_CUDA
 * ══════════════════════════════════════════════════════════════════════════
 */
#ifdef EFV_CUDA

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* ── Bilinear kernel ─────────────────────────────────────────────────── */

__global__ void kernel_resize_bilinear(
    const uint8_t* __restrict__ src,
    uint8_t*       __restrict__ dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h)
{
    const uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    const float x_ratio = (float)src_w / (float)dst_w;
    const float y_ratio = (float)src_h / (float)dst_h;

    const float sx_f = (dx + 0.5f) * x_ratio - 0.5f;
    const float sy_f = (dy + 0.5f) * y_ratio - 0.5f;

    const int sx0 = max(0, min((int)sx_f,        (int)src_w - 1));
    const int sx1 = max(0, min((int)sx_f + 1,    (int)src_w - 1));
    const int sy0 = max(0, min((int)sy_f,        (int)src_h - 1));
    const int sy1 = max(0, min((int)sy_f + 1,    (int)src_h - 1));

    const float wx1 = sx_f - floorf(sx_f);
    const float wx0 = 1.0f - wx1;
    const float wy1 = sy_f - floorf(sy_f);
    const float wy0 = 1.0f - wy1;

    const uint8_t* p00 = src + (sy0 * src_w + sx0) * 4;
    const uint8_t* p10 = src + (sy0 * src_w + sx1) * 4;
    const uint8_t* p01 = src + (sy1 * src_w + sx0) * 4;
    const uint8_t* p11 = src + (sy1 * src_w + sx1) * 4;

    uint8_t* out = dst + (dy * dst_w + dx) * 4;
    for (int c = 0; c < 4; ++c) {
        float v = p00[c] * wx0 * wy0
                + p10[c] * wx1 * wy0
                + p01[c] * wx0 * wy1
                + p11[c] * wx1 * wy1;
        out[c] = (uint8_t)fminf(255.0f, fmaxf(0.0f, v + 0.5f));
    }
}

/* ── Nearest kernel ──────────────────────────────────────────────────── */

__global__ void kernel_resize_nearest(
    const uint8_t* __restrict__ src,
    uint8_t*       __restrict__ dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h)
{
    const uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    const uint32_t sx = min((uint32_t)((float)dx / dst_w * src_w), src_w - 1);
    const uint32_t sy = min((uint32_t)((float)dy / dst_h * src_h), src_h - 1);

    const uint8_t* p   = src + (sy * src_w + sx) * 4;
    uint8_t*       out = dst + (dy * dst_w + dx) * 4;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}

/* ── CUDA device memory helpers ──────────────────────────────────────── */

static cudaError_t alloc_and_copy(const uint8_t* host, uint8_t** dev, size_t bytes) {
    cudaError_t err = cudaMalloc((void**)dev, bytes);
    if (err != cudaSuccess) return err;
    return cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
}

/* ── Public C API ────────────────────────────────────────────────────── */

extern "C" int efv_cuda_init(void) {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) return -1;
    return cudaSetDevice(0) == cudaSuccess ? 0 : -1;
}

extern "C" void efv_cuda_shutdown(void) {
    cudaDeviceReset();
}

extern "C" int efv_cuda_resize_bilinear(
    const uint8_t* src, uint8_t* dst,
    uint32_t sw, uint32_t sh,
    uint32_t dw, uint32_t dh)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    size_t   src_bytes = (size_t)sw * sh * 4;
    size_t   dst_bytes = (size_t)dw * dh * 4;

    if (alloc_and_copy(src, &d_src, src_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_dst, dst_bytes)  != cudaSuccess) goto fail;

    {
        dim3 block(16, 16);
        dim3 grid((dw + 15) / 16, (dh + 15) / 16);
        kernel_resize_bilinear<<<grid, block>>>(d_src, d_dst, sw, sh, dw, dh);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;

fail:
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    return -1;
}

extern "C" int efv_cuda_resize_nearest(
    const uint8_t* src, uint8_t* dst,
    uint32_t sw, uint32_t sh,
    uint32_t dw, uint32_t dh)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    size_t   src_bytes = (size_t)sw * sh * 4;
    size_t   dst_bytes = (size_t)dw * dh * 4;

    if (alloc_and_copy(src, &d_src, src_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_dst, dst_bytes)  != cudaSuccess) goto fail;

    {
        dim3 block(16, 16);
        dim3 grid((dw + 15) / 16, (dh + 15) / 16);
        kernel_resize_nearest<<<grid, block>>>(d_src, d_dst, sw, sh, dw, dh);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;

fail:
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    return -1;
}

#endif /* EFV_CUDA */
