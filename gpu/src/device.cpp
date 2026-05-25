/*
 * ══════════════════════════════════════════════════════════════════════════
 *  EFV · GPU Device Detection
 *  Auto-selects CUDA → OpenCL → Metal → CPU depending on what is compiled
 *  in and available at runtime.
 * ══════════════════════════════════════════════════════════════════════════
 */

#include "../include/efv_gpu.h"
#include <cstdio>
#include <cstring>

/* ── Forward declarations from backend translation units ─────────────── */

#ifdef EFV_CUDA
    extern int  efv_cuda_init(void);
    extern void efv_cuda_shutdown(void);
#endif

#ifdef EFV_OPENCL
    extern int  efv_opencl_init(void);
    extern void efv_opencl_shutdown(void);
#endif

#ifdef EFV_METAL
    extern int  efv_metal_init(void);
    extern void efv_metal_shutdown(void);
#endif

/* ── Global state ────────────────────────────────────────────────────── */

static EfvBackend g_backend    = EFV_BACKEND_CPU;
static int        g_initialised = 0;

static const char* BACKEND_NAMES[] = {
    "CPU",
    "CUDA",
    "OpenCL",
    "Metal",
};

/* ── Public API ──────────────────────────────────────────────────────── */

EfvBackend efv_gpu_init(void) {
    if (g_initialised) return g_backend;

    g_backend = EFV_BACKEND_CPU;   /* safe default */

#ifdef EFV_CUDA
    if (efv_cuda_init() == 0) {
        g_backend    = EFV_BACKEND_CUDA;
        g_initialised = 1;
        fprintf(stderr, "[EFV] GPU backend: CUDA\n");
        return g_backend;
    }
#endif

#ifdef EFV_OPENCL
    if (efv_opencl_init() == 0) {
        g_backend    = EFV_BACKEND_OPENCL;
        g_initialised = 1;
        fprintf(stderr, "[EFV] GPU backend: OpenCL\n");
        return g_backend;
    }
#endif

#ifdef EFV_METAL
    if (efv_metal_init() == 0) {
        g_backend    = EFV_BACKEND_METAL;
        g_initialised = 1;
        fprintf(stderr, "[EFV] GPU backend: Metal\n");
        return g_backend;
    }
#endif

    g_initialised = 1;
    fprintf(stderr, "[EFV] GPU backend: CPU (SIMD)\n");
    return g_backend;
}

void efv_gpu_shutdown(void) {
    if (!g_initialised) return;

#ifdef EFV_CUDA
    if (g_backend == EFV_BACKEND_CUDA) { efv_cuda_shutdown(); }
#endif
#ifdef EFV_OPENCL
    if (g_backend == EFV_BACKEND_OPENCL) { efv_opencl_shutdown(); }
#endif
#ifdef EFV_METAL
    if (g_backend == EFV_BACKEND_METAL) { efv_metal_shutdown(); }
#endif

    g_initialised = 0;
    g_backend     = EFV_BACKEND_CPU;
}

const char* efv_gpu_backend_name(void) {
    return BACKEND_NAMES[(int)g_backend];
}

int efv_gpu_available(void) {
    return g_backend != EFV_BACKEND_CPU;
}

EfvBackend efv_gpu_current_backend(void) {
    return g_backend;
}
