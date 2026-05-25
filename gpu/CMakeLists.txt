cmake_minimum_required(VERSION 3.18)
project(efv_gpu VERSION 1.0.0 LANGUAGES CXX)

# ── Options ───────────────────────────────────────────────────────────────

option(EFV_CUDA    "Enable CUDA GPU acceleration"   OFF)
option(EFV_OPENCL  "Enable OpenCL GPU acceleration" OFF)
option(EFV_METAL   "Enable Apple Metal (macOS only)" OFF)

# ── Auto-detect GPU backends if user hasn't chosen ───────────────────────

if(NOT EFV_CUDA AND NOT EFV_OPENCL AND NOT EFV_METAL)
    # CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(EFV_CUDA ON)
        message(STATUS "[EFV] Auto-enabled CUDA backend")
    endif()

    # OpenCL
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND AND NOT EFV_CUDA)
        set(EFV_OPENCL ON)
        message(STATUS "[EFV] Auto-enabled OpenCL backend")
    endif()

    # Metal (macOS)
    if(APPLE AND NOT EFV_CUDA AND NOT EFV_OPENCL)
        set(EFV_METAL ON)
        message(STATUS "[EFV] Auto-enabled Metal backend")
    endif()

    if(NOT EFV_CUDA AND NOT EFV_OPENCL AND NOT EFV_METAL)
        message(STATUS "[EFV] No GPU toolkit found — using CPU SIMD fallback")
    endif()
endif()

# ── CUDA setup ────────────────────────────────────────────────────────────

if(EFV_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ── Source files ──────────────────────────────────────────────────────────

set(GPU_SOURCES
    src/device.cpp
    src/resize.cpp
    src/composite.cpp
)

if(EFV_CUDA)
    list(APPEND GPU_SOURCES src/resize.cu)
endif()

# ── Shared library ────────────────────────────────────────────────────────

add_library(efv_gpu SHARED ${GPU_SOURCES})

target_include_directories(efv_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Compiler flags ────────────────────────────────────────────────────────

target_compile_features(efv_gpu PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(efv_gpu PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(efv_gpu PRIVATE
        -O3 -ffast-math
        $<$<BOOL:${APPLE}>:-march=native>
        $<$<NOT:$<BOOL:${APPLE}>>:-march=native>
    )
endif()

# ── Backend compile definitions ───────────────────────────────────────────

if(EFV_CUDA)
    target_compile_definitions(efv_gpu PRIVATE EFV_CUDA)
    target_link_libraries(efv_gpu PRIVATE CUDA::cudart)
    set_property(TARGET efv_gpu PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
    set_property(TARGET efv_gpu PROPERTY CUDA_STANDARD 17)
endif()

if(EFV_OPENCL)
    find_package(OpenCL REQUIRED)
    target_compile_definitions(efv_gpu PRIVATE EFV_OPENCL)
    target_link_libraries(efv_gpu PRIVATE OpenCL::OpenCL)
    # Copy OpenCL source at build time so it ships next to the .so
    configure_file(opencl/resize.cl ${CMAKE_BINARY_DIR}/efv_resize.cl COPYONLY)
endif()

if(EFV_METAL)
    target_compile_definitions(efv_gpu PRIVATE EFV_METAL)
    target_link_libraries(efv_gpu PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    # Compile Metal library
    add_custom_command(TARGET efv_gpu POST_BUILD
        COMMAND xcrun -sdk macosx metal -O3
            -o "${CMAKE_BINARY_DIR}/efv_kernels.metallib"
            "${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels.metal"
        COMMENT "Compiling Metal shaders"
    )
endif()

# ── Output naming ─────────────────────────────────────────────────────────

set_target_properties(efv_gpu PROPERTIES
    OUTPUT_NAME      "efv_gpu"
    VERSION          ${PROJECT_VERSION}
    SOVERSION        1
    # Place DLL next to Python package on Windows
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../efv"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../efv"
)

# ── Install ───────────────────────────────────────────────────────────────

install(TARGETS efv_gpu
    RUNTIME DESTINATION efv    # Windows DLL
    LIBRARY DESTINATION efv    # Linux/macOS .so/.dylib
)
install(FILES include/efv_gpu.h DESTINATION include)
