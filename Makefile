# ══════════════════════════════════════════════════════════════════════════
#  EFV – Extreme Fast Video  ·  Build System
# ══════════════════════════════════════════════════════════════════════════

.PHONY: all rust gpu install dev clean test bench help

PYTHON   ?= python3
BUILD_DIR = gpu/build

# ── Main targets ──────────────────────────────────────────────────────────

all: rust gpu          ## Build Rust core AND GPU library

rust:                  ## Build only the Rust/PyO3 extension (maturin develop)
	@echo "==> Building Rust core (efv_core)…"
	maturin develop --release

rust-wheel:            ## Build a distributable wheel
	maturin build --release

gpu:                   ## Build the C++ GPU shared library (auto-detects CUDA/OpenCL/Metal)
	@echo "==> Building C++ GPU library (efv_gpu)…"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		$(GPU_FLAGS) \
		-G "$(CMAKE_GENERATOR)" 2>/dev/null || \
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release $(GPU_FLAGS)
	@cmake --build $(BUILD_DIR) --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
	@echo "==> GPU library built: $(BUILD_DIR)/libefv_gpu.so (or .dll / .dylib)"

gpu-cuda:              ## Build GPU library with CUDA support
	$(MAKE) gpu GPU_FLAGS="-DEFV_CUDA=ON"

gpu-opencl:            ## Build GPU library with OpenCL support
	$(MAKE) gpu GPU_FLAGS="-DEFV_OPENCL=ON"

gpu-metal:             ## Build GPU library with Apple Metal support (macOS only)
	$(MAKE) gpu GPU_FLAGS="-DEFV_METAL=ON"

gpu-cpu:               ## Build GPU library with CPU-SIMD fallback only
	$(MAKE) gpu GPU_FLAGS="-DEFV_CUDA=OFF -DEFV_OPENCL=OFF -DEFV_METAL=OFF"

install:               ## Install the Python package (requires maturin)
	$(MAKE) all
	pip install -e ".[pillow]"

dev:                   ## Install in editable mode with dev extras
	pip install maturin
	$(MAKE) rust
	pip install -e ".[pillow,dev]"
	@echo "==> Dev install complete.  Run: make test"

# ── Testing ───────────────────────────────────────────────────────────────

test:                  ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

bench:                 ## Run benchmarks
	$(PYTHON) examples/benchmark.py

# ── Cleanup ───────────────────────────────────────────────────────────────

clean:                 ## Remove build artifacts
	rm -rf $(BUILD_DIR) target/ dist/ *.egg-info efv/*.so efv/*.pyd efv/*.dll efv/*.dylib
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────────

help:                  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  GPU_FLAGS can be passed to override auto-detection:"
	@echo "    make gpu GPU_FLAGS=\"-DEFV_CUDA=ON\""
