# Build CUDA kernels with nvcc. Run on an NVIDIA GPU (CUDA toolkit required).
#
#   make 01_vector_add   # build one kernel  -> build/01_vector_add
#   make all             # build everything
#   make clean

NVCC    ?= nvcc
# -O3: optimize. -arch=sm_70 covers V100/T4-era; bump for newer GPUs
# (sm_80 = A100, sm_86 = RTX 30xx, sm_89 = L4/4090, sm_90 = H100).
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_70
BUILD   := build

KERNELS := 01_vector_add 02_matmul 03_softmax 04_layernorm 05_fused_attention

.PHONY: all clean $(KERNELS)

all: $(KERNELS)

# Pattern: each target NN_name builds NN_name/<name>.cu -> build/NN_name
$(KERNELS): %:
	@mkdir -p $(BUILD)
	@src=$$(ls $@/*.cu 2>/dev/null | head -n1); \
	if [ -z "$$src" ]; then echo "skip $@ (no .cu yet)"; else \
	  echo "nvcc $$src -> $(BUILD)/$@"; \
	  $(NVCC) $(NVCCFLAGS) $$src -o $(BUILD)/$@; fi

clean:
	rm -rf $(BUILD)
