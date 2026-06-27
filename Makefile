NVCC    ?= nvcc

NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_75
BUILD   := build

KERNELS := 01_vector_add 02_matmul 03_softmax 04_layernorm 05_fused_attention

.PHONY: all clean $(KERNELS)

all: $(KERNELS)

$(KERNELS): %:
	@mkdir -p $(BUILD)
	@src=$$(ls $@/*.cu 2>/dev/null | head -n1); \
	if [ -z "$$src" ]; then echo "skip $@ (no .cu yet)"; else \
	  echo "nvcc $$src -> $(BUILD)/$@"; \
	  $(NVCC) $(NVCCFLAGS) $$src -o $(BUILD)/$@; fi

clean:
	rm -rf $(BUILD)
