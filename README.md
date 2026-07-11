# cuda-kernels

Handwritten CUDA kernels benchmarked against cuBLAS.

From-scratch GPU kernels, each measured against a reference (cuBLAS / PyTorch) so you can see the performance gap. Building up toward a FlashAttention-style fused attention kernel.

Each folder has the kernel, a short README with the math, and benchmark numbers.

## How 2 run

CUDA can't run on a Mac, so develop locally, compile/run on an NVIDIA GPU
Google Colab free T4 is the quickest start.

```bash
# on a machine with the CUDA toolkit installed
make 01_vector_add      # build one kernel
./build/01_vector_add   # run it
make all                # build everything
```

## Google Colab

1. Runtime → Change runtime type → T4 GPU
2. `!nvcc --version` to confirm toolkit
3. `!git clone https://github.com/RazorCrest00/cuda-kernels && cd cuda-kernels && make all`
