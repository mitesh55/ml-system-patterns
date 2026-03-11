import torch
import time
from torch.utils.cpp_extension import load_inline

def build_and_run_coalescing():
    print("--- 🛠️ Compiling CUDA Kernels for Cache Miss Test ---")
    print("Invoking NVCC Compiler (Wait ~10 seconds)...\n")

    # =====================================================================
    # 1. THE C++ WRAPPER
    # =====================================================================
    cpp_source = """
    torch::Tensor contiguous_copy(torch::Tensor input);
    torch::Tensor strided_copy(torch::Tensor input, int rows, int cols);
    """

    # =====================================================================
    # 2. THE CUDA KERNELS (The Silicon)
    # =====================================================================
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // --- KERNEL A: Perfect Memory Coalescing ---
    // Threads read and write adjacent memory addresses.
    __global__ void contiguous_kernel(const float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // Read 1, Write 1 directly next door. The Cache Line is fully utilized.
            output[idx] = input[idx]; 
        }
    }

    // --- KERNEL B: Uncoalesced Writes (The Contiguous Trap) ---
    // Simulates what happens when you process a Transposed (strided) matrix
    __global__ void strided_kernel(const float* input, float* output, int rows, int cols) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols) {
            int r = idx / cols; // Current Row
            int c = idx % cols; // Current Col
            
            // Threads read adjacently, but WRITE with a massive jump (by 'rows').
            // This shatters the Cache Line, forcing the GPU to make thousands of extra VRAM trips.
            output[c * rows + r] = input[r * cols + c];
        }
    }

    // --- C++ HOST FUNCTIONS ---
    torch::Tensor contiguous_copy(torch::Tensor input) {
        auto output = torch::empty_like(input);
        int threads = 256;
        int blocks = (input.numel() + threads - 1) / threads;
        
        contiguous_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), input.numel()
        );
        return output;
    }

    torch::Tensor strided_copy(torch::Tensor input, int rows, int cols) {
        auto output = torch::empty_like(input);
        int threads = 256;
        int blocks = (input.numel() + threads - 1) / threads;
        
        strided_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
        );
        return output;
    }
    """

    custom_module = load_inline(
        name='memory_coalescing_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['contiguous_copy', 'strided_copy'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3']
    )

    print("✅ Compilation Successful!\n")

    # =====================================================================
    # 3. THE HARDWARE STRESS TEST
    # =====================================================================
    ROWS, COLS = 10000, 10000
    N = ROWS * COLS
    print(f"--- 🚨 INITIATING CACHE MISS TEST (Matrix: {ROWS}x{COLS} = {N:,} elements) ---")
    
    # 400 MB Tensor
    matrix = torch.rand(ROWS, COLS, dtype=torch.float32, device='cuda')

    # Warmup
    _ = custom_module.contiguous_copy(matrix)
    _ = custom_module.strided_copy(matrix, ROWS, COLS)
    torch.cuda.synchronize()

    # [A] Run Contiguous Kernel
    start = time.time()
    for _ in range(10): # Loop 10x to exaggerate the time difference
        _ = custom_module.contiguous_copy(matrix)
    torch.cuda.synchronize()
    time_contiguous = (time.time() - start) / 10

    # [B] Run Strided Kernel
    start = time.time()
    for _ in range(10):
        _ = custom_module.strided_copy(matrix, ROWS, COLS)
    torch.cuda.synchronize()
    time_strided = (time.time() - start) / 10

    # =====================================================================
    # 4. THE SHOCKING TRUTH (Terminal Output)
    # =====================================================================
    slowdown = time_strided / time_contiguous

    print("\n--- 📊 RESULTS (Average of 10 runs) ---")
    print(f"[A] Contiguous Memory Access : {time_contiguous:.5f}s")
    print(f"[B] Strided Memory Access    : {time_strided:.5f}s")
    
    print(f"\n⚠️ HARDWARE BOTTLENECK ALERT ⚠️")
    print(f"The exact same math operation ran {slowdown:.2f}x slower simply because the memory was out of order.")
    
    print("\n❓ QUESTION: Why does 'jumping' across memory cause such a massive slowdown?")
    print("💡 INTUITION: Memory Coalescing and Cache Lines")
    print("   The GPU VRAM does not fetch numbers one-by-one. It fetches 128-byte 'Cache Lines'.")
    print("   -> In Kernel A, 32 threads requested 32 adjacent floats. The GPU fetched them all in exactly 1 transaction.")
    print("   -> In Kernel B, 32 threads requested floats that were 10,000 indices apart. The GPU had to make 32 separate trips to VRAM.")
    print("   You essentially forced the GPU memory controller to do 32x more work for the exact same data.")

if __name__ == "__main__":
    build_and_run_coalescing()