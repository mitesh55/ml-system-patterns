import torch
import time
from torch.utils.cpp_extension import load_inline

def build_and_run_cuda():
    print("--- 🛠️ Compiling Custom CUDA Kernel (JIT) ---")
    print("Please wait ~10-20 seconds for the NVCC compiler...\n")

    # =====================================================================
    # 1. THE C++ WRAPPER (The Bridge)
    # =====================================================================
    # This tells Python what the C++ function signature looks like
    cpp_source = """
    torch::Tensor fused_relu_scale(torch::Tensor input, float scale);
    """

    # =====================================================================
    # 2. THE CUDA KERNEL (The Silicon)
    # =====================================================================
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // ---------------------------------------------------------
    // THE GPU THREAD LOGIC (__global__ means it runs on the GPU)
    // ---------------------------------------------------------
    __global__ void fused_relu_scale_kernel(const float* input, float* output, float scale, int size) {
        // 1. Identify which thread this is (Mapping hardware to the array index)
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // 2. Ensure we don't read past the end of the array
        if (idx < size) {
            float val = input[idx];
            
            // 3. The Condition (ReLU) + The Math (Scale)
            // This happens entirely inside the GPU's ultra-fast register memory
            if (val < 0.0) {
                output[idx] = 0.0;
            } else {
                output[idx] = val * scale;
            }
        }
    }

    // ---------------------------------------------------------
    // THE HOST FUNCTION (Runs on CPU, configures the GPU)
    // ---------------------------------------------------------
    torch::Tensor fused_relu_scale(torch::Tensor input, float scale) {
        // Allocate a contiguous memory block for the output
        auto output = torch::empty_like(input);
        
        // Configuration: Group threads into blocks of 256
        const int threads_per_block = 256;
        
        // Calculate how many blocks we need to cover the entire array
        const int blocks = (input.numel() + threads_per_block - 1) / threads_per_block;

        // Launch the kernel! (The <<<blocks, threads>>> syntax is specific to CUDA)
        fused_relu_scale_kernel<<<blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            scale,
            input.numel()
        );

        return output;
    }
    """

    # Compile the extension dynamically
    custom_module = load_inline(
        name='fused_relu_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['fused_relu_scale'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3']
    )

    print("✅ Compilation Successful! Testing execution...\n")

    # =====================================================================
    # 3. THE BENCHMARK
    # =====================================================================
    N = 50_000_000
    # Generate 50 Million random numbers between -1.0 and 1.0, move to GPU
    x = (torch.rand(N, dtype=torch.float32, device='cuda') * 2.0) - 1.0
    scale_factor = 2.0

    print(f"Dataset: {N:,} elements on {torch.cuda.get_device_name(0)}")

    # -- Method A: Standard PyTorch (2 VRAM Trips) --
    # Warmup
    _ = torch.relu(x) * scale_factor
    torch.cuda.synchronize()
    
    start = time.time()
    out_pytorch = torch.relu(x) * scale_factor
    torch.cuda.synchronize()
    time_pytorch = time.time() - start

    # -- Method B: Custom CUDA Fused Kernel (1 VRAM Trip) --
    # Warmup
    _ = custom_module.fused_relu_scale(x, scale_factor)
    torch.cuda.synchronize()

    start = time.time()
    out_cuda = custom_module.fused_relu_scale(x, scale_factor)
    torch.cuda.synchronize()
    time_cuda = time.time() - start

    print("\n--- Benchmark Results ---")
    print(f"[A] PyTorch (Unfused): {time_pytorch:.6f}s")
    print(f"[B] Custom CUDA Kernel:{time_cuda:.6f}s")
    
    speedup = time_pytorch / time_cuda
    print(f"Speedup: {speedup:.2f}x")

    # Verify accuracy
    max_diff = torch.max(torch.abs(out_pytorch - out_cuda)).item()
    if max_diff < 1e-5:
        print("\n✅ Math Verification: SUCCESS. CUDA matches PyTorch perfectly.")

if __name__ == "__main__":
    build_and_run_cuda()