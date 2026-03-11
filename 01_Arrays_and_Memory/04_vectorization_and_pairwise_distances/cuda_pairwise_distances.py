import torch
import time
from torch.utils.cpp_extension import load_inline

def build_and_run_pairwise_cuda():
    print("--- 🛠️ Compiling CUDA Kernel for Fused Distance Matrix ---")
    print("Invoking NVCC Compiler (Wait ~10 seconds)...\n")

    # =====================================================================
    # 1. THE C++ WRAPPER
    # =====================================================================
    cpp_source = """
    torch::Tensor fused_pairwise_distance(torch::Tensor X, torch::Tensor Y);
    """

    # =====================================================================
    # 2. THE CUDA KERNEL (The Silicon)
    # =====================================================================
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <math.h>

    // --- KERNEL: Fused Register Accumulation ---
    // We assign exactly ONE GPU thread to compute the distance between ONE pair of vectors (i, j).
    __global__ void pairwise_distance_kernel(const float* X, const float* Y, float* output, int N, int M, int D) {
        
        // 1. 2D Thread Mapping: Find exactly which (N, M) coordinate this thread owns
        int row = blockIdx.x * blockDim.x + threadIdx.x; // Vector from X
        int col = blockIdx.y * blockDim.y + threadIdx.y; // Vector from Y

        if (row < N && col < M) {
            
            // 2. THE HARDWARE MAGIC: The Register Variable
            // 'dist_sq' lives entirely inside the ultra-fast physical registers of the Streaming Multiprocessor.
            // It NEVER touches the 6GB VRAM on your RTX board.
            float dist_sq = 0.0f;
            
            // 3. The loop runs inside the silicon. 
            // It accumulates the 128 dimensions without expanding memory.
            for (int d = 0; d < D; ++d) {
                float diff = X[row * D + d] - Y[col * D + d];
                dist_sq += diff * diff;
            }
            
            // 4. We only write the final scalar to VRAM once.
            output[row * M + col] = sqrtf(dist_sq);
        }
    }

    // --- C++ HOST FUNCTION ---
    torch::Tensor fused_pairwise_distance(torch::Tensor X, torch::Tensor Y) {
        int N = X.size(0);
        int M = Y.size(0);
        int D = X.size(1);
        
        auto output = torch::empty({N, M}, torch::kFloat32).to(X.device());
        
        // Configuration: Create a 2D Grid of threads (16x16 blocks)
        dim3 threads_per_block(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);
        
        pairwise_distance_kernel<<<blocks, threads_per_block>>>(
            X.data_ptr<float>(), Y.data_ptr<float>(), output.data_ptr<float>(), N, M, D
        );
        
        return output;
    }
    """

    custom_module = load_inline(
        name='fused_distance_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['fused_pairwise_distance'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '-use_fast_math']
    )

    print("✅ Compilation Successful!\n")

    # =====================================================================
    # 3. THE HARDWARE STRESS TEST
    # =====================================================================
    N, M, D = 5000, 5000, 128
    print(f"--- 🚨 INITIATING PAIRWISE DISTANCE (X: {N}x{D}, Y: {M}x{D}) ---")
    
    # Calculate Broadcasting Memory Trap theoretically
    mem_trap = (N * M * D * 4) / (1024**3)
    mem_safe = (N * M * 4) / (1024**3)
    
    print(f"⚠️ Broadcasting Trap requires: {mem_trap:.2f} GB of RAM (Will OOM).")
    print(f"✅ Fused Kernel allocates:     {mem_safe:.3f} GB of RAM.\n")

    X = torch.rand(N, D, dtype=torch.float32, device='cuda')
    Y = torch.rand(M, D, dtype=torch.float32, device='cuda')

    # Warmup
    _ = torch.cdist(X, Y)
    _ = custom_module.fused_pairwise_distance(X, Y)
    torch.cuda.synchronize()

    # [A] Native PyTorch cdist
    start = time.time()
    out_pt = torch.cdist(X, Y)
    torch.cuda.synchronize()
    time_pt = time.time() - start

    # [B] Custom CUDA Fused Kernel
    start = time.time()
    out_cuda = custom_module.fused_pairwise_distance(X, Y)
    torch.cuda.synchronize()
    time_cuda = time.time() - start

    print("--- 📊 RESULTS ---")
    print(f"[A] Native torch.cdist     : {time_pt:.5f}s (Highly optimized cuBLAS GEMM)")
    print(f"[B] Custom Register Kernel : {time_cuda:.5f}s (Our C++ implementation)")
    
    max_diff = torch.max(torch.abs(out_pt - out_cuda)).item()
    if max_diff < 1e-4:
        print(f"\n✅ Math Verification: SUCCESS (Max Diff: {max_diff:.6e})")

if __name__ == "__main__":
    build_and_run_pairwise_cuda()