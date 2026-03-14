import torch
import time
from torch.utils.cpp_extension import load_inline

def build_and_run_fused_bn():
    print("--- 🛠️ Compiling CUDA Kernel for Fused Batch Normalization ---")
    print("Invoking NVCC Compiler (Wait ~10 seconds)...\n")

    cpp_source = """
    torch::Tensor fused_batch_norm(torch::Tensor x, torch::Tensor mu, torch::Tensor var, 
                                   torch::Tensor gamma, torch::Tensor beta, float eps);
    """

    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <math.h>

    // --- KERNEL: Fused Element-wise Normalization ---
    __global__ void fused_bn_kernel(const float* x, const float* mu, const float* var,
                                    const float* gamma, const float* beta,
                                    float* out, float eps, int N, int D) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < N * D) {
            // Figure out which feature column this thread belongs to
            int d = idx % D; 
            
            // THE HARDWARE MAGIC: All 4 math operations happen in the SM Registers!
            // rsqrtf is a hardware-level intrinsic for (1.0 / sqrt(x))
            float std_inv = rsqrtf(var[d] + eps);
            float x_norm = (x[idx] - mu[d]) * std_inv;
            
            // We write to VRAM exactly ONCE.
            out[idx] = x_norm * gamma[d] + beta[d];
        }
    }

    // --- C++ HOST FUNCTION ---
    torch::Tensor fused_batch_norm(torch::Tensor x, torch::Tensor mu, torch::Tensor var, 
                                   torch::Tensor gamma, torch::Tensor beta, float eps) {
        
        int N = x.size(0);
        int D = x.size(1);
        auto out = torch::empty_like(x);
        
        int threads = 256;
        int blocks = (x.numel() + threads - 1) / threads;
        
        fused_bn_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), mu.data_ptr<float>(), var.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            out.data_ptr<float>(), eps, N, D
        );
        
        return out;
    }
    """

    custom_module = load_inline(
        name='fused_bn_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['fused_batch_norm'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '-use_fast_math']
    )

    print("✅ Compilation Successful!\n")

    # =====================================================================
    # 3. BENCHMARKING THE MEMORY TRAP (Apples-to-Apples)
    # =====================================================================
    N, D = 100_000, 1024 # 100k instances, 1024 features (~400 MB Tensor)
    print(f"--- 🚨 INITIATING BATCH NORM BENCHMARK (Batch: {N}, Features: {D}) ---")
    
    X = torch.rand(N, D, dtype=torch.float32, device='cuda')
    gamma = torch.ones(D, device='cuda')
    beta = torch.zeros(D, device='cuda')
    eps = 1e-5

    # Setup Native PyTorch Layer
    bn_layer = torch.nn.BatchNorm1d(num_features=D, eps=eps, affine=True).cuda()
    bn_layer.weight.data = gamma
    bn_layer.bias.data = beta
    bn_layer.train() # CRITICAL: Force it to compute batch stats

    # Warmup
    _ = X.mean(dim=0)
    _ = custom_module.fused_batch_norm(X, X.mean(dim=0), X.var(dim=0, unbiased=False), gamma, beta, eps)
    _ = bn_layer(X)
    torch.cuda.synchronize()

    # ---------------------------------------------------------------------
    # [A] Naive PyTorch API (Separate Stats + 4 Math VRAM Trips)
    start = time.time()
    mu_naive = X.mean(dim=0)
    var_naive = X.var(dim=0, unbiased=False)
    out_naive = gamma * ((X - mu_naive) / torch.sqrt(var_naive + eps)) + beta
    torch.cuda.synchronize()
    time_naive = time.time() - start

    # ---------------------------------------------------------------------
    # [B] Custom CUDA Fused Kernel (Separate Stats + 1 Math VRAM Trip)
    start = time.time()
    mu_cuda = X.mean(dim=0)
    var_cuda = X.var(dim=0, unbiased=False)
    out_cuda = custom_module.fused_batch_norm(X, mu_cuda, var_cuda, gamma, beta, eps)
    torch.cuda.synchronize()
    time_cuda = time.time() - start

    # ---------------------------------------------------------------------
    # [C] Native PyTorch cuDNN (Fully Fused Stats & Math)
    start = time.time()
    out_native = bn_layer(X)
    torch.cuda.synchronize()
    time_native = time.time() - start

    print("--- 📊 RESULTS ---")
    print(f"[A] Naive PyTorch Math : {time_naive:.5f}s (Allocates massive intermediate tensors)")
    print(f"[B] Semi-Fused CUDA    : {time_cuda:.5f}s (Our Kernel saves 1.6 GB RAM)")
    print(f"[C] Native cuDNN Layer : {time_native:.5f}s (NVIDIA's ultimate implementation)")
    
    speedup_b = time_naive / time_cuda
    speedup_c = time_naive / time_native
    print(f"\nSpeedup (Custom vs Naive): {speedup_b:.2f}x")
    print(f"Speedup (cuDNN vs Naive) : {speedup_c:.2f}x")

    # Verify math against the true cuDNN baseline
    max_diff = torch.max(torch.abs(out_native - out_cuda)).item()
    if max_diff < 1e-4:
        print(f"\n✅ Math Verification: SUCCESS. Custom CUDA matches cuDNN (Max Diff: {max_diff:.6e})")

    # =====================================================================
    # 4. THE HARDWARE INTUITION (Terminal Output)
    # =====================================================================
    print("\n--- 🧠 MENTOR NOTES: The Hierarchy of Fusion ---")
    print("❓ QUESTION: Why is NVIDIA's cuDNN still faster than our custom CUDA kernel?")
    print("💡 INTUITION: We only fused the 'Math', but cuDNN fused the 'Statistics'.")
    print("   Look at Method B: mu_cuda = X.mean() ... var_cuda = X.var() ... fused_kernel(...)")
    print("   1. X.mean() forces the GPU to read the 400MB tensor from VRAM to find the average.")
    print("   2. X.var() forces the GPU to read the 400MB tensor AGAIN to find the variance.")
    print("   3. fused_kernel() reads the 400MB tensor a THIRD time to apply the math.")
    print("\n   NVIDIA's cuDNN (Method C) utilizes 'Warp-Level Reductions'.")
    print("   It reads the 400MB tensor ONCE. While the data is sitting inside the ultra-fast L1 cache,")
    print("   it calculates the mean, computes the variance, and applies the normalization simultaneously.")
    print("   To beat cuDNN, you must master Shared Memory and Warp Primitives.")
if __name__ == "__main__":
    build_and_run_fused_bn()