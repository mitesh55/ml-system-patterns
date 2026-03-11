import torch
import time
from torch.utils.cpp_extension import load_inline

def build_and_run_race_condition():
    print("--- 🛠️ Compiling CUDA Kernels for Race Condition Test ---")
    print("Invoking NVCC Compiler (Wait ~10 seconds)...\n")

    # =====================================================================
    # 1. THE C++ WRAPPER
    # =====================================================================
    cpp_source = """
    torch::Tensor naive_scatter(torch::Tensor tokens, int vocab_size);
    torch::Tensor atomic_scatter(torch::Tensor tokens, int vocab_size);
    """

    # =====================================================================
    # 2. THE CUDA KERNELS (The Silicon)
    # =====================================================================
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // --- KERNEL A: The Race Condition (Unsafe Write) ---
    __global__ void naive_scatter_kernel(const int* tokens, int* target, int num_tokens) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_tokens) {
            int vocab_idx = tokens[idx];
            
            // DANGER: Read -> Add -> Write. 
            // If 10,000 threads do this simultaneously, they overwrite each other!
            target[vocab_idx] = target[vocab_idx] + 1;
        }
    }

    // --- KERNEL B: The Hardware Lock (Atomic Write) ---
    __global__ void atomic_scatter_kernel(const int* tokens, int* target, int num_tokens) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_tokens) {
            int vocab_idx = tokens[idx];
            
            // SAFE: atomicAdd locks the specific memory address at the silicon level,
            // does the addition, and unlocks it. 
            atomicAdd(&target[vocab_idx], 1);
        }
    }

    // --- C++ HOST FUNCTIONS ---
    torch::Tensor naive_scatter(torch::Tensor tokens, int vocab_size) {
        auto target = torch::zeros({vocab_size}, torch::kInt32).to(tokens.device());
        int threads = 256;
        int blocks = (tokens.numel() + threads - 1) / threads;
        
        naive_scatter_kernel<<<blocks, threads>>>(
            tokens.data_ptr<int>(), target.data_ptr<int>(), tokens.numel()
        );
        return target;
    }

    torch::Tensor atomic_scatter(torch::Tensor tokens, int vocab_size) {
        auto target = torch::zeros({vocab_size}, torch::kInt32).to(tokens.device());
        int threads = 256;
        int blocks = (tokens.numel() + threads - 1) / threads;
        
        atomic_scatter_kernel<<<blocks, threads>>>(
            tokens.data_ptr<int>(), target.data_ptr<int>(), tokens.numel()
        );
        return target;
    }
    """

    custom_module = load_inline(
        name='atomic_scatter_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['naive_scatter', 'atomic_scatter'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3']
    )

    print("✅ Compilation Successful!\n")

    # =====================================================================
    # 3. THE HARDWARE STRESS TEST
    # =====================================================================
    N = 10_000_000
    Vocab_Size = 10
    
    print(f"--- 🚨 INITIATING RACE CONDITION (N={N:,} threads) ---")
    # We deliberately create 10 Million tokens that are ALL the number '0'.
    # This forces every single GPU thread to attack index 0 of the target array simultaneously.
    tokens = torch.zeros(N, dtype=torch.int32, device='cuda')
    
    print("Goal: Scatter-add '1' for every token. Expected Value at Index 0: 10,000,000\n")

    # [A] Run Naive Kernel
    start = time.time()
    out_naive = custom_module.naive_scatter(tokens, Vocab_Size)
    torch.cuda.synchronize()
    time_naive = time.time() - start
    naive_result = out_naive[0].item()

    # [B] Run Atomic Kernel
    start = time.time()
    out_atomic = custom_module.atomic_scatter(tokens, Vocab_Size)
    torch.cuda.synchronize()
    time_atomic = time.time() - start
    atomic_result = out_atomic[0].item()

    # =====================================================================
    # 4. THE SHOCKING TRUTH (Terminal Output)
    # =====================================================================
    data_loss = (N - naive_result) / N * 100

    print("--- 📊 RESULTS ---")
    print(f"[A] Naive C++ (No Locks) : {naive_result:,}  (Time: {time_naive:.4f}s)")
    print(f"[B] Atomic C++ (Locked)  : {atomic_result:,} (Time: {time_atomic:.4f}s)")
    
    print(f"\n⚠️ DATA CORRUPTION ALERT ⚠️")
    print(f"The naive kernel lost {data_loss:.2f}% of the data!")
    print("\n❓ QUESTION: Where did the millions of missing data points go?")
    print("💡 INTUITION: The 'Read-Modify-Write' Collision")
    print("   Thread 1 reads the value '0' from RAM.")
    print("   Thread 2 reads the value '0' from RAM at the exact same time.")
    print("   Thread 1 calculates 0 + 1 = 1, and writes '1' to RAM.")
    print("   Thread 2 calculates 0 + 1 = 1, and overwrites Thread 1's work with '1'.")
    print("   Millions of additions were entirely erased from existence.")

    print("\n❓ QUESTION: Why is the Atomic version slower?")
    print("💡 INTUITION: Hardware Serialization (The Traffic Jam)")
    print("   `atomicAdd` forces the GPU memory controller to lock the address.")
    print("   If 10 Million threads want the same address, they must wait in line.")
    print("   This prevents corruption, but creates a massive hardware bottleneck.")
    print("   This is why designing ML algorithms to avoid 'hotspots' is critical.")

if __name__ == "__main__":
    build_and_run_race_condition()