import torch
import time
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

def build_benchmark_harness():
    print("--- 🛠️ Compiling CUDA Kernel for Unified Benchmark ---")
    
    cpp_source = "torch::Tensor sequence_sum_1d(torch::Tensor packed_tokens, torch::Tensor cu_seqlens, int num_sequences);"
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void sequence_sum_kernel(const float* packed_tokens, const int* cu_seqlens, float* output) {
        int user_id = blockIdx.x;
        int start_idx = cu_seqlens[user_id];
        int end_idx   = cu_seqlens[user_id + 1];
        
        int tid = threadIdx.x;
        int num_threads = blockDim.x;
        
        __shared__ float shared_sum[256];
        float local_sum = 0.0f;
        
        for (int i = start_idx + tid; i < end_idx; i += num_threads) {
            local_sum += packed_tokens[i];
        }
        
        shared_sum[tid] = local_sum;
        __syncthreads();
        
        for (int stride = 128; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            output[user_id] = shared_sum[0];
        }
    }

    torch::Tensor sequence_sum_1d(torch::Tensor packed_tokens, torch::Tensor cu_seqlens, int num_sequences) {
        auto output = torch::zeros({num_sequences}, torch::kFloat32).to(packed_tokens.device());
        int blocks = num_sequences;
        int threads = 256;
        sequence_sum_kernel<<<blocks, threads>>>(packed_tokens.data_ptr<float>(), cu_seqlens.data_ptr<int>(), output.data_ptr<float>());
        return output;
    }
    """

    custom_module = load_inline(
        name='cu_seqlens_bench', cpp_sources=cpp_source, cuda_sources=cuda_source,
        functions=['sequence_sum_1d'], with_cuda=True, extra_cflags=['-O3'], extra_cuda_cflags=['-O3']
    )
    print("✅ Compilation Successful!\n")

    # =====================================================================
    # BENCHMARK CONFIGURATION
    # =====================================================================
    batch_sizes = [16, 64, 256, 1024, 4096,8192]
    max_seq_len = 200 # Keeping under 256 for the simple CUDA shared memory reduction
    
    times_naive, times_native, times_cuda = [], [], []
    vram_naive, vram_packed = [], []

    print("--- 🚀 RUNNING UNIFIED BENCHMARK ---")
    print(f"{'Batch':<10} | {'Naive 2D (ms)':<15} | {'Native 1D (ms)':<15} | {'CUDA 1D (ms)':<15}")
    print("-" * 60)

    for B in batch_sizes:
        # Generate mixed-length sequences
        lengths = torch.randint(low=10, high=max_seq_len, size=(B,)).tolist()
        sequences = [torch.rand(l, dtype=torch.float32, device='cuda') for l in lengths]
        
        # [A] Prepare Naive 2D Padded Data
        padded_2d = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
        
        # [B] Prepare Native 1D Packed Data (Requires seq_ids array for scatter_add)
        packed_1d = torch.cat(sequences, dim=0)
        seq_ids = torch.cat([torch.full((l,), i, dtype=torch.int64, device='cuda') for i, l in enumerate(lengths)])
        
        # [C] Prepare CUDA 1D Packed Data (Requires cu_seqlens array)
        cu_seqlens_long = torch.cumsum(torch.tensor([0] + lengths, dtype=torch.int32, device='cuda'), dim=0)
        cu_seqlens = cu_seqlens_long.to(torch.int32)
        
        # Warmup
        _ = padded_2d.sum(dim=1)
        _ = torch.zeros(B, dtype=torch.float32, device='cuda').scatter_add_(0, seq_ids, packed_1d)
        _ = custom_module.sequence_sum_1d(packed_1d, cu_seqlens, B)
        torch.cuda.synchronize()

        # Track VRAM Usage
        vram_naive.append(padded_2d.element_size() * padded_2d.numel() / (1024**2))
        vram_packed.append((packed_1d.element_size() * packed_1d.numel()) / (1024**2))

        # 1. Benchmark Naive 2D
        start = time.time()
        out_naive = padded_2d.sum(dim=1)
        torch.cuda.synchronize()
        t_naive = (time.time() - start) * 1000

        # 2. Benchmark Native 1D
        start = time.time()
        out_native = torch.zeros(B, dtype=torch.float32, device='cuda').scatter_add_(0, seq_ids, packed_1d)
        torch.cuda.synchronize()
        t_native = (time.time() - start) * 1000

        # 3. Benchmark Custom CUDA 1D
        start = time.time()
        out_cuda = custom_module.sequence_sum_1d(packed_1d, cu_seqlens, B)
        torch.cuda.synchronize()
        t_cuda = (time.time() - start) * 1000

        times_naive.append(t_naive)
        times_native.append(t_native)
        times_cuda.append(t_cuda)
        
        print(f"{B:<10} | {t_naive:<15.3f} | {t_native:<15.3f} | {t_cuda:<15.3f}")

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    print("\n--- 📊 GENERATING PLOTS ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Execution Time
    ax1.plot(batch_sizes, times_naive, marker='o', color='red', label='Naive 2D Padded (Wasteful)')
    ax1.plot(batch_sizes, times_native, marker='s', color='blue', label='Native 1D Packed (scatter_add)')
    ax1.plot(batch_sizes, times_cuda, marker='^', color='green', label='Custom CUDA 1D (cu_seqlens)')
    ax1.set_title("Execution Time vs Batch Size", fontweight='bold')
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_xscale('log')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot 2: VRAM Allocation
    ax2.bar([str(b) for b in batch_sizes], vram_naive, width=0.4, label='2D Padded VRAM', color='red', align='edge')
    ax2.bar([str(b) for b in batch_sizes], vram_packed, width=-0.4, label='1D Packed VRAM', color='green', align='edge')
    ax2.set_title("VRAM Allocation (Ghost Compute)", fontweight='bold')
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("VRAM (MB)")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("unified_packing_benchmark.png", dpi=300)
    print("✅ Benchmark complete! Saved to 'unified_packing_benchmark.png'")

if __name__ == "__main__":
    build_benchmark_harness()

"""

"""