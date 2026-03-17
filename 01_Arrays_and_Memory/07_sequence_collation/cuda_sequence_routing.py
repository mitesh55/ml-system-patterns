import torch
from torch.utils.cpp_extension import load_inline

def test_cuda_cu_seqlens():
    print("--- 🛠️ Compiling CUDA Kernel for 1D Sequence Routing ---")
    
    cpp_source = """
    torch::Tensor sequence_sum_1d(torch::Tensor packed_tokens, torch::Tensor cu_seqlens, int num_sequences);
    """

    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // --- KERNEL: The Brick Wall Router ---
    // We assign exactly ONE Thread Block to ONE User Sequence.
    __global__ void sequence_sum_kernel(const float* packed_tokens, const int* cu_seqlens, float* output) {
        
        // 1. Which user am I processing? (blockIdx.x is the User ID)
        int user_id = blockIdx.x;
        
        // 2. Read the "Brick Walls" (cu_seqlens)
        int start_idx = cu_seqlens[user_id];     // Where does my data start?
        int end_idx   = cu_seqlens[user_id + 1]; // Where does it end?
        
        // 3. Thread mapping
        int tid = threadIdx.x;
        int num_threads = blockDim.x;
        
        // 4. Do the math ONLY inside the fences
        // We use shared memory to sum up all tokens for this specific user
        __shared__ float shared_sum[256];
        float local_sum = 0.0f;
        
        // Iterate only between start_idx and end_idx
        for (int i = start_idx + tid; i < end_idx; i += num_threads) {
            local_sum += packed_tokens[i];
        }
        
        shared_sum[tid] = local_sum;
        __syncthreads();
        
        // Simple reduction for the block (assume 256 threads for simplicity)
        for (int stride = 128; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }
        
        // Write the final sum for this user to VRAM exactly once
        if (tid == 0) {
            output[user_id] = shared_sum[0];
        }
    }

    // --- C++ HOST FUNCTION ---
    torch::Tensor sequence_sum_1d(torch::Tensor packed_tokens, torch::Tensor cu_seqlens, int num_sequences) {
        auto output = torch::zeros({num_sequences}, torch::kFloat32).to(packed_tokens.device());
        
        // Assign 1 Block per Sequence, 256 Threads per Block
        int blocks = num_sequences;
        int threads = 256;
        
        sequence_sum_kernel<<<blocks, threads>>>(
            packed_tokens.data_ptr<float>(), 
            cu_seqlens.data_ptr<int>(), 
            output.data_ptr<float>()
        );
        
        return output;
    }
    """

    custom_module = load_inline(
        name='cu_seqlens_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['sequence_sum_1d'],
        with_cuda=True
    )
    print("✅ Compilation Successful!\n")

    # =====================================================================
    # 3. VERIFICATION
    # =====================================================================
    # Let's create 3 sequences of different lengths
    # User 0: [1, 1, 1, 1] (Sum = 4)
    # User 1: [2, 2]       (Sum = 4)
    # User 2: [3, 3, 3]    (Sum = 9)
    seqs = [
        torch.ones(4, dtype=torch.float32, device='cuda'),
        torch.full((2,), 2.0, dtype=torch.float32, device='cuda'),
        torch.full((3,), 3.0, dtype=torch.float32, device='cuda')
    ]
    
    packed_1d = torch.cat(seqs)
    lengths = [len(s) for s in seqs]
    cu_seqlens = torch.tensor([0, 4, 6, 9], dtype=torch.int32, device='cuda')
    
    print("Input 1D Array:  ", packed_1d.tolist())
    print("cu_seqlens Fences:", cu_seqlens.tolist())
    
    # Run the custom hardware router
    output = custom_module.sequence_sum_1d(packed_1d, cu_seqlens, len(lengths))
    
    print("\n--- 📊 HARDWARE ROUTING RESULT ---")
    print("Expected Sums: [4.0, 4.0, 9.0]")
    print("Hardware Sums:", output.tolist())
    
    if torch.allclose(output, torch.tensor([4.0, 4.0, 9.0], device='cuda')):
        print("\n✅ SUCCESS: The GPU successfully isolated the sequences in 1D memory using cu_seqlens!")

if __name__ == "__main__":
    test_cuda_cu_seqlens()