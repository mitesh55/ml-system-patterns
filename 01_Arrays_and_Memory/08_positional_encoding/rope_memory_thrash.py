import torch
import time
from torch.utils.cpp_extension import load_inline

def benchmark_rope_memory():
    print("--- 🌀 Pattern 08: RoPE Memory Thrash (Training vs Inference) ---\n")
    
    B, H, S, D = 8, 32, 4096, 128
    print(f"Simulating Attention State: [Batch={B}, Heads={H}, Seq={S}, Dim={D}]")
    
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32, requires_grad=True)
    cos = torch.randn(S, D, device='cuda', dtype=torch.float32)
    sin = torch.randn(S, D, device='cuda', dtype=torch.float32)
    
    print(f"Base VRAM (Data Loaded): {torch.cuda.memory_allocated() / (1024**2):.2f} MB\n")

    def apply_rope_pytorch(q, cos, sin):
        d_2 = q.shape[-1] // 2
        q1 = q[..., :d_2]
        q2 = q[..., d_2:]
        rotated_half = torch.cat([-q2, q1], dim=-1)
        return (q * cos) + (rotated_half * sin)

    # =====================================================================
    # [A] Native PyTorch (Training / Autograd ON)
    # =====================================================================
    vram_before_a = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    q_train = apply_rope_pytorch(q, cos, sin)
    torch.cuda.synchronize()
    
    time_a = (time.time() - start) * 1000
    waste_a = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_a

    print("[A] Native PyTorch (Training / Autograd ON):")
    print(f"    Time:      {time_a:.3f} ms")
    print(f"    Spike:     +{waste_a:.2f} MB (Saves intermediate tensors for Backprop)\n")

    # Clean up graph for next tests
    del q_train
    torch.cuda.empty_cache()

    # =====================================================================
    # [B] Native PyTorch (Inference / no_grad)
    # =====================================================================
    vram_before_b = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        start = time.time()
        q_infer = apply_rope_pytorch(q, cos, sin)
        torch.cuda.synchronize()
    
    time_b = (time.time() - start) * 1000
    waste_b = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_b

    print("[B] Native PyTorch (Inference / no_grad):")
    print(f"    Time:      {time_b:.3f} ms")
    print(f"    Spike:     +{waste_b:.2f} MB (Slicing and torch.cat still force allocations!)\n")

    # =====================================================================
    # [C] PyTorch 2.x Compiler (Inference / Triton Fusion)
    # =====================================================================
    print("⏳ Running torch.compile() JIT Warmup...")
    compiled_rope = torch.compile(apply_rope_pytorch)
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_rope(q, cos, sin)
        torch.cuda.synchronize()

    vram_before_c = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        start = time.time()
        q_compiled = compiled_rope(q, cos, sin)
        torch.cuda.synchronize()
        
    time_c = (time.time() - start) * 1000
    waste_c = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_c

    print("[C] PyTorch Compiled (Inference / Triton):")
    print(f"    Time:      {time_c:.3f} ms")
    print(f"    Spike:     +{waste_c:.2f} MB (Fast, but Python semantics still require a new output object)\n")

    # =====================================================================
    # [D] Custom In-Place C++ Kernel
    # =====================================================================
    print("🛠️ Compiling In-Place C++ RoPE Kernel...")
    cpp_source = "void apply_rope_inplace(torch::Tensor q, torch::Tensor cos, torch::Tensor sin);"
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    __global__ void rope_inplace_kernel(float* q, const float* cos, const float* sin, int num_elements, int seq_len, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int d_2 = head_dim / 2;
        if (idx < num_elements / 2) {
            int dim_idx = idx % d_2;
            int token_idx_global = idx / d_2; 
            int s = token_idx_global % seq_len;
            
            int pos1 = token_idx_global * head_dim + dim_idx;       
            int pos2 = pos1 + d_2; 
            int cache_pos1 = s * head_dim + dim_idx;
            int cache_pos2 = cache_pos1 + d_2;
            
            float q1 = q[pos1]; float q2 = q[pos2];
            float c1 = cos[cache_pos1]; float s1 = sin[cache_pos1];
            float c2 = cos[cache_pos2]; float s2 = sin[cache_pos2];
            
            q[pos1] = (q1 * c1) - (q2 * s1);
            q[pos2] = (q2 * c2) + (q1 * s2);
        }
    }
    void apply_rope_inplace(torch::Tensor q, torch::Tensor cos, torch::Tensor sin) {
        int num_elements = q.numel();
        int seq_len = q.size(-2);
        int head_dim = q.size(-1);
        int threads = 256;
        int blocks = ((num_elements / 2) + threads - 1) / threads;
        rope_inplace_kernel<<<blocks, threads>>>(q.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), num_elements, seq_len, head_dim);
    }
    """
    custom_module = load_inline(name='rope_inplace', cpp_sources=cpp_source, cuda_sources=cuda_source, functions=['apply_rope_inplace'], with_cuda=True, extra_cuda_cflags=['-O3'])

    with torch.no_grad():
        q_clone = q.clone() 
    torch.cuda.synchronize()
    
    vram_before_d = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    custom_module.apply_rope_inplace(q_clone, cos, sin)
    torch.cuda.synchronize()
    
    time_d = (time.time() - start) * 1000
    waste_d = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_d

    print("[D] In-Place C++ Kernel (Inference):")
    print(f"    Time:      {time_d:.3f} ms")
    print(f"    Spike:     +{waste_d:.2f} MB (Zero allocations. Raw pointer mutation!)\n")

if __name__ == "__main__":
    benchmark_rope_memory()


"""
OUTPUT in 4050 RTX 

--- 🌀 Pattern 08: RoPE Memory Thrash (Training vs Inference) ---

Simulating Attention State: [Batch=8, Heads=32, Seq=4096, Dim=128]
Base VRAM (Data Loaded): 516.00 MB

[A] Native PyTorch (Training / Autograd ON):
    Time:      51.171 ms
    Spike:     +2048.00 MB (Saves intermediate tensors for Backprop)

[B] Native PyTorch (Inference / no_grad):
    Time:      47.686 ms
    Spike:     +2048.00 MB (Slicing and torch.cat still force allocations!)

⏳ Running torch.compile() JIT Warmup...
[C] PyTorch Compiled (Inference / Triton):
    Time:      7.372 ms
    Spike:     +512.00 MB (Fast, but Python semantics still require a new output object)

🛠️ Compiling In-Place C++ RoPE Kernel...
[D] In-Place C++ Kernel (Inference):
    Time:      7.353 ms
    Spike:     +0.00 MB (Zero allocations. Raw pointer mutation!)


"""