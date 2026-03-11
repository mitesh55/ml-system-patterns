import torch
import time

def benchmark_scatter_one_hot():
    """
    Benchmarks converting millions of categorical indices into sparse One-Hot vectors.
    """
    N = 10_000_000
    Vocab_Size = 100
    print(f"--- Benchmarking One-Hot Encoding ({N:,} tokens, Vocab: {Vocab_Size}) ---\n")
    
    tokens = torch.randint(0, Vocab_Size, (N,))

    # ---------------------------------------------------------
    # Method A: Python Loop (Unoptimized Baseline)
    # ---------------------------------------------------------
    start = time.time()
    
    # We only run a subset (100k) and extrapolate to prevent the system from freezing
    subset_N = 100_000
    tokens_subset = tokens[:subset_N]
    
    canvas_a = torch.zeros(subset_N, Vocab_Size, dtype=torch.int8)
    for i in range(subset_N):
        canvas_a[i, tokens_subset[i]] = 1
        
    time_a = (time.time() - start) * (N / subset_N)
    print(f"[A] Python Loop          | Time: ~{time_a:.4f}s | Hardware: CPU Scalar (Sequential)")

    # ---------------------------------------------------------
    # Method B: scatter_ (Hardware-Aware Memory Routing)
    # ---------------------------------------------------------
    start = time.time()
    
    canvas_b = torch.zeros(N, Vocab_Size, dtype=torch.int8)
    canvas_b.scatter_(1, tokens.unsqueeze(1), value=1)
    
    time_b = time.time() - start
    print(f"[B] Native scatter_      | Time:  {time_b:.6f}s | Hardware: C++ Parallel Writes")

    # ---------------------------------------------------------
    # Method C: torch.nn.functional.one_hot (The Wrapper)
    # ---------------------------------------------------------
    start = time.time()
    
    canvas_c = torch.nn.functional.one_hot(tokens, num_classes=Vocab_Size)
    
    time_c = time.time() - start
    print(f"[C] F.one_hot (Wrapper)  | Time:  {time_c:.6f}s | Hardware: C++ Parallel Writes")

    # ---------------------------------------------------------
    # The Verdict & Under the Hood
    # ---------------------------------------------------------
    speedup_loop = time_a / time_b
    speedup_wrapper = time_c / time_b
    
    print("\n--- Engineering Insight ---")
    print(f"1. vs Python Loops: scatter_ is {speedup_loop:.0f}x faster.")
    print(f"2. vs F.one_hot:    scatter_ is {speedup_wrapper:.1f}x faster.")
    
    print("\n❓ QUESTION: What does `scatter_` do under the hood that Python loops cannot?")
    print("💡 INTUITION: Bypassing the GIL and Parallelizing Memory Writes")
    print("   Python `for` loops are strictly sequential. The Python GIL forces the CPU")
    print("   to write index 0, then index 1, then index 2. One by one.")
    print("   `scatter_` drops down to a C++ ATen kernel. It spins up thousands of")
    print("   parallel threads simultaneously. Every thread grabs one token and writes")
    print("   it directly to RAM. It uses hardware-level 'Atomic Operations' to ensure")
    print("   two threads don't overwrite each other. Massively parallel, zero Python overhead.")

    print("\n❓ QUESTION: If F.one_hot uses scatter, why is it slower?")
    print("💡 INTUITION: The Memory Bandwidth Trap")
    print("   `F.one_hot` automatically allocates a 64-bit integer tensor (`torch.int64`).")
    print("   Our `scatter_` explicitly used `torch.int8` (1 byte). Moving 8x less data")
    print("   through the hardware memory bus drastically reduces execution time.")

if __name__ == "__main__":
    benchmark_scatter_one_hot()