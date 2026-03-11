import torch
import time
import math

def benchmark_euclidean_distance():
    """
    Benchmarks calculating the pairwise Euclidean distance matrix 
    between two sets of vectors (e.g., K-Means, KNN, or Attention Query-Keys).
    """
    N, M, D = 1000, 1000, 128  
    print(f"--- Benchmarking Pairwise Distances ({N}x{D} vs {M}x{D}) ---\n")
    
    X = torch.rand(N, D)
    Y = torch.rand(M, D)

    # ---------------------------------------------------------
    # Method A: Python Nested Loops (Unoptimized Baseline)
    # ---------------------------------------------------------
    start = time.time()
    
    # We only run a subset (100x100) and extrapolate, otherwise this takes minutes.
    subset_N, subset_M = 100, 100
    dist_a_subset = torch.zeros(subset_N, subset_M)
    
    for i in range(subset_N):
        for j in range(subset_M):
            # Scalar math in Python
            dist_a_subset[i, j] = math.sqrt(sum((X[i, k] - Y[j, k])**2 for k in range(D)))
            
    time_a = (time.time() - start) * ( (N * M) / (subset_N * subset_M) )
    print(f"[A] Naive Python Loops | Time: ~{time_a:.4f}s | Hardware: CPU Scalar")

    # ---------------------------------------------------------
    # Method B: Vectorized Broadcasting (Hardware-Aware)
    # ---------------------------------------------------------
    start = time.time()
    
    # X shape: (N, 1, D)
    # Y shape: (1, M, D)
    # Broadcasting creates a virtual (N, M, D) tensor before summing over D
    dist_b = torch.sqrt(((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(dim=2))
    
    time_b = time.time() - start
    print(f"[B] PyTorch Broadcast  | Time:  {time_b:.6f}s | Hardware: SIMD Vectorized")

    # ---------------------------------------------------------
    # Method C: ATen / BLAS Optimized (Native)
    # ---------------------------------------------------------
    start = time.time()
    
    # Uses highly optimized C++ backend (Matrix Multiplication under the hood)
    dist_c = torch.cdist(X, Y)
    
    time_c = time.time() - start
    print(f"[C] Native torch.cdist | Time:  {time_c:.6f}s | Hardware: C++ ATen/BLAS")

    # ---------------------------------------------------------
    # The Verdict
    # ---------------------------------------------------------
    speedup_b = time_a / time_b
    speedup_c = time_a / time_c
    
    print("\n--- Engineering Insight ---")
    print(f"Vectorization is {speedup_b:.0f}x faster than Python loops.")
    print(f"Native ATen is {speedup_c:.0f}x faster by avoiding large intermediate memory allocations.")

if __name__ == "__main__":
    benchmark_euclidean_distance()

"""
OUTPUT(On RTX 4050) :

--- Benchmarking Pairwise Distances (1000x128 vs 1000x128) ---

[A] Naive Python Loops | Time: ~1018.4807s | Hardware: CPU Scalar
[B] PyTorch Broadcast  | Time:  0.235506s | Hardware: SIMD Vectorized
[C] Native torch.cdist | Time:  0.011292s | Hardware: C++ ATen/BLAS

--- Engineering Insight ---
Vectorization is 4325x faster than Python loops.
Native ATen is 90199x faster by avoiding large intermediate memory allocations.

"""