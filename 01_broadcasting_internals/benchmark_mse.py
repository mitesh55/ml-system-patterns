import numpy as np
import time
import sys

def get_size(obj):
    """Helper to show memory usage in MB"""
    return sys.getsizeof(obj) / (1024 * 1024)

def benchmark_broadcast_vs_allocation():
    N = 10_000_000  # 10 Million elements to make memory usage visible
    print(f"--- Benchmarking: Single-Target MSE on {N:,} elements ---\n")
    
    # The Setup: 10 Million predictions, but only ONE target value.
    predictions = np.random.rand(N).astype(np.float32)
    target_scalar = 1.0 

    # ---------------------------------------------------------
    # Method A: The "Naive" Python Loop
    # ---------------------------------------------------------
    # Proof: Shows interpreter overhead.
    start = time.time()
    error = 0.0
    for i in range(len(predictions)):
        # CPU fetches 'target_scalar' repeatedly from stack
        diff = predictions[i] - target_scalar 
        error += diff ** 2
    mse_a = error / N
    time_a = time.time() - start
    print(f"[A] Naive Loop         | Time: {time_a:.4f}s | MSE: {mse_a:.4f}")

    # ---------------------------------------------------------
    # Method B: Manual Expansion (The "Bad" Vectorization)
    # ---------------------------------------------------------
    # Proof: Shows cost of Memory Allocation.
    # We manually create an array of 10M '1.0's to match shapes.
    # This is what beginners do when they don't know broadcasting.
    start = time.time()
    
    # 🛑 MEMORY SPIKE HERE: Allocating 40MB just to store '1.0' 10M times
    expanded_targets = np.full(N, target_scalar, dtype=np.float32) 
    
    diff = predictions - expanded_targets
    mse_b = np.mean(diff ** 2)
    time_b = time.time() - start
    
    mem_b = get_size(expanded_targets)
    print(f"[B] Manual Expansion   | Time: {time_b:.4f}s | MSE: {mse_b:.4f} | Waste: {mem_b:.1f} MB RAM")

    # ---------------------------------------------------------
    # Method C: Broadcasting 
    # ---------------------------------------------------------
    # Proof: Shows Zero-Copy Optimization.
    # numpy effectively "pretends" the scalar is an array of size N
    # by setting the stride to 0.
    start = time.time()
    
    # ✅ NO MEMORY SPIKE. The scalar is read repeatedly from register/cache.
    diff = predictions - target_scalar  
    mse_c = np.mean(diff ** 2)
    time_c = time.time() - start
    
    print(f"[C] Broadcasting       | Time: {time_c:.4f}s | MSE: {mse_c:.4f} | Waste: 0.0 MB RAM")

    # ---------------------------------------------------------
    # The Verdict
    # ---------------------------------------------------------
    print("\n--- Engineering Insight ---")
    print(f"Broadcasting is {time_b / time_c:.1f}x faster than Allocation (Method B).")
    print("Why? Because it avoids `malloc` (memory allocation) and memory write bandwidth.")

if __name__ == "__main__":
    benchmark_broadcast_vs_allocation()

'''
The naive loop dispatches one CPU instruction per element. NumPy uses SIMD (Single Instruction, Multiple Data), specifically AVX2 instructions, allowing the CPU to process 8 floats (256-bit register) in a single clock cycle.
'''