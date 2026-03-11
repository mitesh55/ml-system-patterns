import numpy as np
import time

def benchmark_relu_filtering():
    """
    Benchmarks: 
    1. List Comprehension (Branch Misprediction Nightmare)
    2. Boolean Masking (Vectorized 'Where')
    3. In-Place Masking (Memory Optimized)
    
    Scenario: ReLU Activation (x = max(0, x)) on 10M elements.
    """
    N = 10_000_000
    print(f"--- Benchmarking ReLU (Filtering) on {N:,} elements ---\n")
    
    # Create random data between -1.0 and 1.0
    data = np.random.uniform(-1, 1, N).astype(np.float32)
    
    # ---------------------------------------------------------
    # Method A: Python List Comprehension (The "Loop")
    # ---------------------------------------------------------
    # Why this fails:
    # 1. Branch Misprediction: CPU hates random 'if' conditions inside loops.
    # 2. Memory Thrashing: Creates a new list, then converts back to array.
    
    data_list = data.tolist() # Convert to list for fair python comparison
    start = time.time()
    
    # The naive logic: check every number one by one
    relu_a = [x if x > 0 else 0 for x in data_list]
    
    time_a = time.time() - start
    print(f"[A] Python Loop        | Time: {time_a:.4f}s")

    # ---------------------------------------------------------
    # Method B: Boolean Masking (New Allocation)
    # ---------------------------------------------------------
    # Logic: 
    # 1. Generate a Boolean Mask (Bitmask) of True/False for (data > 0).
    # 2. Use the mask to multiply or select.
    # Note: 'data * (data > 0)' is a common trick.
    
    data_b = data.copy()
    start = time.time()
    
    # Creates a temporary boolean array, then performs multiplication
    mask = data_b > 0
    relu_b = data_b * mask 
    
    time_b = time.time() - start
    print(f"[B] Boolean Mask       | Time: {time_b:.4f}s")

    # ---------------------------------------------------------
    # Method C: In-Place Indexing 
    # ---------------------------------------------------------
    # Logic: 
    # 1. Use the mask directly as an index: `data[mask]`.
    # 2. Assign 0 to those specific locations in memory.
    # 3. Zero memory allocation for the result.
    
    data_c = data.copy()
    start = time.time()
    
    # "Find all indices where data < 0, and write 0 there."
    # No multiplication math needed. Just memory assignment.
    data_c[data_c < 0] = 0
    
    time_c = time.time() - start
    print(f"[C] In-Place Indexing  | Time: {time_c:.4f}s")

    # ---------------------------------------------------------
    # The Verdict
    # ---------------------------------------------------------
    speedup = time_a / time_c
    print("\n--- Engineering Insight ---")
    print(f"In-Place Indexing is {speedup:.1f}x faster than Loops.")
    print("Why? It avoids Branch Prediction failures. The CPU processes the mask as a vector.")

if __name__ == "__main__":
    benchmark_relu_filtering()