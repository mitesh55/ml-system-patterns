import torch
import time

def benchmark_sliding_window():
    """
    Benchmarks extracting a sliding window (e.g., for 1D Convolutions or Time-Series).
    Window Size: 3
    """
    N = 10_000_000
    print(f"--- Benchmarking Sliding Window on {N:,} elements ---\n")
    
    # Create the 1D sequence
    data = torch.arange(N, dtype=torch.float32)
    window_size = 3
    num_windows = N - window_size + 1
    
    # ---------------------------------------------------------
    # Method A: Python Loop & Stack
    # ---------------------------------------------------------
    start = time.time()
    
    # Warning: Do not actually run this loop on 10M elements, it will freeze your laptop.
    # We will benchmark on 100k elements and multiply to show the catastrophic failure.
    small_N = 100_000
    small_data = data[:small_N]
    
    windows_list = []
    for i in range(small_N - window_size + 1):
        windows_list.append(small_data[i : i + window_size])
    
    naive_result = torch.stack(windows_list)
    time_a = (time.time() - start) * 100  # Extrapolate to 10M
    
    print(f"[A] Loop & Stack (Extrapolated) | Time: ~{time_a:.4f}s | Memory: O(N * W)")

    # ---------------------------------------------------------
    # Method B: Stride Manipulation
    # ---------------------------------------------------------
    start = time.time()
    
    # The Magic Trick: 
    # Shape = (number_of_windows, window_size)
    # Stride = (1 step to next window, 1 step to next element)
    stride_result = torch.as_strided(
        data, 
        size=(num_windows, window_size), 
        stride=(1, 1)
    )
    
    time_b = time.time() - start
    print(f"[B] torch.as_strided            | Time:  {time_b:.6f}s | Memory: O(1)")

    # ---------------------------------------------------------
    # The Verdict
    # ---------------------------------------------------------
    speedup = time_a / time_b if time_b > 0 else float('inf')
    print("\n--- Engineering Insight ---")
    print(f"Stride Manipulation is {speedup:.0f}x faster.")
    
    if data.data_ptr() == stride_result.data_ptr():
        print("✅ ZERO memory was copied. The 2D matrix is a pure illusion over the 1D array.")

if __name__ == "__main__":
    benchmark_sliding_window()