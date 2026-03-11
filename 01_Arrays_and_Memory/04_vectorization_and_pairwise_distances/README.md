# ⚡ Pattern 04: Vectorization & Pairwise Distances

> **Core Principle:** "Loops kill compute. Algebra saves memory. Push all iteration down to the hardware via SIMD."

## 1. The Engineering Challenge
Calculating Euclidean distances between massive sets of vectors is the central bottleneck of modern AI architecture. It is the foundation for:
* **Vector Databases (RAG):** Comparing user queries to millions of document embeddings.
* **Transformer Attention:** Calculating Query-Key similarity matrices.
* **Computer Vision:** K-Means clustering and feature matching.

* **Unoptimized Baseline:** Using Python nested `for` loops. The interpreter handles one scalar number at a time, ignoring the parallel architecture of modern processors.
* **Hardware-Aware Implementation:** Structuring the data mathematically so the backend C++ ATen library can execute SIMD (Single Instruction, Multiple Data) operations or Matrix Multiplications (GEMM), processing thousands of dimensions in a single clock cycle.

---

## 2. 📉 Performance Benchmark ($N^2$ Distance Matrix)
Comparing two tensors of shape `(1000, 128)` — calculating the distance between 1,000 vectors and another 1,000 vectors.

| Implementation | Time (s) | Speedup | Hardware Execution |
| :--- | :--- | :--- | :--- |
| **[A] Naive Python Loops** | `~1050.7265s` | 1x (Baseline) | CPU Scalar (1 math op per cycle) |
| **[B] PyTorch Broadcast** | `0.1910s` | **5,499x** 🚀 | SIMD Vectorized |
| **[C] Native ATen (`cdist`)** | **`0.0042s`** | **249,099x** ⚡ | C++ BLAS (Highly Optimized) |

*(Run `python benchmark_pairwise_distance.py` to reproduce results)*

---

## 3. 💡 Intuition 1: The Orthogonal Grid (Cartesian Product)
To calculate pairwise distances without loops, we rely on PyTorch Broadcasting: `X.unsqueeze(1) - Y.unsqueeze(0)`. 

**Question:** Why do we use `unsqueeze(1)` for X and `unsqueeze(0)` for Y? Why not use the same axis?

* `X.unsqueeze(1)` converts shape `(N, D)` to `(N, 1, D)`. It becomes a vertical column. PyTorch stretches it horizontally.
* `Y.unsqueeze(0)` converts shape `(M, D)` to `(1, M, D)`. It becomes a horizontal row. PyTorch stretches it vertically.



When they subtract, they intersect orthogonally to form a massive `(N, M, D)` grid containing every possible combination. If we unsqueezed them both at the same axis (e.g., both at `1`), they would stay parallel and never cross, resulting in element-wise subtraction instead of a Cartesian product!

---

## 4. 🚨 Intuition 2: The Broadcasting Trap & Memory Spikes
While broadcasting is 5,499x faster than loops, it introduces a catastrophic memory risk. 

**The Vulnerability:** The exact moment memory spikes is at the subtraction operator `-`. 
To compute `(1000, 1, 128) - (1, 1000, 128)`, PyTorch expands the tensors to `(1000, 1000, 128)` and must physically write all 128,000,000 intermediate differences into RAM. 
* **RAM Required:** `488.28 MB`. 
If you scale this to a 100,000-document Vector Database, your system will instantly hit an Out-Of-Memory (OOM) crash.

---

## 5. ⚙️ The Hardware Solution: GEMM Algebraic Expansion
How does the Native ATen (`cdist`) method run 249,000x faster and avoid the OOM crash? It completely bypasses the `D` dimension allocation using algebra:

$$(X - Y)^2 = X^2 + Y^2 - 2XY$$

Because the $2XY$ component is just a **Matrix Multiplication (GEMM)**, the GPU calculates the sum of the `128` feature dimensions directly inside its hardware registers. It *never* writes the intermediate `128` values to main VRAM.



**The Impact:**
By transforming the subtraction problem into Matrix Multiplication, the maximum allocation shape drops to `(1000, 1000)`. 
* **RAM Required:** `3.81 MB`.
* **Result:** A **128x reduction in memory**, completely neutralizing the OOM threat while maximizing calculation speed.

*(Run `python vectorization_internals.py` to see the memory footprint tracking and math verification in the terminal)*

---


## 6. ⚙️ The Descent to Silicon: Hardware Registers vs VRAM
In Section 4, we noted that PyTorch broadcasting causes a memory spike because it writes the intermediate $D$ dimension to physical VRAM. How exactly does the hardware avoid this?

By dropping down to inline CUDA, we can map a 2D grid of threads directly to the $(N, M)$ output matrix and force the GPU to do the math inside its silicon registers.

**The Register Loop:**
```cpp
float dist_sq = 0.0f; // This variable lives inside the SM Register, NOT VRAM!
for (int d = 0; d < D; ++d) {
    float diff = X[row * D + d] - Y[col * D + d];
    dist_sq += diff * diff;
}
output[row * M + col] = sqrtf(dist_sq); // We write to VRAM only ONCE.



### ⏱️ The Reality Check: 0.07s vs 0.004s

Running this custom kernel successfully bypasses the **11.92 GB** broadcasting memory trap, safely allocating only **0.093 GB**. However, looking at the execution logs reveals a massive performance gap:

* **Custom Register Kernel:** `0.07015s`
* **Native `torch.cdist`:** `0.00440s`

**Why is our custom C++ kernel ~16x slower than PyTorch?**
Look closely at the `for` loop above. While the *accumulation* happens in the ultra-fast register (`dist_sq`), the *data fetching* (`X[...]` and `Y[...]`) pulls directly from Global VRAM on every single iteration.

PyTorch's `cdist` is powered by NVIDIA's highly optimized **cuBLAS** library. Instead of reading from VRAM sequentially, cuBLAS utilizes **Tiled Shared Memory** (the GPU's physical L1 Cache). It loads large mathematical "tiles" of the `X` and `Y` matrices into the cache *once*, computes all possible pairwise combinations for that tile, and then moves to the next.

This is the ultimate lesson of system engineering: Simply writing CUDA code isn't enough.

```
