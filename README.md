# ⚡ ML System Patterns: From LeetCode to LLMs

> **Mission:** Deconstructing the computer science fundamentals behind modern ML infrastructure. 
> This repository bridges the gap between **Algorithms** (Theory) and **Production AI** (Hardware-Aware Implementation).

![CI/CD](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Language](https://img.shields.io/badge/python-3.10%2B-blue) ![Hardware](https://img.shields.io/badge/hardware-CUDA%20%2F%20CPU-orange)

## 🏗️ Core Philosophy
Standard DSA focuses on asymptotic complexity ($O(N)$). ML Engineering requires **Mechanical Sympathy**—understanding how the hardware (CPU/GPU) actually executes the math.

This repository explores:
* **Vectorization & Broadcasting:** Why stride manipulation beats loops.
* **Memory Management:** Avoiding the GIL and reducing cache misses.
* **Custom Kernels:** Writing Triton/CUDA when PyTorch isn't fast enough.

---

## 📊 Featured Benchmarks & Patterns

| DSA Foundation | ML Pattern | Context | Naive (Python) | Optimized (CUDA/Torch) | Speedup / Impact | Key Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arrays / Nested Loops** | [Broadcasting](./01_Arrays_and_Memory/01_broadcasting_vs_nested_loops) | Single-Target MSE | `8.6485s` | `0.0610s` | **~141x** 🚀 | Zero-Copy Views |
| **Conditionals / Branching** | [Advanced Indexing](./01_Arrays_and_Memory/02_advanced_indexing_vs_branching) | ReLU / Dropout | `0.3047s` | `0.0255s` | **~12x** ⚡ | Branchless Programming (Bitmasks) |
| **Sliding Window / 2D Arrays** | [Stride Manipulation](./01_Arrays_and_Memory/03_memory_layouts_and_strides) | 1D Convolutions | `~12.50s` | `0.00001s` | **O(1) Memory** 🧠 | Zero-Allocation Views (`as_strided`) |
| **Cartesian Product / Math**| [Vectorization & GEMM](./01_Arrays_and_Memory/04_vectorization_and_pairwise_distances) | KNN / Attention Distances | `~1050.72s` | `0.0042s` | **~249,000x** 🤯 | SIMD & Algebraic Expansion |
| **Pointers / Array Indexing** | [Scatter & Gather](./01_Arrays_and_Memory/05_scatter_and_gather) | One-Hot / LLM Logits | `~35.76s` | `0.14s` | **~250x** ⚡ | Atomic Parallel Memory Routing |
| **Math / Array Reductions** | [Batch Norm & Fusion](./01_Arrays_and_Memory/06_batch_normalization) | Intermediate Tensor Trap | `0.0307s` | `0.0082s` | **~3.7x** ⚡ (1.6GB RAM Saved) | Register Fusion & cuDNN Warp Reductions |
---
| **Arrays / Ragged Data** | [1D Sequence Packing](./01_Arrays_and_Memory/07_sequence_collation) | LLM Serving & FlashAttention | `2D Padding (OOM Risk)` | `1D Concat + cu_seqlens` | **Massive VRAM Reclaimed** 🧠 | Eliminating Ghost Compute & `cu_seqlens` hardware routing |

## 📂 Repository Structure & Roadmap

This repository follows a structured roadmap, mapping classic DSA categories directly to production ML infrastructure.

### 📍 Phase 1: Arrays, Matrices & Memory Layouts (Current)
*Bridging continuous memory structures with hardware-aware tensor operations.*

* **1. [Array Traversal -> Broadcasting Internals](./01_Arrays_and_Memory/01_broadcasting_vs_nested_loops)**
    * **DSA Concept:** Array traversal and $O(N^2)$ nested loop optimization.
    * **ML Application:** How NumPy/PyTorch broadcast shapes without allocating memory (The "Zero-Copy" Pattern). Proof that Stride logic is 140x faster than loops.
* **2. [If/Else Branching -> Advanced Indexing & Filtering](./01_Arrays_and_Memory/02_advanced_indexing_vs_branching)**
    * **DSA Concept:** Conditionals and array filtering.
    * **ML Application:** Why `if x > 0` kills CPU performance (Branch Prediction Failures). Replacing branch logic with Boolean Masks for a 12x speedup and memory-safe in-place operations.
* **3. [Sliding Window -> Memory Layouts & Stride Manipulation](./01_Arrays_and_Memory/03_memory_layouts_and_strides)**
    * **DSA Concept:** Sliding Window and Matrix Transposition.
    * **ML Application:** Emulating 2D/3D convolutions on 1D physical RAM without memory copying. Why `for` loops cause Out-Of-Memory (OOM) errors on GPUs, and avoiding the `is_contiguous()` trap for CUDA kernels.
* **4. [Nested Loops -> Vectorization & Pairwise Distances](./01_Arrays_and_Memory/04_vectorization_and_pairwise_distances)**
    * **DSA Concept:** Cartesian Products and $O(N^2)$ Distance Matrices.
    * **ML Application:** The math powering Vector Databases (RAG) and Transformer Attention. Why PyTorch broadcasting `unsqueeze()` causes catastrophic OOM errors, and how algebraic expansion (GEMM) pushes computation directly into hardware registers to save massive amounts of RAM.
* **5. [Array Indexing -> Memory Routing (Scatter & Gather)](./01_Arrays_and_Memory/05_scatter_and_gather)**
    * **DSA Concept:** Pointers, Array Indexing, and Write/Read Routing.
    * **ML Application:** How `scatter_` handles One-Hot encoding and Mixture of Experts (MoE) routing, and how `gather` extracts true-token probabilities from LLM logits. Exposing why sequential Python loops cause race conditions, and how the C++ backend utilizes atomic operations for massively parallel memory writes.

* **6. [Statistical Math -> Batch Normalization & Operator Fusion](./01_Arrays_and_Memory/06_batch_normalization)**
    * **DSA Concept:** Array reductions (Mean, Variance) and multi-pass algorithms.
    * **ML Application:** Bypassing the "Intermediate Tensor Trap." Proof that naive PyTorch math dynamically allocates 1.6 GB of temporary VRAM for a 400MB batch, and how writing a fused CUDA kernel drops memory allocation to zero. Understanding why cuDNN's Warp-Level reductions are the ultimate hardware ceiling.
* **7. [Ragged Arrays -> Sequence Collation & 1D Packing](./01_Arrays_and_Memory/07_sequence_collation)**
    * **DSA Concept:** Variable-length arrays, memory alignment, and ragged data structures.
    * **ML Application:** Why `torch.stack` and 2D padding cause "Ghost Compute" (wasted FLOPs and VRAM). How modern LLM inference engines (vLLM) and FlashAttention use 1D array packing (`torch.cat`) and `cu_seqlens` pointers to physically isolate user prompts at the silicon level, bypassing padding entirely.
     
### 📍 Phase 2: Hash Maps & Search Optimization (Upcoming)
*Bridging exact key-value retrieval with approximate semantic search.*
* **DSA Concept:** Hashing, Tries, Collision Resolution.
* **ML Application:** Vector Databases, K-NN approximations, and optimized embedding retrieval for RAG pipelines.

### 📍 Phase 3: Trees, Graphs & Execution Engines (Upcoming)
*Bridging node traversals with model compilation.*
* **DSA Concept:** Trees, DFS/BFS, Directed Acyclic Graphs (DAGs).
* **ML Application:** Computation Graphs, Autograd engines, and AST parsing for dynamic graph execution.

### 📍 Phase 4: Concurrency & Distributed Systems (Upcoming)
*Bridging multi-threading with GPU scaling.*
* **DSA Concept:** Concurrency, Locks, Distributed Algorithms.
* **ML Application:** Custom CUDA kernels, TensorRT operator fusion, vLLM continuous batching, and multi-GPU orchestrations.

---

## 🛠️ Getting Started

```bash
# Clone the repository
git clone [https://github.com/mitesh55/ml-system-patterns.git](https://github.com/mitesh55/ml-system-patterns.git)

# Install dependencies (Minimal)
pip install numpy torch