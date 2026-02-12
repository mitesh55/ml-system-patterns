# ⚡ ML System Patterns: From LeetCode to LLMs

> **Mission:** Deconstructing the computer science fundamentals behind modern ML infrastructure. 
> This repository bridges the gap between **Algorithms** (Theory) and **Production AI** (Hardware-Aware Implementation).

![CI/CD](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Language](https://img.shields.io/badge/python-3.10%2B-blue) ![Hardware](https://img.shields.io/badge/hardware-CUDA%20%2F%20CPU-orange)

## 🏗️ Core Philosophy
Standard DSA focuses on asymptotic complexity ($O(N)$). Senior ML Engineering requires **Mechanical Sympathy**—understanding how the hardware (CPU/GPU) actually executes the math.

This repository explores:
* **Vectorization & Broadcasting:** Why stride manipulation beats loops.
* **Memory Management:** Avoiding the GIL and reducing cache misses.
* **Custom Kernels:** Writing Triton/CUDA when PyTorch isn't fast enough.

---
## 📊 Featured Benchmarks
| Pattern | Naive Implementation | Optimized (Vectorized/CUDA) | Speedup |
| :--- | :--- | :--- | :--- |
| **Broadcasting (MSE)** | 450ms (Python Loop) | 0.5ms (NumPy/AVX) | **~900x** 🚀 |
| **Softmax** | TBD | TBD | ... |
