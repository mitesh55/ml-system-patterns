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

## 📊 Featured Benchmarks & Patterns

| Pattern | Context | Naive (Python/Loop) | Optimized (Vectorized/CUDA) | Speedup | Key Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[Broadcasting](./01_broadcasting_internals)** | Single-Target MSE | `8.6485s` | `0.0610s` | **~141x** 🚀 | Zero-Copy Views (Stride Manipulation) |

---

## 📂 Repository Structure

### 1. [Broadcasting Internals (The "Zero-Copy" Pattern)](./01_broadcasting_internals)
* **Concept:** How NumPy/PyTorch broadcast shapes without allocating memory.
* **Code:** * `benchmark_mse.py`: Proof that Broadcasting is 140x faster than loops and 1.3x faster than manual allocation.
    * `broadcasting_logic.py`: A from-scratch implementation of Virtual Views using Stride logic.

---

## 🛠️ Getting Started

```bash
# Clone the repository
git clone [https://github.com/mitesh55/ml-system-patterns.git](https://github.com/mitesh55/ml-system-patterns.git)

# Install dependencies (Minimal)
pip install numpy torch
