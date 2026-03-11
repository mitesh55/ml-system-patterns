# ⚡ Pattern 02: Advanced Indexing & Filtering

> **Core Principle:** "Avoid Control Flow (`if/else`) inside loops. Use Data Flow (Masks) instead."

## 1. The Engineering "Why" (The Hook)
In Data Loaders and Preprocessing pipelines (e.g., Cleaning Data, ReLU Activation, Dropout), we often need to filter millions of elements.

* **The Junior Mistake:** Using Python loops with `if` statements. This kills the CPU pipeline due to **Branch Misprediction**.
* **The Senior Solution:** Using **Boolean Masks** to process data as vectors, allowing the CPU to execute math without "guessing" the outcome.

---

## 2. 📉 Performance Benchmark (ReLU Activation)
Comparing methods to filter negative values from **10 Million** elements.

| Method | Time (s) | Speedup | Memory Behavior |
| :--- | :--- | :--- | :--- |
| **[A] Python Loop** | `0.3047s` | 1x (Baseline) | **Slowest.** CPU stalls on branch prediction failures. |
| **[B] Boolean Math** | **`0.0255s`** | **~12x** 🚀 | **Fastest Calculation.** Allocates new memory for result. |
| **[C] In-Place Indexing** | `0.0505s` | **~6x** ⚡ | **Memory Efficient.** Modifies original array (No extra allocation). |

### 🔍 Analysis of Your Results
* **Why was [B] fastest?** It used pure arithmetic (`data * mask`). The CPU just multiplies. It doesn't have to "write" to scattered memory locations.
* **Why use [C] then?** Method B creates a *copy* of the array. If your dataset is 20GB, Method B crashes your RAM. Method C modifies the data **in-place** (`data[mask] = 0`).

*(Run `python benchmark_filtering.py` to reproduce results)*

---

## 3. Deep Dive: The "Branch Prediction" Problem
Why is the loop (`if x > 0`) so slow?

Modern CPUs fetch instructions in a pipeline (Fetch $\to$ Decode $\to$ Execute). When they see an `if`, they must **guess** which path to take before knowing the data.



* **The Loop:** Random data means the CPU guesses wrong ~50% of the time. Every wrong guess forces a **Pipeline Flush** (throwing away work).
* **The Mask:** No `if` statement. The CPU executes `val * 1` or `val * 0`. It's just math. The pipeline never stops.

---

## 4. View vs. Copy
Understanding when NumPy/PyTorch creates a **Copy** (New Memory) vs a **View** (Pointer to existing memory) is critical for optimization.

*(Run `python indexing_internals.py` to see this in action)*

| Operation | Syntax | Memory Type | Risk Level |
| :--- | :--- | :--- | :--- |
| **Basic Slicing** | `arr[0:500]` | **VIEW** ✅ | Low. Fast and memory efficient. |
| **Fancy Indexing** | `arr[[0, 1, 5]]` | **COPY** ⚠️ | High. **Doubles RAM usage** instantly. |
| **Boolean Masking** | `arr[arr > 0]` | **COPY** ⚠️ | High. Result size is unknown, so it forces a copy. |



### 💡 Key Takeaway for System Design
If you are building a **Data Loader** for a 100GB dataset:
1.  Use **Slicing** (`batch = data[i:i+batch_size]`) whenever possible to avoid RAM spikes.
2.  Avoid **Fancy Indexing** (`data[random_indices]`) unless necessary, as it triggers `malloc`.

---