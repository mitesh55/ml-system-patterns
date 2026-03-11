import torch

def distance_algebra_intuition():
    print("--- Memory-Optimized Distances: The Broadcasting Trap vs GEMM ---\n")
    
    # Simulating a small Vector Database query
    # N = 1000 Document Chunks, M = 1000 User Queries, D = 128 Embedding Dimensions
    N, M, D = 1000, 1000, 128
    X = torch.rand(N, D)
    Y = torch.rand(M, D)

    print(f"Dataset: X is {X.shape}, Y is {Y.shape}\n")

    # =====================================================================
    # 1. THE BROADCASTING TRAP (Memory Heavy)
    # =====================================================================
    print("❓ QUESTION: Why does broadcasting crash systems on large datasets?")
    
    # Calculate how much memory an (N, M, D) float32 tensor requires
    mem_broadcast = N * M * D * 4 / (1024**2) 
    print(f"-> To compute `X - Y`, broadcasting expands shapes to ({N}, {M}, {D}).")
    print(f"-> This requires allocating {mem_broadcast:.2f} MB of continuous RAM.")
    
    print("\n💡 INTUITION: The Memory Spike")
    print("   The exact moment memory spikes is at the subtraction operator `-`.")
    print("   PyTorch must physically write all 128,000,000 intermediate differences")
    print("   into RAM before the next step (`** 2`) can read them.")
    
    # The actual execution
    dist_broadcast = torch.sqrt(((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(dim=2))
    print("❓ QUESTION: Why we did X.unsqueeze(1) - Y.unsqueeze(0) and why not X.unsqueeze(0) - Y.unsqueeze(0) ?")
    print("\n💡 INTUITION: The Orthogonal Grid (Cartesian Product)")
    print("   Why do we use X.unsqueeze(1) and Y.unsqueeze(0)?")
    print("   -> X becomes (N, 1, D) : A vertical column. PyTorch stretches it horizontally.")
    print("   -> Y becomes (1, M, D) : A horizontal row. PyTorch stretches it vertically.")
    print("   When they subtract, they intersect perfectly to form an (N, M, D) grid.")
    print("   If we unsqueezed them both at (1), they would stay parallel and never cross!")
    # =====================================================================
    # 2. THE ALGEBRAIC SOLUTION (Compute Heavy, Memory Light)
    # =====================================================================
    print("\n-------------------------------------------------------------------")
    print("❓ QUESTION: How do we avoid allocating the D dimension in RAM?")
    
    print("\n💡 INTUITION: The GEMM Hardware Trick")
    print("   We algebraically expand the formula: (X - Y)^2 = X^2 + Y^2 - 2XY")
    print("   Because `XY` is a Matrix Multiplication (GEMM), the GPU calculates")
    print("   the sum of `D` dimensions directly inside its hardware registers.")
    print("   It never writes the intermediate `D` values to main VRAM.")
    
    # Calculate the memory required for the largest intermediate tensor (N, M)
    mem_algebra = (N * M) * 4 / (1024**2)
    print(f"\n-> By using Matrix Multiplication, our max allocation is ({N}, {M}).")
    print(f"-> This only requires {mem_algebra:.2f} MB of RAM (a {D}x reduction).")
    
    # Step A: X^2 -> Shape (N, 1)
    x_sq = X.pow(2).sum(dim=1, keepdim=True)
    
    # Step B: Y^2 -> Shape (1, M)
    y_sq = Y.pow(2).sum(dim=1).unsqueeze(0)
    
    # Step C: 2XY -> Matrix Multiplication -> Shape (N, M)
    # This is where the hardware magic happens.
    xy = torch.matmul(X, Y.t())
    
    # Step D: Combine -> (X^2 - 2XY + Y^2) -> Shape (N, M)
    dist_algebra_sq = x_sq - 2 * xy + y_sq
    
    # Clamp to avoid extremely small negative numbers due to floating point inaccuracies
    dist_algebra = torch.sqrt(torch.clamp(dist_algebra_sq, min=1e-8))

    # =====================================================================
    # 3. VERIFICATION
    # =====================================================================
    print("\n-------------------------------------------------------------------")
    max_diff = torch.max(torch.abs(dist_broadcast - dist_algebra))
    print(f"Max difference between mathematical results: {max_diff:.6e}")
    
    if max_diff < 1e-4:
        print("✅ SUCCESS: The algebraic expansion perfectly matches the broadcast distance.")
        print("   You just saved your GPU from an Out-Of-Memory (OOM) crash.")

if __name__ == "__main__":
    distance_algebra_intuition()


"""
OUTPUT (On RTX 4050):

--- Memory-Optimized Distances: The Broadcasting Trap vs GEMM ---

Dataset: X is torch.Size([1000, 128]), Y is torch.Size([1000, 128])

❓ QUESTION: Why does broadcasting crash systems on large datasets?
-> To compute `X - Y`, broadcasting expands shapes to (1000, 1000, 128).
-> This requires allocating 488.28 MB of continuous RAM.

💡 INTUITION: The Memory Spike
   The exact moment memory spikes is at the subtraction operator `-`.
   PyTorch must physically write all 128,000,000 intermediate differences
   into RAM before the next step (`** 2`) can read them.
❓ QUESTION: Why we did X.unsqueeze(1) - Y.unsqueeze(0) and why not X.unsqueeze(0) - Y.unsqueeze(0) ?

💡 INTUITION: The Orthogonal Grid (Cartesian Product)
   Why do we use X.unsqueeze(1) and Y.unsqueeze(0)?
   -> X becomes (N, 1, D) : A vertical column. PyTorch stretches it horizontally.
   -> Y becomes (1, M, D) : A horizontal row. PyTorch stretches it vertically.
   When they subtract, they intersect perfectly to form an (N, M, D) grid.
   If we unsqueezed them both at (1), they would stay parallel and never cross!

-------------------------------------------------------------------
❓ QUESTION: How do we avoid allocating the D dimension in RAM?

💡 INTUITION: The GEMM Hardware Trick
   We algebraically expand the formula: (X - Y)^2 = X^2 + Y^2 - 2XY
   Because `XY` is a Matrix Multiplication (GEMM), the GPU calculates
   the sum of `D` dimensions directly inside its hardware registers.
   It never writes the intermediate `D` values to main VRAM.

-> By using Matrix Multiplication, our max allocation is (1000, 1000).
-> This only requires 3.81 MB of RAM (a 128x reduction).

-------------------------------------------------------------------
Max difference between mathematical results: 6.675720e-06
✅ SUCCESS: The algebraic expansion perfectly matches the broadcast distance.
   You just saved your GPU from an Out-Of-Memory (OOM) crash.

"""