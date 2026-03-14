import torch

def batchnorm_intuition():
    print("--- Neural Net Internals: Demystifying Batch Normalization ---\n")
    
    # 1. The Setup: A batch of 4 examples, each with 3 features (N=4, D=3)
    # Imagine these are the activations coming out of a Linear layer before a ReLU.
    X = torch.tensor([
        [ 2.0,  5.0, -1.0],
        [ 4.0,  5.0,  1.0],
        [ 6.0,  1.0,  3.0],
        [ 8.0,  1.0,  5.0]
    ], dtype=torch.float32)
    
    N, D = X.shape
    print(f"Input Matrix (X) Shape: {N} instances, {D} features\n{X}\n")

    # =====================================================================
    # 2. THE MATHEMATICS (Vectorized)
    # =====================================================================
    print("❓ QUESTION: How do we calculate stats across the 'Batch' dimension?")
    print("💡 INTUITION: We crush the rows (dim=0) to find the average of each feature column.\n")
    
    # Step A: Mean (mu)
    mu = X.mean(dim=0)
    print(f"-> Mean (mu) of each feature: {mu.tolist()}")
    
    # Step B: Variance (var) - Note: PyTorch uses unbiased=False for BatchNorm
    var = X.var(dim=0, unbiased=False)
    print(f"-> Variance (var) of each feature: {var.tolist()}\n")
    
    # Step C: Normalize (Zero mean, Unit variance)
    eps = 1e-5 # Epsilon prevents dividing by zero
    X_norm = (X - mu) / torch.sqrt(var + eps)
    
    # Step D: Scale and Shift (Gamma and Beta)
    # The network Learns these parameters to optionally undo the normalization if needed
    gamma = torch.ones(D)  # Scale (Weight)
    beta = torch.zeros(D)  # Shift (Bias)
    
    X_out = gamma * X_norm + beta
    
    print("--- Final Normalized Output ---")
    print(X_out)

# =====================================================================
    # 3. VERIFICATION & THE EVAL TRAP
    # =====================================================================
    print("\n-------------------------------------------------------------------")
    print("❓ QUESTION: Does our manual math match PyTorch's native layer?")
    
    bn_layer = torch.nn.BatchNorm1d(num_features=D, eps=eps, momentum=None, affine=True)
    bn_layer.weight.data = gamma
    bn_layer.bias.data = beta
    
    # --- THE TRAP ---
    bn_layer.eval() 
    trap_out = bn_layer(X)
    print("\n🚨 THE EVAL TRAP:")
    print("   If you run bn_layer.eval(), PyTorch ignores the batch and uses historical stats (0 and 1).")
    print("   Look what happens: the output is identical to the input!")
    print(trap_out[0]) # Print just the first row to prove it
    
    # --- THE FIX ---
    bn_layer.train() # Switch back to training mode
    native_out = bn_layer(X)
    
    max_diff = torch.max(torch.abs(X_out - native_out)).item()
    if max_diff < 1e-4:
        print(f"\n✅ SUCCESS: In train() mode, our manual math perfectly matches PyTorch (Max Diff: {max_diff:.6e}).")
    # =====================================================================
    # 4. THE SYSTEMS BOTTLENECK
    # =====================================================================
    print("\n🚨 SYSTEM DESIGN WARNING: The Intermediate Tensor Trap")
    print("   Look at this line: X_out = gamma * ((X - mu) / sqrt(var)) + beta")
    print("   In pure Python/PyTorch, this creates 4 temporary tensors in VRAM:")
    print("   1. Temp1 = X - mu")
    print("   2. Temp2 = Temp1 / std")
    print("   3. Temp3 = gamma * Temp2")
    print("   4. Final = Temp3 + beta")
    print("   If your batch is 2GB, this simple equation just allocated 8GB of VRAM!")

if __name__ == "__main__":
    batchnorm_intuition()