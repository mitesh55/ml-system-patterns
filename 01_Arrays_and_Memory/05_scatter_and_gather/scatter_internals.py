import torch

def scatter_intuition():
    print("--- Memory Routing: Demystifying scatter_ ---\n")
    
    # 1. The Setup: A batch of 5 token IDs
    tokens = torch.tensor([2, 0, 4, 1, 2])
    vocab_size = 5
    batch_size = len(tokens)
    
    print(f"Input Token IDs: {tokens.tolist()}")
    
    # =====================================================================
    # 2. THE DIMENSION REQUIREMENT
    # =====================================================================
    print("\n❓ QUESTION: Why does `scatter_` throw dimensionality errors?")
    print("💡 INTUITION: The Index tensor must have the exact same number of dimensions as the Target tensor.")
    
    # Reshape (5,) into (5, 1) so it maps cleanly to the 2D canvas.
    index_2d = tokens.unsqueeze(1) 
    print(f"-> Reshaped Index Tensor:\n{index_2d.tolist()}")
    
    # =====================================================================
    # 3. THE CANVAS & THE ROUTING
    # =====================================================================
    canvas = torch.zeros(batch_size, vocab_size, dtype=torch.int32)
    
    print("\n-------------------------------------------------------------------")
    print("❓ QUESTION: What does `canvas.scatter_(dim=1, index, value=1)` actually do?")
    print("💡 INTUITION: The Hardware Postal Worker")
    print("   `dim=1`   : Move horizontally across columns.")
    print("   `index`   : Read the specific column destination from this tensor.")
    print("   `value=1` : The constant scalar being injected.")
    print("   *(Note: PyTorch strictly separates `value=` for scalars and `src=` for tensors)*\n")
    
    # The actual C++ execution (Fixed keyword: value=1)
    canvas.scatter_(dim=1, index=index_2d, value=1)
    
    # Recreate the routing logic to prove the math
    print("--- The Hardware Routing Map ---")
    for row in range(batch_size):
        col_address = index_2d[row][0].item()
        print(f"Thread {row}: Routed value '1' -> Memory Coordinate [Row {row}, Col {col_address}]")
        
    print("\n--- Final One-Hot Matrix ---")
    print(canvas)
    
    # =====================================================================
    # 4. VERIFICATION
    # =====================================================================
    native_one_hot = torch.nn.functional.one_hot(tokens, num_classes=vocab_size)
    if torch.equal(canvas, native_one_hot):
        print("\n✅ SUCCESS: Our scatter_ implementation perfectly matches F.one_hot().")
        print("   You have successfully reverse-engineered the PyTorch functional API.")

if __name__ == "__main__":
    scatter_intuition()