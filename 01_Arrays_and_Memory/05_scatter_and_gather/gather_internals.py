import torch

def gather_intuition():
    print("--- Memory Routing: Demystifying gather ---\n")
    
    # 1. The Setup: LLM Logits (Probabilities)
    # Batch Size = 3, Vocab Size = 5
    # These are the probabilities the model assigned to each word in the vocab.
    logits = torch.tensor([
        [0.1, 0.2, 0.5, 0.1, 0.1],  # Model predicts word 2 (0.5)
        [0.8, 0.0, 0.1, 0.0, 0.1],  # Model predicts word 0 (0.8)
        [0.1, 0.1, 0.1, 0.6, 0.1]   # Model predicts word 3 (0.6)
    ])
    
    # These are the ACTUAL correct words (Ground Truth)
    true_tokens = torch.tensor([2, 4, 3]) 
    
    print(f"Logits (Predictions):\n{logits}")
    print(f"\nGround Truth Tokens: {true_tokens.tolist()}")
    
    # =====================================================================
    # 2. THE DIMENSION REQUIREMENT
    # =====================================================================
    # Just like scatter, the index must match the target's dimensions.
    index_2d = true_tokens.unsqueeze(1)
    
    # =====================================================================
    # 3. THE GATHERING
    # =====================================================================
    print("\n-------------------------------------------------------------------")
    print("❓ QUESTION: What does `torch.gather(logits, dim=1, index)` actually do?")
    print("💡 INTUITION: The Hardware Claw Machine")
    print("   `dim=1` : Move horizontally across columns to grab the prize.")
    print("   `index` : Read the specific column coordinate from this tensor.")
    print("   The C++ backend executes: result[row][0] = logits[row][ index[row][0] ]\n")
    
    # The actual C++ execution
    gathered_probs = torch.gather(input=logits, dim=1, index=index_2d)
    
    # Recreate the routing logic to prove the math
    print("--- The Hardware Read Map ---")
    for row in range(len(true_tokens)):
        col_address = index_2d[row][0].item()
        val = logits[row][col_address].item()
        print(f"Thread {row}: Read coordinate [Row {row}, Col {col_address}] -> Grabbed Probability: {val:.1f}")
        
    print(f"\n--- Final Gathered Probabilities ---")
    print(gathered_probs.squeeze().tolist())

if __name__ == "__main__":
    gather_intuition()