import torch
from torch.utils.data import DataLoader

def analyze_collate_and_padding():
    print("--- 🧩 Pattern 07: Variable-Length Sequences & Collation ---\n")
    # 1. The Real-World Scenario: Variable length NLP tokens
    # User A types a 10-word prompt. User B types a 3-word prompt.

    seq_A = torch.tensor([101, 54, 89, 23, 11, 88, 92, 44, 12, 102]) # 10
    seq_B = torch.tensor([101,77,102]) # 3
    seq_C = torch.tensor([200,30,100,56]) # 4

    mock_batch = [seq_A, seq_B, seq_C]

    # =====================================================================
    # 2. THE NAIVE DATALOADER CRASH
    # =====================================================================
    print("🚨 THE SHAPE MISMATCH CRASH:")
    print("   If you pass variable-length arrays to a default DataLoader, PyTorch uses `torch.stack`.")
    print("   `torch.stack` tries to create a perfect 2D matrix, but lengths [10, 3, 5] don't fit.")
    
    try :
        torch.utils.data.default_collate(mock_batch)

    except Exception as e:
        print(f"   ❌ PyTorch Error: {e}\n")

    # =====================================================================
    # 3. THE PYTHONIC FIX: Custom Collate & Padding
    # =====================================================================
    print("🛠️ THE FIX: Writing a custom `collate_fn` to pad with zeros.")

    def pad_collate_fn(batch):
        # batch is a list of 1D tensors.
        # pad_sequence finds the max length in the batch and adds 0s to the rest.
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return padded
    
    # Pass our custom function to the DataLoader
    loader = DataLoader(mock_batch, batch_size=3, collate_fn=pad_collate_fn)
    padded_batch = next(iter(loader))
    
    print(f"\n✅ Padded Batch Shape: {padded_batch.shape}")
    print(padded_batch)

    # =====================================================================
    # 4. THE HARDWARE REALITY: Ghost Compute
    # =====================================================================
    print("\n👻 THE SYSTEMS PROBLEM: 'Ghost Compute'")
    total_elements = padded_batch.numel()
    real_elements = sum([len(seq) for seq in mock_batch])
    padding_elements = total_elements - real_elements
    waste_percentage = (padding_elements / total_elements) * 100

    print(f"   Total Elements in Matrix: {total_elements}")
    print(f"   Actual Token Elements:    {real_elements}")
    print(f"   Useless Padding (0s):     {padding_elements}")
    print(f"   Hardware Waste:           {waste_percentage:.1f}% of VRAM and Compute is doing absolutely nothing.")
    print("   In a standard LLM Matrix Multiplication (GEMM), the GPU will still multiply those zeros!")

if __name__ == "__main__":
    analyze_collate_and_padding()