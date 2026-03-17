import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_memory_layouts():
    print("--- 📊 Generating Memory Visualizations ---")
    
    # 1. Create dummy sequences (4 users with different length prompts)
    lengths = [8, 3, 6, 2]
    # We use different values to represent different users visually (User 0=10, User 1=20, etc.)
    sequences = [torch.full((l,), (i+1)*10, dtype=torch.float32) for i, l in enumerate(lengths)]
    
    # =====================================================================
    # [A] 2D Padded Matrix (The Waste)
    # =====================================================================
    padded_2d = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # =====================================================================
    # [B] 1D Packed Sequence (The Efficiency)
    # =====================================================================
    packed_1d = torch.cat(sequences, dim=0).unsqueeze(0) # Unsqueeze just for 2D plotting
    cu_seqlens = torch.cumsum(torch.tensor([0] + lengths), dim=0).tolist()

    # =====================================================================
    # Plotting
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot A: The Padded Matrix
    cax1 = ax1.matshow(padded_2d.numpy(), cmap='viridis', vmin=0, vmax=40)
    ax1.set_title("2D Padded Batch (Notice the wasted zero-padding)", pad=15, fontweight='bold')
    ax1.set_ylabel("Batch (Users)")
    ax1.set_xlabel("Sequence Length (Time)")
    ax1.set_yticks(range(len(lengths)))
    ax1.set_yticklabels([f"User {i}" for i in range(len(lengths))])
    
    # Plot B: The 1D Packed Array
    cax2 = ax2.matshow(packed_1d.numpy(), cmap='viridis', vmin=0, vmax=40)
    ax2.set_title("1D Packed Sequence (100% Dense, Zero Waste)", pad=15, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlabel("Flat Memory Address (Total Tokens)")
    
    # Draw the "Brick Walls" (cu_seqlens)
    for fence in cu_seqlens:
        ax2.axvline(x=fence - 0.5, color='red', linestyle='--', linewidth=2)
        if fence < cu_seqlens[-1]: # Add text labels
            ax2.text(fence, -0.6, f"cu_len:{fence}", color='red', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("padding_vs_packing.png", dpi=300)
    print("✅ Saved visualization to 'padding_vs_packing.png'. Open it to see the VRAM waste!")

if __name__ == "__main__":
    plot_memory_layouts()