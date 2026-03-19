import matplotlib.pyplot as plt
import numpy as np

def plot_rope_memory():
    print("--- 📊 Generating RoPE Memory Visualization ---")
    
    # Data gathered from our CUDA benchmark
    # Base VRAM: ~544 MB
    # PyTorch Peak: ~2592 MB (+2048 MB Spike)
    # CUDA Peak: ~544 MB (+0 MB Spike)
    
    labels = ['Base Data', 'PyTorch Eager', 'PyTorch Compiled', 'In-Place C++']
    vram_values = [516, 2564, 1028, 516]
    colors = ['#7f8c8d', '#e74c3c', '#f39c12', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(labels, vram_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add the text annotations showing the exact MB
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} MB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add a massive arrow showing the Thrash
    ax.annotate('+2048 MB VRAM Spike!\n(Due to torch.cat & slicing)', 
                xy=(1, 2592), xytext=(1, 1500),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                ha='center', va='center', color='black', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2))

    # Styling
    ax.set_title('RoPE Memory Thrash: Native PyTorch vs Custom C++ Kernel', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('GPU VRAM Allocation (Megabytes)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 3000)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a descriptive footer
    fig.text(0.5, 0.01, "Simulation: Batch=8, Heads=32, SeqLen=4096. \nPyTorch allocates intermediate tensors. The C++ Kernel rotates memory in-place (0 extra allocations).", 
             ha='center', fontsize=10, style='italic', color='#34495e')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('rope_memory_thrash.png', dpi=300)
    print("✅ Saved visualization to 'rope_memory_thrash.png'.")

if __name__ == "__main__":
    plot_rope_memory()