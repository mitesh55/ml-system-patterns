import matplotlib.pyplot as plt
import numpy as np
import os

# Create visual directory if it doesn't exist
os.makedirs('visuals', exist_ok=True)

# Use a professional dark theme style for an ML engineering aesthetic
plt.style.use('dark_background')
# Custom aesthetic colors
COLORS = {'red': '#ff6b6b', 'green': '#1dd1a1', 'blue': '#54a0ff', 'orange': '#ff9f43'}

# --- DATA FROM LOGS ---
BASE_VRAM_MB = 192.00
CUTMIX_NAIVE_SPIKE = 640.00
CUTMIX_OPTIMAL_SPIKE = 12.00
FLIP_ALLOCATION_SPIKE = 192.00
CONV2D_ACTIVATION_SPIKE = 4085.99

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'visuals/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: visuals/{filename}")

# =====================================================================
# PLOT 1: CutMix VRAM Thrash vs Optimization
# =====================================================================
def plot_cutmix():
    print("Generating CutMix Visualization...")
    fig, ax = plt.subplots(figsize=(10, 7))
    methods = ['Base Image Batch\n(No Augmentation)', '[A] Algebraic Masking\n(Naive Way)', '[B] Batched In-Place Roll\n(Optimal Way)']
    vram_used = [BASE_VRAM_MB, BASE_VRAM_MB + CUTMIX_NAIVE_SPIKE, BASE_VRAM_MB + CUTMIX_OPTIMAL_SPIKE]
    bar_colors = [COLORS['blue'], COLORS['red'], COLORS['green']]
    
    bars = ax.bar(methods, vram_used, color=bar_colors, edgecolor='white', width=0.6)
    ax.set_ylabel('Total Allocated VRAM (MB)', fontsize=13, fontweight='bold', color='white')
    ax.set_title('🗑️ Pattern 09: CutMix Memory Thrash (CVPR 2026)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(vram_used) * 1.15)

    for bar, spike, method in zip(bars, [0, CUTMIX_NAIVE_SPIKE, CUTMIX_OPTIMAL_SPIKE], methods):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 20, f'{height:.0f} MB', ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')
        if spike > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height - (height*0.2), f'+{spike:.0f} MB\nSPIKE', ha='center', va='center', fontsize=11, color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    save_plot('01_cutmix_memory_trap.png')

# =====================================================================
# PLOT 2: The High-Res Reality Check 
# =====================================================================
def plot_high_res_reality():
    print("Generating High-Res Reality Check Visualization...")
    fig, ax = plt.subplots(figsize=(10, 7))
    scenarios = ['Base Batch\n(192MB Input)', 'torch.flip()\n(Forces Copy)', 'Conv2d Activation\n(3 -> 64 Channels)']
    spikes = [0, FLIP_ALLOCATION_SPIKE, CONV2D_ACTIVATION_SPIKE]
    bar_colors = [COLORS['blue'], COLORS['blue'], COLORS['orange']]
    
    bars = ax.bar(scenarios, spikes, color=bar_colors, edgecolor='white', width=0.6)
    ax.set_ylabel('VRAM Memory Spike (MB)', fontsize=13, fontweight='bold', color='white')
    ax.set_title('⚠️ Pattern 09: The High-Res Reality Check', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, CONV2D_ACTIVATION_SPIKE * 1.15)

    for bar, spike in zip(bars, spikes):
        height = bar.get_height()
        if spike == 0: continue
        if spike > 1000:
             ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'+{spike:,.0f} MB\n(4.1 GB)', ha='center', va='center', fontsize=14, color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 50, f'+{spike:.0f} MB', ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')

    ax.text(1.5, CONV2D_ACTIVATION_SPIKE * 0.8, "Systems Insight:\nIn High-Res Vision, memory bottlenecks are\nACTIVATIONS (feature maps), not model weights.\nResNet-50 layer 1 explodes memory 21x.", fontsize=11, color='white', bbox=dict(facecolor='#2c3e50', alpha=0.8, boxstyle='round,pad=0.5'))
    save_plot('02_high_res_bottleneck.png')

# =====================================================================
# PLOT 3: The "Memory Tax" Timeline (Early vs Late)
# =====================================================================
def plot_memory_tax_timeline():
    print("Generating Memory Tax Timeline Visualization...")
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = ['Negative Stride [::-1]\n(Pays Tax LATER)', 
               'Algebraic Masking\n(Pays Tax EARLY)', 
               'In-Place Roll\n(Pays NO Tax)']

    # Phase 1: The Augmentation Step
    early_tax = [0, CUTMIX_NAIVE_SPIKE, CUTMIX_OPTIMAL_SPIKE]
    # Phase 2: The Forward Pass 
    late_tax = [FLIP_ALLOCATION_SPIKE, 0, 0]

    # Create Stacked Bars
    p1 = ax.bar(methods, early_tax, color=COLORS['red'], edgecolor='white', width=0.5, label='Early Tax (Allocated during Augmentation)')
    p2 = ax.bar(methods, late_tax, bottom=early_tax, color=COLORS['orange'], edgecolor='white', width=0.5, label='Late Tax (Forced Copy during Forward Pass)')

    # Override the In-Place color to green to show it is the optimal path
    p1[2].set_color(COLORS['green'])
    p1[2].set_edgecolor('white')

    # Styling
    ax.set_ylabel('Wasted VRAM (MB)', fontsize=13, fontweight='bold', color='white')
    ax.set_title('⏳ Pattern 09: The Memory Tax Timeline', fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Legend formatting
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=12, labelcolor='white')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 800)

    # Add Text Labels to the bars
    # Flip Bar (Late Tax)
    ax.text(0, FLIP_ALLOCATION_SPIKE + 20, f'+{FLIP_ALLOCATION_SPIKE:.0f} MB\n(Contiguity Crash)', ha='center', va='bottom', fontsize=11, color=COLORS['orange'], fontweight='bold')
    ax.text(0, 20, '0 MB\n(Metadata)', ha='center', va='bottom', fontsize=11, color='white')

    # Algebraic Bar (Early Tax)
    ax.text(1, CUTMIX_NAIVE_SPIKE + 20, f'+{CUTMIX_NAIVE_SPIKE:.0f} MB\n(Math Tensors)', ha='center', va='bottom', fontsize=11, color=COLORS['red'], fontweight='bold')

    # In-Place Bar (Optimal)
    ax.text(2, CUTMIX_OPTIMAL_SPIKE + 20, f'+{CUTMIX_OPTIMAL_SPIKE:.0f} MB\n(Pointer Swap)', ha='center', va='bottom', fontsize=11, color=COLORS['green'], fontweight='bold')

    save_plot('03_memory_tax_timeline.png')

if __name__ == "__main__":
    plot_cutmix()
    plot_high_res_reality()
    plot_memory_tax_timeline()
    print("\nVisuals ready for README.md in the 'visuals/' folder.")