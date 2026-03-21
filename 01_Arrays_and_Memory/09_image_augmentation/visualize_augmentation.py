import matplotlib.pyplot as plt
import numpy as np
import os

# Create visual directory if it doesn't exist
os.makedirs('visuals', exist_ok=True)

# Use a professional dark theme style for an ML engineering aesthetic
plt.style.use('dark_background')
# Custom aesthetic colors: Red (Waste), Green (Optimal), Blue (Information)
COLORS = {'red': '#ff6b6b', 'green': '#1dd1a1', 'blue': '#54a0ff'}

# --- DATA FROM LOGS ---
# Batch=16, H=1024, W=1024
BASE_VRAM_MB = 192.00

# Part 1: CutMix
CUTMIX_NOVICE_SPIKE = 640.00
CUTMIX_OPTIMAL_SPIKE = 12.00

# Part 2: Flips & Conv2d
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

    # Define bars
    methods = ['Base Image Batch\n(No Augmentation)', 
               '[A] Algebraic Masking\n(Naive Way)', 
               '[B] Batched In-Place Roll\n(Optimal Way)']
    vram_used = [BASE_VRAM_MB, 
                 BASE_VRAM_MB + CUTMIX_NOVICE_SPIKE, 
                 BASE_VRAM_MB + CUTMIX_OPTIMAL_SPIKE]
    
    # Define colors for the scenario
    bar_colors = [COLORS['blue'], COLORS['red'], COLORS['green']]

    # Plot bars
    bars = ax.bar(methods, vram_used, color=bar_colors, edgecolor='white', width=0.6)

    # Styling
    ax.set_ylabel('Total Allocated VRAM (MB)', fontsize=13, fontweight='bold', color='white')
    ax.set_title('🗑️ Pattern 09: CutMix Memory Thrash (CVPR 2026)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Increase y-axis limit to accommodate labels
    ax.set_ylim(0, max(vram_used) * 1.15)

    # Add data labels on top of bars
    for bar, spike, method in zip(bars, [0, CUTMIX_NOVICE_SPIKE, CUTMIX_OPTIMAL_SPIKE], methods):
        height = bar.get_height()
        
        # Label Total VRAM
        ax.text(bar.get_x() + bar.get_width() / 2, height + 20,
                f'{height:.0f} MB',
                ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')
        
        # Label the specific SPIKE amount for novice/optimal methods
        if spike > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height - (height*0.2),
                    f'+{spike:.0f} MB\nSPIKE',
                    ha='center', va='center', fontsize=11, color='black', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    save_plot('01_cutmix_memory_trap.png')

# =====================================================================
# PLOT 2: The High-Res Reality Check (Augmentation vs Conv Activation)
# =====================================================================
def plot_high_res_reality():
    print("Generating High-Res Reality Check Visualization...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define bars to show the scale difference
    scenarios = ['Base Batch\n(192MB Input)', 
                 'torch.flip()\n(Forces Copy)', 
                 'Conv2d Activation\n(3 -> 64 Channels)']
    spikes = [0, FLIP_ALLOCATION_SPIKE, CONV2D_ACTIVATION_SPIKE]
    
    # We use blue because these are system facts, not necessarily "waste"
    bar_colors = [COLORS['blue'], COLORS['blue'], '#ff9f43'] # Orange for the massive one

    bars = ax.bar(scenarios, spikes, color=bar_colors, edgecolor='white', width=0.6)

    # Styling
    ax.set_ylabel('VRAM Memory Spike (MB)', fontsize=13, fontweight='bold', color='white')
    ax.set_title('⚠️ Pattern 09: The High-Res Reality Check', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Accommodate the massive label
    ax.set_ylim(0, CONV2D_ACTIVATION_SPIKE * 1.15)

    # Add data labels
    for bar, spike in zip(bars, spikes):
        height = bar.get_height()
        if spike == 0: continue
            
        # If it's the massive activation, put the label inside the bar
        if spike > 1000:
             ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
                    f'+{spike:,.0f} MB\n(4.1 GB)',
                    ha='center', va='center', fontsize=14, color='black', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 50,
                    f'+{spike:.0f} MB',
                    ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')

    # Custom Annotation explaining why
    ax.text(1.5, CONV2D_ACTIVATION_SPIKE * 0.8, 
            "Systems Insight:\nIn High-Res Vision, memory bottlenecks are\nACTIVATIONS (feature maps), not model weights.\nResNet-50 layer 1 explodes memory 21x.",
            fontsize=11, color='white', bbox=dict(facecolor='#2c3e50', alpha=0.8, boxstyle='round,pad=0.5'))

    save_plot('02_high_res_bottleneck.png')

if __name__ == "__main__":
    plot_cutmix()
    plot_high_res_reality()
    print("\nVisuals ready for README.md in the 'visuals/' folder.")