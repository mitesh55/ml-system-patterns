import torch
import torch.nn as nn
import time

def benchmark_augmentations():
    print("--- 🖼️ Pattern 09: CV Augmentation Memory Traps ---\n")
    
    # Simulate a batch of High-Res images (Batch=16, Channels=3, H=1024, W=1024)
    B, C, H, W = 16, 3, 1024, 1024
    print(f"Simulating Image Batch: [Batch={B}, Channels={C}, Height={H}, Width={W}]")
    
    # Base Memory: 16 * 3 * 1024 * 1024 * 4 bytes = 192 MB
    img_batch = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
    
    base_vram = torch.cuda.memory_allocated() / (1024**2)
    print(f"Base VRAM (Data Loaded): {base_vram:.2f} MB\n")

    # =====================================================================
    # [PART 1] THE CUTMIX TRAP: Algebraic Masking vs In-Place
    # =====================================================================
    print("--- ✂️ Part 1: The CutMix Memory Trap ---")
    
    box_size = 256
    y1, x1 = 384, 384
    y2, x2 = y1 + box_size, x1 + box_size
    rolled_indices = torch.roll(torch.arange(B), shifts=1)

    # --- [A] The Novice Way: Algebraic Masking ---
    # Creates a binary mask (1s outside the box, 0s inside)
    mask = torch.ones(B, 1, H, W, device='cuda', dtype=torch.float32)
    mask[:, :, y1:y2, x1:x2] = 0

    torch.cuda.reset_peak_memory_stats()
    vram_before_mask = torch.cuda.memory_allocated() / (1024**2)
    
    start = time.time()
    # The algebraic calculation allocates massive intermediate tensors
    cutmix_novice = (img_batch * mask) + (img_batch[rolled_indices] * (1 - mask))
    torch.cuda.synchronize()
    
    time_mask = (time.time() - start) * 1000
    waste_mask = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_mask
    
    print("[A] CutMix via Algebraic Masking:")
    print(f"    Time:      {time_mask:.3f} ms")
    print(f"    Spike:     +{waste_mask:.2f} MB (Allocates intermediate math tensors & new output!)\n")
    
    del cutmix_novice, mask
    torch.cuda.empty_cache()

    # --- [B] The Hardware friendly Way: Batched In-Place Roll ---
    torch.cuda.reset_peak_memory_stats()
    img_clone = img_batch.clone() 
    vram_before_inplace = torch.cuda.memory_allocated() / (1024**2)
    
    start = time.time()
    # Direct memory overwrite using the rolled indices
    img_clone[:, :, y1:y2, x1:x2] = img_clone[rolled_indices, :, y1:y2, x1:x2]
    torch.cuda.synchronize()
    
    time_inplace = (time.time() - start) * 1000
    waste_inplace = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_inplace

    print("[B] CutMix via Batched In-Place Roll:")
    print(f"    Time:      {time_inplace:.3f} ms")
    print(f"    Spike:     +{waste_inplace:.2f} MB (Zero allocations. Direct pointer memory swap.)\n")

# =====================================================================
    # [PART 2] THE FLIP TRAP: Negative Strides vs Contiguous Memory
    # =====================================================================
    print("--- 🔄 Part 2: The Horizontal Flip Contiguity Trap ---")
    
    dummy_conv = nn.Conv2d(3, 64, kernel_size=3).cuda()
    
    # 1. The "Free" Flip
    vram_before_flip = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    # Use PyTorch's explicit flip (changes metadata, shares storage)  bcz doing with slicing is restricted in pytorch.  
    flipped_batch = torch.flip(img_batch, dims=[3])   
    
    waste_flip = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_flip
    print(f"[C] Pythonic Flip `torch.flip()`:")
    print(f"    Spike:     +{waste_flip:.2f} MB (It's an illusion. Only metadata changed.)")
    print(f"    Contiguous? {flipped_batch.is_contiguous()}")

    # 2. The Conv2d Crash
    vram_before_conv = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    # Forward Pass
    _ = dummy_conv(flipped_batch)
    torch.cuda.synchronize()
    
    waste_conv = (torch.cuda.max_memory_allocated() / (1024**2)) - vram_before_conv
    
    print("\n[D] Forward Pass into Conv2d (The Silent Killer):")
    print(f"    Spike:     +{waste_conv:.2f} MB")
    print("    Intuition: The Conv2d C++ kernel requires forward-reading contiguous memory.")
    print("               It panicked at the negative stride and silently forced a massive `.contiguous()`")
    print("               copy of your entire batch right before the math started!")

if __name__ == "__main__":
    benchmark_augmentations()