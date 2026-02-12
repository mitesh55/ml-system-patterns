import numpy as np

def check_memory_sharing():
    print(f"Advanced Indexing : View vs. Copy (The silent memory killer) -- \n")

    # large matrix (100MB)
    original = np.zeros((1000,1000))

    print(f"Original Adress : {hex(id(original))}")

    # Case - 1 : Slicing 
    # Slicing creates a View. It points to the Same Memory.
    sliced_view = original[0:500]
    print("\n[Case 1] Slicing `original[0:500]`")
    if np.shares_memory(original, sliced_view):
        print(f"✅ Result: VIEW (Shares Memory)")
        print("   -> Modifying this WILL affect the original array.")
    else:
        print("❌ Result: COPY (New Memory)")

    # Case 2 : Fancy Indexing 
    # Using list of indices creates a new copy
    # This is expensive but necessary if indices are not contiguos.
    indices = [0,1,2,7,5]
    fancy_copy = original[indices]

    print("\n[Case 2] Fancy Indexing `original[[0, 1, 2...]]`")
    if np.shares_memory(original, fancy_copy):
        print("✅ Result: VIEW (Shares Memory)")
    else:
        print("⚠️ Result: COPY (New Memory Allocation!)")
        print("   -> Modifying this will NOT affect the original array.")
        print("   -> 🚨 WARNING: This doubled your memory usage for these rows.")

    # Case 3 : Boolean Masking 
    # Masking also creates a COPY of the selected data.
    mask = original > 3
    # print(f" Mask : {mask}")
    masked_copy = original[mask]

    print("\n[Case 3] Boolean Masking `original[mask]`")
    if np.shares_memory(original, masked_copy):
        print("✅ Result: VIEW")
    else:
        print("⚠️ Result: COPY") 
        print("   -> The result is a compacted array, so it MUST be a copy.")
if __name__ == "__main__":
    check_memory_sharing()