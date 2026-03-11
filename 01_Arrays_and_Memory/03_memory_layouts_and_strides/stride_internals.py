import torch

def check_tensor_illusions():
    print("--- Memory Layouts: Tensors are Illusions ---\n")

    # 1. Create a 2D tensor (under the hood, it's 1D RAM)
    A = torch.tensor([[1, 2, 3], 
                      [4, 5, 6]])
    
    print(f"Original Shape: {A.shape}")
    print(f"Original Strides: {A.stride()}")
    print(f"Memory Address (data_ptr): {hex(A.data_ptr())}\n")

    # 2. Transpose the tensor (swap rows and columns)
    A_T = A.t()
    
    print(f"Transposed Shape: {A_T.shape}")
    print(f"Transposed Strides: {A_T.stride()}")
    print(f"Transposed Memory Address: {hex(A_T.data_ptr())}")
    
    # Prove the memory hasn't moved
    if A.data_ptr() == A_T.data_ptr():
        print("✅ The memory address is EXACTLY the same. Zero bytes copied.")
        print("   -> PyTorch just swapped the strides to trick the math engine.")

    # 3. The Contiguous Trap
    print("\n--- The Contiguous Trap ---")
    print(f"Is Original contiguous? {A.is_contiguous()}")
    print(f"Is Transposed contiguous? {A_T.is_contiguous()}")
    
    if not A_T.is_contiguous():
        print("🚨 WARNING: Transposed tensor is NOT contiguous.")
        print("   -> If you send this to a custom CUDA kernel, it will cause massive cache misses.")
        print("   -> You must call .contiguous() before low-level C++ ops, which WILL force a memory copy.")

if __name__ == "__main__":
    check_tensor_illusions()