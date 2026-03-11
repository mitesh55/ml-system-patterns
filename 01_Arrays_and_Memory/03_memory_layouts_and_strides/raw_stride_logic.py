import numpy as np

class MockTensor:
    def __init__(self, storage, shape, strides, offset=0):
        self.storage = storage  
        self.shape = shape      
        self.strides = strides  
        self.offset = offset    

    def get_item(self, indices):
        """Calculates the physical 1D RAM index using the stride formula."""
        memory_index = self.offset
        for i, idx in enumerate(indices):
            memory_index += idx * self.strides[i]
        return self.storage[memory_index]

    def transpose(self, dim0, dim1):
        """
        Creates a transposed view by swapping the shapes and strides of two dimensions.
        Zero memory is copied.
        """
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        
        # Swap the metadata
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
        
        return MockTensor(self.storage, tuple(new_shape), tuple(new_strides), self.offset)

    def is_contiguous(self):
        """Checks if the logical layout matches the sequential physical layout."""
        expected_stride = 1
        for i in reversed(range(len(self.shape))):
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]
        return True

    def data_ptr(self):
        return hex(id(self.storage))

    def to_list(self):
        """Recursively builds nested lists to visualize the logical tensor shape."""
        def build(dim, current_indices):
            # Base case: we reached the exact element
            if dim == len(self.shape):
                return self.get_item(current_indices)
            
            # Recursive case: build the nested structure
            result = []
            for i in range(self.shape[dim]):
                result.append(build(dim + 1, current_indices + [i]))
            return result
        
        return build(0, [])

    def __str__(self):
        """Renders the tensor exactly like PyTorch does."""
        formatted_list = str(self.to_list()).replace("],", "],\n ")
        return f"MockTensor(\n {formatted_list}\n)"


if __name__ == "__main__":
    print("--- 1. Physical Memory (The Storage) ---")
    # A standard block of contiguous memory
    raw_data = np.arange(10) 
    base_tensor = MockTensor(storage=raw_data, shape=(10,), strides=(1,))
    
    print(f"Base Data: {base_tensor.to_list()}")
    print(f"Base Address: {base_tensor.data_ptr()}\n")

    # -----------------------------------------------------------------
    print("--- 2. The Sliding Window (Zero-Copy View) ---")
    # Emulating: torch.as_strided(data, size=(8, 3), stride=(1, 1))
    
    sliding_window_view = MockTensor(
        storage=raw_data, 
        shape=(8, 3), 
        strides=(1, 1), 
        offset=0
    )
    
    print(sliding_window_view)
    print(f"View Address:  {sliding_window_view.data_ptr()} (Same as Base)\n")

    # -----------------------------------------------------------------
    print("--- 3. The Contiguous Trap (Transposing) ---")
    # Step 3a: Create a standard 2x3 matrix
    # To go down a row (dim 0), we jump 3 indices. To go right (dim 1), we jump 1 index.
    matrix_2x3 = MockTensor(
        storage=raw_data,
        shape=(2, 3),    
        strides=(3, 1),  
        offset=0
    )
    
    print("Original 2x3 Matrix:")
    print(matrix_2x3)
    print(f"Strides: {matrix_2x3.strides} | Contiguous: {matrix_2x3.is_contiguous()}\n")

    # Step 3b: Dynamically transpose it (Swap Dim 0 and Dim 1)
    matrix_3x2 = matrix_2x3.transpose(0, 1)
    
    print("Transposed 3x2 Matrix (Zero Copy):")
    print(matrix_3x2)
    print(f"Strides: {matrix_3x2.strides} | Contiguous: {matrix_3x2.is_contiguous()}")
    
    if not matrix_3x2.is_contiguous():
        print("\n⚠️  Warning: The logical columns (0, 3, 6...) are no longer physically adjacent in RAM.")
        print("Passing this view to a custom CUDA kernel will cause cache misses.")