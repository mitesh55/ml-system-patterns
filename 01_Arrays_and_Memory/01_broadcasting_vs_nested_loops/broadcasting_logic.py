from dataclasses import dataclass
from typing import List, Tuple

Matrix = List[List[int]]

def shape(mat: Matrix) -> Tuple[int, int]:
    if not mat or not mat[0]:
        return (0, 0)
    return (len(mat), len(mat[0]))


@dataclass
class BroadcastView:
    base: Matrix
    base_shape: Tuple[int, int]
    target_shape: Tuple[int, int]

    def get(self, i: int, j: int) -> int:
        """
        Returns the element at (i, j) in the *broadcasted* view
        without copying the base matrix.
        """
        br, bc = self.base_shape
        tr, tc = self.target_shape

        # If base dim is 1, it repeats => index must map to 0 (stride=0 behavior)
        bi = 0 if br == 1 else i
        bj = 0 if bc == 1 else j

        return self.base[bi][bj]


def broadcast_shapes(a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> Tuple[int, int]:
    ar, ac = a_shape
    br, bc = b_shape

    # Check compatibility and determine target row
    if ar == br:
        tr = ar
    elif ar == 1:
        tr = br
    elif br == 1:
        tr = ar
    else:
        raise ValueError(f"Incompatible row dims: {a_shape} vs {b_shape}")

    # Check compatibility and determine target col
    if ac == bc:
        tc = ac
    elif ac == 1:
        tc = bc
    elif bc == 1:
        tc = ac
    else:
        raise ValueError(f"Incompatible col dims: {a_shape} vs {b_shape}")

    return (tr, tc)


def add_with_broadcast(A: Matrix, B: Matrix) -> Matrix:
    a_shape = shape(A)
    b_shape = shape(B)

    target = broadcast_shapes(a_shape, b_shape)
    print(f"target : {target}")
    a_view = BroadcastView(A, a_shape, target)
    print(f"a_view : {a_view}")
    b_view = BroadcastView(B, b_shape, target)
    print(f"b_view : {b_view}")

    tr, tc = target
    C = [[0] * tc for _ in range(tr)]

    for i in range(tr):
        for j in range(tc):
            
            C[i][j] = a_view.get(i, j) + b_view.get(i, j)

    return C


# ---- Demo ----
A = [[10], [20]]   # (2,1)
B = [[5]]          # (1,1)

print(add_with_broadcast(A, B))
# Output: [[15], [25]]
