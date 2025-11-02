import numpy as np


def SwapRows(A, L):
    # flip rows around
    b = A.copy()
    b[[L[0], L[1]]] = b[[L[1], L[0]]]
    return b


def MulRow(A, r, c):
    # scale row by c
    b = A.copy()
    b[r] *= c
    return b


def AddMul(A, L, c):
    # row combo: add c*row to another
    b = A.copy()
    b[L[1]] += c * b[L[0]]
    return b


def Pivot(A, L):
    # do the pivot thing
    if L[0] >= A.shape[0] or L[1] >= A.shape[1] or A[L[0], L[1]] == 0:
        return 0  # nope, can't pivot on zero or out of bounds
    b = A.copy().astype(float)
    # make pivot = 1
    b[L[0]] /= b[L[0], L[1]]
    # zero out column
    for i in range(b.shape[0]):
        if i != L[0]:
            b[i] -= b[i, L[1]] * b[L[0]]
    return b


def rref(A):
    # get that sweet rref + rank
    b, r, rank = A.copy().astype(float), A.shape[0], 0
    for c in range(min(A.shape)):
        # hunt for pivot row
        pRow = next((row for row in range(rank, r)
                    if abs(b[row, c]) > 1e-10), None)
        if pRow is not None:
            # swap & pivot
            b[[rank, pRow]] = b[[pRow, rank]] if pRow != rank else b[[rank, pRow]]
            piv = Pivot(b, (rank, c))
            if isinstance(piv, np.ndarray):
                b, rank = piv, rank + 1
    return (b, rank)


def PoolMatrix(A, f, type=0):
    # pooling magic: max or avg
    # type=0 = max, type=1 = avg
    return np.array([[np.max(A[i:i+f, j:j+f]) if type == 0 else np.mean(A[i:i+f, j:j+f]) for j in range(A.shape[1] - f + 1)] for i in range(A.shape[0] - f + 1)])
