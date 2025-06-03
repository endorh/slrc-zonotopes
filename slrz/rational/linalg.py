"""
Basic rational linear algebra,
implemented with `Fraction` numpy n-dimensional arrays (of dtype=object).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, NamedTuple, TypeVar

import numpy as np

from slrz.util.optional_numba import njit

__all__ = [
    'as_fraction_array',
    'pLU_decomposition',
    'pLUq_decomposition',
    'row_hnf',
    'col_hnf',
    'determinant',
    'solve',
    'inverse',
    'adjugate',
    'rank',
    'is_full_rank',
    'update_inverse',
    'update_adjugate',

    'PLUDecomposition',
    'PLUQDecomposition',
    'RowHNFDecomposition',
    'ColumnHNFDecomposition',
]


# Type Variables
Shape = TypeVar("Shape", bound=tuple)
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
MatrixOrVectorShape = tuple[N, int] | tuple[N]
T = TypeVar("T", int, Fraction)


def as_fraction_array(a: tuple | list | np.ndarray[Shape, int | float]) -> np.ndarray[Shape, Fraction]:
    """
    Convert array to `dtype=object` with `Fraction` values.

    Accepts tuple or list inputs, which are pre-converted with `np.array`.
    """
    if isinstance(a, (list, tuple)):
        a = np.array(a)
    # The following results in dtype=object
    # It is not possible to check at runtime whether `dtype=Fraction`.
    object_array = a.astype(dtype=Fraction)
    object_array *= Fraction(1)
    return object_array

class PLUDecomposition(NamedTuple):
    """
    PLU decomposition of an integer or rational square nonsingular n×n matrix A
    into matrices P, L and U, such that P@L@U = A, with:

    :ivar P: Orthogonal n×n matrix encoding row swaps taken to arrive at U.
    :ivar L: Lower triangular n×n matrix encoding row eliminations taken to
             arrive at U.
    :ivar U: Upper triangular n×n matrix obtained from A by Gaussian elimination
             with partial row pivoting.

    For convenience, the determinant of P is computed during the decomposition and
    stored as :ivar det_P:.
    """
    P: np.ndarray[tuple[N, N], int]
    L: np.ndarray[tuple[N, N], Fraction]
    U: np.ndarray[tuple[N, N], Fraction]
    det_P: Literal[-1, 1]

def pLU_decomposition(
    a: tuple | list | np.ndarray[tuple[N, N], Fraction], *,
    allow_drop_of_rank_in_last_row: bool = False,
) -> PLUDecomposition:
    """
    PLU decomposition of a square nonsingular matrix, using partial
    row pivoting to avoid zeroes in the diagonal.

    Raises a `ValueError` if the matrix is singular, unless the drop of rank occurs in the last row and
    `allow_drop_of_rank_in_last_row` is set to `True`.

    :param a: (n×n) square nonsingular matrix.
    :return: `PLUDecomposition` named tuple with P, L, U and det(P).
    """
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if np.issubdtype(a.dtype, np.integer):
        a = as_fraction_array(a)
    n = a.shape[0]
    assert a.shape == (n, n), "matrix must be square"

    det_P = 1
    P: np.ndarray[tuple[N, N], int] = np.eye(n, dtype=int)
    L: np.ndarray[tuple[N, N], Fraction] = np.eye(n, dtype=int) * Fraction(1, 1)
    U = a.copy()
    for c in range(n - 1):
        # Pivot rows to avoid zero in diagonal
        if U[c, c] == 0:
            nz, = np.asarray(U[c+1:, c] != 0).nonzero()
            if not len(nz):
                raise ValueError("matrix is singular and has no PLU decomposition")
            r_swap = c + 1 + nz[0]
            P[:, (c, r_swap)] = P[:, (r_swap, c)]
            L[(c, r_swap), :c] = L[(r_swap, c), :c]  # only the first c-1 columns are different from I
            U[(c, r_swap), :] = U[(r_swap, c), :]
            det_P *= -1

        # Clear column below (c, c)
        cc = U[c, c]
        for r in range(c + 1, n):
            rc = U[r, c]
            q = rc / cc
            U[r, :] -= U[c, :] * q
            L[r, c] = q
            assert U[r, c] == 0
    if not allow_drop_of_rank_in_last_row and U[n-1, n-1] == 0:
        raise ValueError("matrix is singular and has no PLU decomposition")
    return PLUDecomposition(P, L, U, det_P)

class PLUQDecomposition(NamedTuple):
    """
    PLUQ decomposition of a n integer or rational m×n matrix A into matrices
    P, L, U and Q, such that P@L@U@Q = A, with:

    :ivar P: Orthogonal n×n matrix encoding row swaps taken to arrive at U.
    :ivar L: Lower triangular n×n matrix encoding row eliminations taken to
             arrive at U.
    :ivar U: Row echelon form of A (and thus, an n×m matrix) obtained by
             Gaussian elimination with full pivoting, left-aligned.
    :ivar Q: Orthogonal m×m matrix encoding column swaps taken to arrive at U.

    For convenience, the determinants of P and Q are computed during the
    decomposition and stored as :ivar det_P: and :ivar det_Q:, and their
    product is stored as :ivar det_PQ:.
    """
    P: np.ndarray[tuple[N, N], int]
    L: np.ndarray[tuple[N, N], Fraction]
    U: np.ndarray[tuple[N, M], Fraction]
    Q: np.ndarray[tuple[M, M], int]
    det_P: Literal[-1, 1]
    det_Q: Literal[-1, 1]
    det_PQ: Literal[-1, 1]

def pLUq_decomposition(a: tuple | list | np.ndarray[tuple[N, M], Fraction]) -> PLUQDecomposition:
    """
    PLUQ decomposition of a 2D matrix, using full pivoting of rows and columns to
    obtain a row echelon form of `a`, `U`.

    :param a: (n×m) 2D matrix.
    :return: `PLUQDecomposition` named tuple with P, L, U, Q, det(P), det(Q), det(P)·det(Q).
    """
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if np.issubdtype(a.dtype, np.integer):
        a = as_fraction_array(a)
    n = a.shape[0]
    m = a.shape[1]
    assert a.shape == (n, m), "matrix must be square"

    P: np.ndarray[tuple[N, N], int] = np.eye(n, dtype=int)
    L: np.ndarray[tuple[N, N], Fraction] = np.eye(n, dtype=int) * Fraction(1, 1)
    U = a.copy()
    Q: np.ndarray[tuple[M, M], int] = np.eye(m, dtype=int)
    det_P, det_Q = 1, 1
    for c in range(min(n - 1, m - 1)):
        # Pivot rows and columns to avoid zeros
        if U[c, c] == 0:
            nz, = np.asarray(U[c+1:, c] != 0).nonzero()
            if not len(nz):
                # A row pivot is insufficient, find column to pivot
                nzr, nzc = np.asarray(U[c:, c+1:] != 0).nonzero()
                if not len(nzc):
                    break  # Row-echelon form reached
                c_swap = c+1 + nzc[0]
                Q[(c, c_swap), :] = Q[(c_swap, c), :]
                U[:, (c, c_swap)] = U[:, (c_swap, c)]
                r_swap = c + nzr[0]
                det_Q *= -1
            else:
                r_swap = c + 1 + nz[0]
            P[:, (c, r_swap)] = P[:, (r_swap, c)]
            L[(c, r_swap), :c] = L[(r_swap, c), :c]  # only the first c-1 columns are different from I
            U[(c, r_swap), :] = U[(r_swap, c), :]
            det_P *= -1

        # Clear column below (c, c)
        cc = U[c, c]
        for r in range(c + 1, n):
            rc = U[r, c]
            q = rc / cc
            U[r, :] -= U[c, :] * q
            L[r, c] = q
            assert U[r, c] == 0
    return PLUQDecomposition(P, L, U, Q, det_P, det_Q, det_P*det_Q)

class RowHNFDecomposition(NamedTuple):
    """
    (Row) Hermite Normal Form decomposition of an n×m matrix A, such that:
        H = U·A
    where H is in row echelon form and has positive pivots, and U is unimodular, hence:
        A = Uᵗ·H
    """
    H: np.ndarray[tuple[N, M], int]
    U: np.ndarray[tuple[N, M], int]

@njit
def _row_hnf(
    A: np.ndarray[tuple[N, M], int],
    reduce_non_pivots: bool,
) -> RowHNFDecomposition:
    """
    Row Hermite Normal Form decomposition of an n×m matrix A, such that:
        H = U·A
    where H is in row echelon form and has positive pivots, and U is unimodular, hence:
        A = Uᵗ·H
    """
    n, m = A.shape
    assert A.shape == (n, m), "matrix must be a 2D array"
    H = A.copy()
    U = np.eye(n, dtype=A.dtype)
    swap_row = np.zeros(max(n, m), dtype=A.dtype)
    i = 0
    j = 0
    while i < n and j < m:
        # Find pivot
        r = i
        p = abs(H[i, j])
        for ii in range(i+1, n):
            pp = abs(H[ii, j])
            if pp != 0 and (p == 0 or pp < p):
                p = pp
                r = ii
        if p == 0:
            j += 1
            continue
        p = H[r, j]

        # Swap rows
        if r > i:
            swap_row[:m] = H[i, :]
            H[i, :] = H[r, :]
            H[r, :] = swap_row[:m]
            swap_row[:n] = U[i, :]
            U[i, :] = U[r, :]
            U[r, :] = swap_row[:n]

        # Reduce rows below
        done = True
        for r in range(i+1, n):
            q = H[r, j] // p
            if q != 0:
                H[r, j:] -= H[i, j:] * q
                U[r, :] -= U[i, :] * q
                if H[r, j] != 0:
                    done = False

        if done:
            # Correct pivot sign
            if p < 0:
                H[i, j:] *= -1
                U[i, :] *= -1
                p = -p

            # Reduce rows above
            if reduce_non_pivots:
                for r in range(i):
                    q = H[r, j] // p
                    if q != 0:
                        H[r, j:] -= H[i, j:] * q
                        U[r, :] -= U[i, :] * q

            i += 1
            j += 1
    return H, U

def row_hnf(
        A: np.ndarray[tuple[N, M], int],
        reduce_non_pivots: bool = True,
) -> RowHNFDecomposition:
    """
    Row Hermite Normal Form decomposition of an n×m matrix A, such that:
        H = U·A
    where H is in row echelon form and has positive pivots, and U is unimodular, hence:
        A = Uᵗ·H
    """
    H, U = _row_hnf(A, reduce_non_pivots)
    return RowHNFDecomposition(H, U)

class ColumnHNFDecomposition(NamedTuple):
    """
    (Column) Hermite Normal Form decomposition of an n×m matrix A, such that:
        H = A·U
    where H is in column echelon form and has positive pivots, and U is unimodular, hence:
        A = H·Uᵗ
    """
    H: np.ndarray[tuple[N, M], int]
    U: np.ndarray[tuple[N, M], int]

def col_hnf(
    A: np.ndarray[tuple[N, M], int],
    reduce_non_pivots: bool = True,
) -> ColumnHNFDecomposition:
    """
    Column Hermite Normal Form decomposition of an n×m matrix A, such that:
        H = A·U
    where H is in column echelon form and has positive pivots, and U is unimodular, hence:
        A = H·Uᵗ
    """
    H, U = _row_hnf(A.T, reduce_non_pivots)
    return ColumnHNFDecomposition(H.T, U.T)

def permutation_matrix_determinant(P: np.ndarray[tuple[N, N], T]) -> int:
    n = P.shape[0]
    assert P.shape == (n, n), "matrix must be square"
    P = P.copy()
    det = 1
    for i in range(n):
        if P[i, i] == 0:
            nz, = P[i:, i].nonzero()
            assert len(nz) == 1, "matrix is not a permutation matrix"
            j = nz[0]
            P[(i, j), i:] = P[(j, i), i:]
            det *= -1
    return det

def determinant(a: np.ndarray[tuple[N, N], T], *, pLU: PLUDecomposition = ...) -> T:
    """
    det(a)
    """
    if pLU is ...:
        try:
            pLU = pLU_decomposition(a)
        except ValueError:
            # Singular matrix
            return 0
    P, L, U, det_P = pLU
    det = det_P * L.diagonal().prod() * U.diagonal().prod()

    if np.issubdtype((a if isinstance(a, np.ndarray) else np.array(a)).dtype, np.integer):
        assert det.denominator == 1, "determinant of integer matrix must be integer!"
        return det.numerator
    return det

def solve_lower_triangular(
    L: np.ndarray[tuple[N, N], Fraction],
    y: np.ndarray[MatrixOrVectorShape[N], Fraction],
) -> np.ndarray[MatrixOrVectorShape[N], Fraction]:
    n = L.shape[0]
    assert L.shape == (n, n), "matrix must be square"
    if y.ndim == 1:
        assert len(y) == n, "incompatible rhs shape"
        m = 1
        shape = (n,)
    else:
        m = y.shape[1]
        assert y.shape == (n, m), "incompatible rhs shape"
        shape = y.shape
    x = np.zeros((n, m), dtype=object)
    for c in range(m):
        for i in range(n):
            yy = y[i, c]
            for j in range(i):
                yy -= L[i, j] * x[j, c]
            if L[i, i] == 0 and yy != 0:
                raise ValueError("incompatible system")
            x[i, c] = yy / L[i, i]
    return x.reshape(shape)

def solve_upper_triangular(
    U: np.ndarray[tuple[N, N], Fraction],
    y: np.ndarray[MatrixOrVectorShape[N], Fraction],
) -> np.ndarray[MatrixOrVectorShape[N], Fraction]:
    n = U.shape[0]
    assert U.shape == (n, n), "matrix must be square"
    if y.ndim == 1:
        assert len(y) == n, "incompatible rhs shape"
        m = 1
        shape = (n,)
    else:
        m = y.shape[1]
        assert y.shape == (n, m), "incompatible rhs shape"
        shape = y.shape
    x = np.zeros((n, m), dtype=object)
    for c in range(m):
        for i in range(n-1, -1, -1):
            yy = y[i, c]
            for j in range(i+1, n):
                yy -= U[i, j] * x[j, c]
            if U[i, i] == 0 and yy != 0:
                raise ValueError("incompatible system")
            x[i, c] = yy / U[i, i]
    return x.reshape(shape)

def solve(A: np.ndarray[tuple[N, N], int | Fraction], y: np.ndarray[MatrixOrVectorShape[N], int | Fraction], *, pLU: PLUDecomposition = ...) -> np.ndarray[MatrixOrVectorShape[N], Fraction]:
    if pLU is ...:
        pLU = pLU_decomposition(A)
    P, L, U, _ = pLU
    y_P = P.T @ y
    y_L = solve_lower_triangular(L, y_P)
    x = solve_upper_triangular(U, y_L)
    return x

def inverse_lower(L: np.ndarray[tuple[N, N], Fraction]) -> np.ndarray[tuple[N, N], Fraction]:
    """
    Inverse matrix of a square lower triangular matrix.
    """
    n = L.shape[0]
    Li = np.zeros((n, n), dtype=int) * Fraction(0, 1)
    for i in range(n):
        Li[i, i] = 1 / L[i, i]
        for j in range(i+1, n):
            Li[j, i] = -np.dot(L[j, :j], Li[:j, i]) / L[j, j]
    return Li

def inverse_upper(U: np.ndarray[tuple[N, N], Fraction]) -> np.ndarray[tuple[N, N], Fraction]:
    """
    Inverse matrix of a square upper triangular matrix.
    """
    return inverse_lower(U.T).T

def inverse(a: np.ndarray[tuple[N, N], Fraction], *, pLU: PLUDecomposition = ...) -> np.ndarray[tuple[N, N], Fraction]:
    """
    Inverse matrix of a square non-singular matrix.
    """
    if pLU is ...:
        pLU = pLU_decomposition(a)
    P, L, U, _ = pLU
    return inverse_upper(U) @ inverse_lower(L) @ P.T

def adjugate(a: np.ndarray[tuple[N, N], Fraction], *, pLU: PLUDecomposition = ...) -> np.ndarray[tuple[N, N], Fraction]:
    """
    Adjugate matrix of a square non-singular matrix.

    Each column is the normal vector of the hyperplane defined by the other columns,
    following cyclic left-hand ordering.
    """
    is_integer = np.issubdtype(a.dtype, np.integer)
    if pLU is ...:
        pLU = pLU_decomposition(a)
    det = determinant(a, pLU=pLU)
    inv = inverse(a, pLU=pLU)
    adj = inv * det
    if is_integer:
        adj = adj.astype(int)
    return adj

def rank(A: np.ndarray[tuple[N, M], int | Fraction], *, pLUq: PLUQDecomposition = ...) -> int:
    assert A.ndim == 2, "matrix must be 2D"
    n, m = A.shape

    if pLUq is ...:
        pLUq = pLUq_decomposition(A)

    U = pLUq.U
    if n > m:
        n, m = m, n
        U = U.T
    for i in reversed(range(n)):
        if np.any(U[i, i:] != 0):
            return i + 1
    return 0

def is_full_rank(
    A: np.ndarray[tuple[N, M], int | Fraction],
) -> bool:
    assert A.ndim == 2, "matrix must be 2D"
    m, n = A.shape
    return rank(A) == min(m, n)

def update_inverse(
    A: np.ndarray[tuple[N, N], Fraction],
    Ai: np.ndarray[tuple[N, N], Fraction],
    u: np.ndarray[tuple[N, Literal[1]], Fraction],
    v: np.ndarray[tuple[N, Literal[1]], Fraction],
) -> tuple[np.ndarray[tuple[N, N], Fraction], np.ndarray[tuple[N, N], Fraction]]:
    """
    Sherman-Morrison inverse matrix update.

    Computes Auv = A + u(v.T) and Auv^(-1) from A and A^(-1), Ai.
    """
    n = len(u.flat)
    u, v = u.reshape((n, 1)), v.reshape((n, 1))
    uvT = u@v.T
    det_ratio = 1 + v.T @ (Ai @ u)  # det(A + uvT) / det(A)
    if det_ratio == 0:
        raise ValueError("The updated matrix is singular")
    return A + uvT, Ai - (Ai @ uvT @ Ai) / det_ratio

def update_adjugate(
    A: np.ndarray[tuple[N, N], int],
    Adj: np.ndarray[tuple[N, N], int],
    u: np.ndarray[tuple[N], int],
    v: np.ndarray[tuple[N], int],
) -> tuple[np.ndarray[tuple[N, N], int], np.ndarray[tuple[N, N], int]]:
    """
    Sherman-Morrison formula adapted to update the adjugate matrix
    of an integer matrix.

    Computes Auv = (A + u(v.T)) and its adjugate from A and its adjugate, Adj.
    """
    n = len(u.flat)
    u, v = u.reshape((n, 1)), v.reshape((n, 1))
    uvT = u@v.T

    det: int = A[0, :].dot(Adj[:, 0])    # = det(A)
    det_updated = det + v.T @ (Adj @ u)  # = det(A + uvT)
    if det_updated == 0:
        raise ValueError("The updated matrix is singular")
    # Auv_i = Ai - (Ai @ uvT @ Ai) / (det(A + uvT) / det(A))
    # Auv_adj = det(A + uvT) * (Ai - (Ai @ uvT @ Ai)*det(A) / det(A + uvT))
    # Auv_adj = det(A + uvT) * det(A) * (Ai - (Ai @ uvT @ Ai)*det(A) / det(A + uvT)) / det(A)
    # Auv_adj = det(A + uvT) * (Adj - (Adj @ uvT @ Adj) / det(A + uvT)) / det(A)
    # Auv_adj = (det(A + uvT) * Adj - (Adj @ uvT @ Adj)) / det(A)
    return A + uvT, (det_updated * Adj - (Adj @ uvT @ Adj)) // det
