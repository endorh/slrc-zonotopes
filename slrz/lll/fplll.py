from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np

from slrz.util.optional_numba import njit


@njit
def update_fp_gram_schmidt(
    basis: np.ndarray[tuple[int, int], float],
    ortho: np.ndarray[tuple[int, int], float],
    ortho_sdot: np.ndarray[tuple[int], float],
    r: np.ndarray[tuple[int, int], float],
    i: int,
):
    """
    Updates the `i`-th column of a running Gram-Schmidt orthogonalization process
    on a square integer array, `basis`, using only integer arithmetic.
    Note that by Gram-Schmidt process we refer to orthogonalization only, not
    orthonormalization.

    :param basis: Source basis being orthogonalized.
    :param i: Column to update.
    """
    # We index with `...` rather than `:` because numba's typing
    # does not understand otherwise that column-major column slices are
    # memory-contiguous, resulting in degraded lowering specialization
    # for `np.dot` (and likely also the column copies).
    # See https://github.com/numba/numba/issues/8131
    ortho[..., i] = basis[..., i]
    for j in range(i):
        mu_i_j = ortho[..., j].dot(basis[..., i]) / ortho_sdot[j]
        r[j, i] = mu_i_j
        ortho[..., i] -= mu_i_j * ortho[..., j]
    ortho_sdot[i] = ortho[..., i].dot(ortho[..., i])


@njit
def _lll_reduction_fp(
    basis: np.ndarray[tuple[int, int], int], /,
    delta: np.ndarray[tuple, float] = np.float64(0.75),
) -> np.ndarray[tuple[int, int], int]:
    """
    LLL reduction of a lattice basis generated by the columns of an integer matrix `basis`,
    using floating-point arithmetic for intermediate values.

    See `lll_reduction_fp` for a friendlier interface.

    :param basis:
        2D integer matrix with the lattice generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        Should be in (0.25, 1).

        Its dtype will be used as dtype for intermediate computations.
        For example, you can specify np.float80 for extra precision.

    :return:
        Reduced lattice basis with the same shape and dtype as `basis`.
    """

    # Preconditions
    assert basis.ndim == 2, "A must be a 2D matrix"
    # if not np.issubdtype(gs_dtype, np.floating):
    #     warn("dtype for lll_reduction_fp should be a floating-point type", RuntimeWarning)
    gs_dtype = np.array(delta).dtype

    n, m = basis.shape
    if m > n:
        raise ValueError("Redundant basis are not supported.")
    # if not 0.25 < delta <= 1:
    #     warn("LLL is only well-defined for delta in (0.25, 1].", RuntimeWarning)
    # elif delta == 1:
    #     warn("LLL polynomial time is only guaranteed for delta in (0.25, 1), consider lowering delta.", RuntimeWarning)

    # Column-major for faster column dots:
    #   We use np.asfortranarray because numba does not support `copy('F')`
    #   We convert to column-major after the copy, because otherwise `copy()`
    #   resets to row-major.
    #   The algorithm would avoid this double copy if it were transposed.
    out: np.ndarray[tuple[int, int], int] = np.asfortranarray(basis.copy())

    # Buffers to swap columns efficiently
    column_buffer_int = np.empty(n, out.dtype)
    column_buffer_float = np.empty(n, gs_dtype)

    # FP copy to avoid converting to float on every GS update, since
    # numba does not support int dots
    basis_fp: np.ndarray[tuple[int, int], float] = out.astype(gs_dtype)

    # Orthogonalized basis
    # See notes above regarding the use of `np.asfortranarray` and `copy()`
    ortho: np.ndarray[tuple[int, int], int] = np.asfortranarray(np.zeros((n, m), gs_dtype))

    # Self dots of the columns of ortho:
    ortho_sdot: np.ndarray[tuple[int], int] = np.empty(m, gs_dtype)

    # The non-diagonal entries from the R matrix from the QR decomposition:
    r: np.ndarray[tuple[int, int], int] = np.empty((n, m), gs_dtype)

    update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, 0)

    k = 1
    step = 0
    while k < m:
        step += 1
        update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, k)
        for j in range(k - 1, -1, -1):
            mu_kj = r[j, k]

            # Size condition: mu_kj <= 1/2
            if abs(mu_kj) > 0.5:
                # See note above regarding https://github.com/numba/numba/issues/8131
                out[..., k] -= out[..., j] * round(mu_kj)
                basis_fp[..., k] = out[..., k]
                update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, k)

        # Lovász condition:
        #   ‖u'_k‖² >= (δ - μ_{k, k-1}²) ‖u'_{k-1}‖²
        if ortho_sdot[k] >= (delta - r[k - 1, k] ** 2) * ortho_sdot[k - 1]:
            k += 1
        else:
            column_buffer_int[:] = out[..., k]
            out[..., k] = out[..., k-1]
            out[..., k-1] = column_buffer_int
            column_buffer_float[:] = basis_fp[..., k]
            basis_fp[..., k] = basis_fp[..., k-1]
            basis_fp[..., k-1] = column_buffer_float
            if k > 1:
                k -= 1
            else:
                update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, 0)
    
    return out


def lll_reduction_fp_big_int(
        basis: np.ndarray[tuple[int, int], int], /,
        delta: float = 0.75, *,
        gs_dtype: Any = np.float64,
) -> np.ndarray[tuple[int, int], int]:
    """
    LLL reduction of a lattice basis generated by the columns of an integer matrix `basis`,
    using floating-point arithmetic for intermediate values.
    Unlike `lll_reduction_fp`, this function supports large integer entries
    in both the input basis and the output, with the consequent loss of precision
    derived from using a floating-point approximation of the Gram-Schmidt process.

    :param basis:
        2D integer matrix with the lattice generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        Should be in (0.25, 1).
    :param gs_dtype:
        Floating point dtype used for intermediate computations.

    :return:
        Reduced lattice basis with the same shape and dtype as `basis`.
    """

    # Preconditions
    assert basis.ndim == 2, "A must be a 2D matrix"
    if not np.issubdtype(gs_dtype, np.floating):
        warn("dtype for lll_reduction_fp should be a floating-point type", RuntimeWarning)

    n, m = basis.shape
    if m > n:
        raise ValueError("Redundant basis are not supported.")
    if not 0.25 < delta <= 1:
        warn("LLL is only well-defined for delta in (0.25, 1].", RuntimeWarning)
    elif delta == 1:
        warn("LLL polynomial time is only guaranteed for delta in (0.25, 1), consider lowering delta.", RuntimeWarning)

    # Column-major for faster column dots:
    #   We use np.asfortranarray because numba does not support `copy('F')`
    #   We convert to column-major after the copy, because otherwise `copy()`
    #   resets to row-major.
    #   The algorithm would avoid this double copy if it were transposed.
    out: np.ndarray[tuple[int, int], int] = np.asfortranarray(basis.copy().astype(object))

    # Buffers to swap columns efficiently
    column_buffer_int = np.empty(n, dtype=object)
    column_buffer_float = np.empty(n, gs_dtype)

    # FP copy to avoid converting to float on every GS update, since
    # numba does not support int dots
    basis_fp: np.ndarray[tuple[int, int], float] = out.astype(gs_dtype)

    # Orthogonalized basis
    # See notes above regarding the use of `np.asfortranarray` and `copy()`
    ortho: np.ndarray[tuple[int, int], int] = np.asfortranarray(np.zeros((n, m), gs_dtype))

    # Self dots of the columns of ortho:
    ortho_sdot: np.ndarray[tuple[int], int] = np.empty(m, gs_dtype)

    # The non-diagonal entries from the R matrix from the QR decomposition:
    r: np.ndarray[tuple[int, int], int] = np.empty((n, m), gs_dtype)

    update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, 0)

    k = 1
    step = 0
    while k < m:
        step += 1
        update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, k)
        for j in range(k - 1, -1, -1):
            mu_kj = r[j, k]

            # Size condition: mu_kj <= 1/2
            if abs(mu_kj) > 0.5:
                # See note above regarding https://github.com/numba/numba/issues/8131
                out[..., k] -= out[..., j] * round(mu_kj)
                basis_fp[..., k] = out[..., k]
                update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, k)

        # Lovász condition:
        #   ‖u'_k‖² >= (δ - μ_{k, k-1}²) ‖u'_{k-1}‖²
        if ortho_sdot[k] >= (delta - r[k - 1, k] ** 2) * ortho_sdot[k - 1]:
            k += 1
        else:
            column_buffer_int[:] = out[..., k]
            out[..., k] = out[..., k - 1]
            out[..., k - 1] = column_buffer_int
            column_buffer_float[:] = basis_fp[..., k]
            basis_fp[..., k] = basis_fp[..., k - 1]
            basis_fp[..., k - 1] = column_buffer_float
            if k > 1:
                k -= 1
            else:
                update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, 0)

    return out

def lll_reduction_fp(
    basis: np.ndarray[tuple[int, int], int], /,
    delta: float = 0.75, *,
    gs_dtype: Any = np.float64,
) -> np.ndarray[tuple[int, int], int]:
    """
    LLL reduction of a lattice basis generated by the columns of an integer matrix `basis`,
    using floating-point arithmetic for intermediate values.

    :param basis:
        2D integer matrix with the lattice generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        Should be in (0.25, 1).
    :param gs_dtype:
        Floating point dtype used for intermediate computations.

    :return:
        Reduced lattice basis with the same shape and dtype as `basis`.
    """
    return _lll_reduction_fp(basis, gs_dtype(delta))
