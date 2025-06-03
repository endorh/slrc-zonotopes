from __future__ import annotations

from typing import NamedTuple, Any
from warnings import warn

import numpy as np

from slrz.gcd import gcd, gcd_array_flat, gcd_generic

from slrz.util.optional_numba import njit, NUMBA_AVAILABLE


def round_div(num: int, den: int) -> int:
    """
    Round a rational number given by num/den.
    Half-integers are rounded to the nearest even integer, as is tradition.

    :param num: Dividend
    :param den: Divisor
    :return: round(num/den)
    """
    floor, mod = divmod(num, den)
    mod2 = mod * 2
    if mod2 > den:
        return floor + 1
    elif mod2 < den:
        return floor
    else:
        return floor + (floor % 2)

@njit
def _round_div(num: int, den: int) -> int:
    """
    Njit version of `round_div`.
    """
    floor, mod = divmod(num, den)
    mod2 = mod * 2
    if mod2 > den:
        return floor + 1
    elif mod2 < den:
        return floor
    else:
        return floor + (floor % 2)

def update_rational_gram_schmidt(
    basis: np.ndarray[tuple[int, int], int],
    ortho_num: np.ndarray[tuple[int, int], int],
    ortho_den: np.ndarray[tuple[int], int],
    ortho_sdot_num: np.ndarray[tuple[int], int],
    r_num: np.ndarray[tuple[int, int], int],
    i: int,
):
    """
    Updates the `i`-th column of a running Gram-Schmidt orthogonalization process
    on a square integer array, `basis`, using only integer arithmetic.
    Note that by Gram-Schmidt process we refer to orthogonalization only, not
    orthonormalization, as that would require irrational coefficients.

    :param basis: Source basis being orthogonalized.
    :param ortho_num: Numerators of the orthogonalized basis result.
    :param ortho_den: Denominators of the columns of `ortho_num`.
    :param ortho_sdot_num: Self dots of the columns of `ortho_num`, cached for efficiency.
    :param r_num: Upper triangular matrix where projection coefficients are stored.
        If initialized to the identity matrix, will result in the R matrix from a
        QR decomposition without normalization.
    :param i: Column to update.
    """
    big_int_arithmetic = basis.dtype.hasobject
    _gcd = gcd_generic if big_int_arithmetic else gcd
    _gcd_array = (lambda a: gcd_generic(*a)) if big_int_arithmetic else gcd_array_flat

    ortho_num[:, i] = basis[:, i]
    den = 1
    for j in range(i):
        sdot_j_num = ortho_sdot_num[j]
        j_den = ortho_den[j]
        # We don't simplify ortho_sdot_num in order to simplify below
        # sdot_j_den = ortho_sdot_den[j] = j_den ** 2

        # ⟨u_j, v_i⟩ / ⟨u_j, u_j⟩ = Fraction(j_num.dot(basis_i) * sdot_j_den, j_den * sdot_j_num)
        # ⟨u_j, v_i⟩ / ⟨u_j, u_j⟩ u_j = Fraction(j_num.dot(basis_i) * sdot_j_den, j_den * j_den * sdot_j_num) j_num
        #     = Fraction(j_num.dot(basis_i) * sdot_j_den, j_den * j_den * sdot_j_num) j_num
        #     = Fraction(j_num.dot(basis_i), sdot_j_num) j_num
        dot = ortho_num[:, j].dot(basis[:, i])
        g = _gcd(dot, sdot_j_num)
        proj_num = dot // g
        proj_den = sdot_j_num // g

        # ⟨u_j, v_i⟩ / ⟨u_j, u_j⟩ = Fraction(proj_num * j_den, proj_den)
        #     = Fraction(dot * j_den, sdot_j_num)
        r_num[j, i] = dot * j_den

        # u_i = u_i - ⟨u_j, u_i⟩ / ⟨u_j, u_j⟩ u_j
        # u_i = Fraction(num_new, den_new)
        #   den_new = lcm(den, proj_den)
        #   num_new = (i_num * den_new // den) - (proj_num * den_new // proj_den) * j_num
        #           = i_num * (proj_den // den_gcd) - (proj_num * (den // den_gcd)) * j_num
        den_gcd = _gcd(den, proj_den)
        den_mul = proj_den // den_gcd
        ortho_num[:, i] *= den_mul
        ortho_num[:, i] -= (proj_num * (den // den_gcd)) * ortho_num[:, j]
        den *= den_mul

    # Reduce column
    if den > 1:
        g = _gcd_array(ortho_num[:, i])
        if g > 1:
            gg = _gcd(g, den)
            ortho_num[:, i] //= gg
            den //= gg

    ortho_den[i] = den

    # We don't simplify by gcd(sdot_num, den**2) as explained above
    if big_int_arithmetic:
        ortho_sdot_num[i] = np.dot(ortho_num[:, i], ortho_num[:, i])
    else:
        # NOTE: `ortho_num[:, i] * ortho_num[:, i]` can hide an overflow, so we need to check explicitly
        #       I don't understand why numpy allows it, even when `np.seterr` is configured to raise.
        #       This is (mildly) concerning, as there are multiple other places where we cannot explicitly
        #       check for overflows ourselves without cost.
        #       We have to trust that no infinite loop derived from overflows will skip this check forever.
        ortho_square = ortho_num[:, i] * ortho_num[:, i]
        if not big_int_arithmetic and np.any(ortho_square < 0):
            raise OverflowError("Norm of Gram Schmidt intermediate value too large")
        ortho_sdot = ortho_square.sum()
        if not big_int_arithmetic and ortho_sdot < 0:
            raise OverflowError("Norm of Gram Schmidt intermediate value too large")
        ortho_sdot_num[i] = ortho_sdot

@njit
def _update_rational_gram_schmidt(
    basis: np.ndarray[tuple[int, int], int],
    ortho_num: np.ndarray[tuple[int, int], int],
    ortho_den: np.ndarray[tuple[int], int],
    ortho_sdot_num: np.ndarray[tuple[int], int],
    r_num: np.ndarray[tuple[int, int], int],
    i: int
):
    """
    Njit version of `update_gram_schmidt`.
    """
    ortho_num[:, i] = basis[:, i]
    den = 1
    for j in range(i):
        sdot_j_num = ortho_sdot_num[j]
        j_den = ortho_den[j]
        # We don't simplify ortho_sdot_num in order to simplify below
        # sdot_j_den = ortho_sdot_den[j] = j_den ** 2

        # ⟨u_j, v_i⟩ / ⟨u_j, u_j⟩ = Fraction(j_num.dot(basis_i) * sdot_j_den, j_den * sdot_j_num)
        # ⟨u_j, v_i⟩ / ⟨u_j, u_j⟩ u_j = Fraction(j_num.dot(basis_i) * sdot_j_den, j_den * j_den * sdot_j_num) j_num
        #     = Fraction(j_num.dot(basis_i) * sdot_j_den, j_den * j_den * sdot_j_num) j_num
        #     = Fraction(j_num.dot(basis_i), sdot_j_num) j_num
        # NOTE: numba does not support np.dot for ints, and `@` fails during lowering because
        #       it is reduced to dot products.
        dot = (ortho_num[:, j] * basis[:, i]).sum()
        g = gcd(dot, sdot_j_num)
        proj_num = dot // g
        proj_den = sdot_j_num // g

        # ⟨u_j, v_i⟩ / ⟨u_j, u_j⟩ = Fraction(proj_num * j_den, proj_den)
        #     = Fraction(dot * j_den, sdot_j_num)
        r_num[j, i] = dot * j_den

        # u_i = u_i - ⟨u_j, u_i⟩ / ⟨u_j, u_j⟩ u_j
        # u_i = Fraction(num_new, den_new)
        #   den_new = lcm(den, proj_den)
        #   num_new = (i_num * den_new // den) - (proj_num * den_new // proj_den) * j_num
        #           = i_num * (proj_den // den_gcd) - (proj_num * (den // den_gcd)) * j_num
        den_gcd = gcd(den, proj_den)
        den_mul = proj_den // den_gcd
        ortho_num[:, i] *= den_mul
        ortho_num[:, i] -= (proj_num * (den // den_gcd)) * ortho_num[:, j]
        den *= den_mul

    if den > 1:
        g = gcd_array_flat(ortho_num[:, i])
        if g > 1:
            gg = gcd(g, den)
            ortho_num[:, i] //= gg
            den //= gg

    ortho_den[i] = den

    # We don't simplify by gcd(sdot_num, den**2) as explained above
    # Note: `ortho_num[:, i] * ortho_num[:, i]` can hide an overflow, so we need to check explicitly
    #       I don't understand why numpy allows it, even when `np.seterr` is configured to raise.
    #       This is (mildly) concerning, as there are multiple other places where we cannot explicitly
    #       check for overflows ourselves without cost.
    #       We have to trust that no infinite loop derived from overflows will skip this check forever.
    ortho_square = ortho_num[:, i] * ortho_num[:, i]
    if np.any(ortho_square < 0):
        raise OverflowError("Norm of Gram Schmidt intermediate value too large")
    # NOTE: numba does not support np.dot for ints, and `@` fails during lowering because it is
    #       reduced to dot products.
    ortho_sdot = ortho_square.sum()
    if ortho_sdot < 0:
        raise OverflowError("Norm of Gram Schmidt intermediate value too large")
    ortho_sdot_num[i] = ortho_sdot

class RationalGramSchmidtOrthogonalization(NamedTuple):
    """
    Rational Gram-Schmidt orthogonalization of an integer vector basis.

    :ivar ortho_num:
        Numerators of the orthogonalized basis.
    :ivar ortho_den:
        Shared denominators for each column of `ortho_num`.
    :ivar r_num:
        Numerators of the R matrix from the QR decomposition,
        encoding row eliminations performed by the Gram-Schmidt process.
        Their implicit denominators are `r_den`, which may be unsimplified.
    :ivar r_den:
        Self dots of the columns of `ortho_num`.
        Their implicit denominators are `ortho_den**2`.
    """
    ortho_num: np.ndarray[tuple[int, int], int]
    ortho_den: np.ndarray[tuple[int], int]
    r_num: np.ndarray[tuple[int, int], int]
    r_den: np.ndarray[tuple[int], int]

def rational_gram_schmidt(basis: np.ndarray[tuple[int, int], int]) -> RationalGramSchmidtOrthogonalization:
    """
    Gram-Schmidt orthogonalization of a basis generated by the columns of an integer matrix.
    Unlike in QR decomposition, no normalization is performed, as it is not closed for
    the rational numbers.

    :param basis:
        2D integer matrix representing a vector basis generated by its columns.
        May be rectangular (m×n) with m <= n.
    """
    n = basis.shape[0]
    m = basis.shape[1]
    ortho_num = basis.copy('F')
    ortho_den: np.ndarray[tuple[int], int] = np.ones_like(basis, shape=(m,))
    ortho_sdot_num: np.ndarray[tuple[int], int] = np.zeros_like(basis, shape=(m,))
    r_num: np.ndarray[tuple[int, int], int] = np.zeros_like(basis)
    for i in range(m):
        r_num[m, m] = 1
    for i in range(m):
        update_rational_gram_schmidt(basis, ortho_num, ortho_den, ortho_sdot_num, r_num, i)
    return RationalGramSchmidtOrthogonalization(ortho_num, ortho_den, r_num, ortho_sdot_num)

@njit
def _rational_gram_schmidt_numba(basis: np.ndarray[tuple[int, int], int]) -> tuple[
    np.ndarray[tuple[int, int], int],
    np.ndarray[tuple[int], int],
    np.ndarray[tuple[int, int], int],
    np.ndarray[tuple[int], int],
]:
    """
    See `rational_gram_schmidt_numba`.
    """
    n = basis.shape[0]
    m = basis.shape[1]
    ortho_num: np.ndarray[tuple[int, int], int] = np.asfortranarray(basis).copy()
    ortho_den: np.ndarray[tuple[int], int] = np.ones((m,), dtype=basis.dtype)
    ortho_sdot_num: np.ndarray[tuple[int], int] = np.zeros((m,), dtype=basis.dtype)
    r_num: np.ndarray[tuple[int, int], int] = np.zeros((n, m), dtype=basis.dtype)
    for i in range(m):
        r_num[m, m] = 1
    for i in range(m):
        update_rational_gram_schmidt(basis, ortho_num, ortho_den, ortho_sdot_num, r_num, i)
    return ortho_num, ortho_den, r_num, ortho_sdot_num

def rational_gram_schmidt_numba(basis: np.ndarray[tuple[int, int], int]) -> RationalGramSchmidtOrthogonalization:
    """
    Njit version of `rational_gram_schmidt`.
    May raise an OverflowError.

    :param basis:
        2D integer matrix with the lattice generators as columns.
        May be rectangular (m×n) with m <= n.
    """
    ortho_num, ortho_den, r_num, r_den = _rational_gram_schmidt_numba(basis)
    return RationalGramSchmidtOrthogonalization(ortho_num, ortho_den, r_num, r_den)

def _lll_reduction_pure(
    basis: np.ndarray, /,
    delta: float = 0.75, *,
    dtype: Any = np.int64,
) -> np.ndarray:
    """
    LLL reduction of a lattice basis generated by the columns of an integer matrix A.
    See `lll_reduction` for a friendlier interface.

    May raise an OverflowError if intermediate values escape the representable
    range of the chosen dtype.
    It may also theoretically loop endlessly if the overflow detecting measures
    happen to be avoided forever within an overflow-caused endless loop, or,
    even produce an incorrect result due to overflow, though very unlikely.

    Consider using `dtype=object` to avoid all overflow-related problems, at the
    expense of less optimized performance.

    :param basis:
        2D integer matrix with the lattice generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        Should be in (0.25, 1).
    :param dtype:
        Numpy dtype used for intermediate and final results.
        Use `dtype=object` to avoid all overflow-related problems, at the expense
        of less optimized performance.
    :return:
        Reduced basis.
    """

    # Preconditions
    assert basis.ndim == 2, "A must be a 2D matrix"
    n, m = basis.shape
    if m > n:
        raise ValueError("Redundant basis are not supported.")
    if not 0.25 < delta <= 1:
        warn("LLL is only well-defined for delta in (0.25, 1].", RuntimeWarning)
    elif delta == 1:
        warn("LLL polynomial time is only guaranteed for delta in (0.25, 1), consider lowering delta.", RuntimeWarning)

    # Column-major copy for faster column dots:
    out: np.ndarray[tuple[int, int], int] = basis.astype(dtype, order='F').copy()

    # Numerators of the orthogonalized basis of A
    ortho_num: np.ndarray[tuple[int, int], int] = out.copy()

    # Denominators of the columns of ortho_num:
    ortho_den: np.ndarray[tuple[int], int] = np.ones((m,), dtype)

    # Self dots of the columns of ortho_num:
    #   Implicit denominator: ortho_den**2
    ortho_sdot_num: np.ndarray[tuple[int], int] = np.zeros((m,), dtype)

    # The non-diagonal entries from the R matrix from the QR decomposition:
    #   Denominator: ortho_sdot_num
    r_num: np.ndarray[tuple[int, int], int] = np.zeros((n, m), dtype)

    update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, 0)

    k = 1
    step = 0
    while k < m:
        step += 1
        update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, k)
        for j in range(k - 1, -1, -1):
            mu_kj_num = r_num[j, k]
            mu_kj_den = ortho_sdot_num[j]

            # Size condition: mu_kj <= 1/2
            if abs(mu_kj_num) * 2 > mu_kj_den:
                out[:, k] -= out[:, j] * round_div(mu_kj_num, mu_kj_den)
                update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, k)

        mu_k_prev = r_num[k-1, k] / ortho_sdot_num[k-1]
        ortho_k_sdot = float(ortho_sdot_num[k]) / ortho_den[k]**2
        ortho_prev_sdot = float(ortho_sdot_num[k-1]) / ortho_den[k-1]**2

        # Lovász condition: ‖u'_k‖² >= (δ - μ_{k, k-1}²) ‖u'_{k-1}‖²
        if ortho_k_sdot >= (delta - mu_k_prev**2) * ortho_prev_sdot:
            k += 1
        else:
            out[:, (k-1, k)] = out[:, (k, k-1)]
            if k > 1:
                k -= 1
            else:
                update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, 0)

    return out

@njit
def _lll_reduction_numba(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """
    Njit version of _lll_reduction_pure.
    See `lll_reduction` for a friendlier interface.

    May raise an OverflowError if intermediate values escape the representable
    range of int64.

    It may also theoretically loop endlessly if the overflow detecting measures
    happen to be avoided forever within an overflow-caused endless loop, or,
    even produce an incorrect result due to overflow, though very unlikely.

    :param basis:
        2D integer matrix with the lattice generators as columns.
        May be rectangular (n×m) with m <= n.
        Its dtype will be used for intermediate values and the final result.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        Should be in (0.25, 1).
    :return:
        Reduced basis.
    """
    assert basis.ndim == 2, "A must be a 2D matrix"
    n = basis.shape[0]
    m = basis.shape[1]
    if m > n:
        raise ValueError("Redundant basis are not supported.")

    # Column-major copy for faster column dots
    #   numba does not support the `order` parameter, so we use `np.asfortranarray`
    out: np.ndarray[tuple[int, int], int] = np.asfortranarray(basis).copy()

    # Numerators of the orthogonalized basis of A
    ortho_num: np.ndarray[tuple[int, int], int] = np.asfortranarray(basis).copy()

    # Denominators of the columns of ortho_num:
    ortho_den: np.ndarray[tuple[int], int] = np.ones((m,), basis.dtype)

    # Self dots of the columns of ortho_num:
    #   Implicit denominator: ortho_den**2
    ortho_sdot_num: np.ndarray[tuple[int], int] = np.zeros((m,), basis.dtype)

    # The non-diagonal entries from the R matrix from the QR decomposition:
    #   Denominator: ortho_sdot_num
    r_num: np.ndarray[tuple[int, int], int] = np.zeros((n, m), basis.dtype)

    _update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, 0)

    k = 1
    step = 0
    while k < m:
        step += 1
        _update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, k)
        for j in range(k - 1, -1, -1):
            mu_kj_num = r_num[j, k]
            mu_kj_den = ortho_sdot_num[j]

            # Size condition: mu_kj <= 1/2
            if abs(mu_kj_num) * 2 > mu_kj_den:
                out[:, k] -= out[:, j] * _round_div(mu_kj_num, mu_kj_den)
                _update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, k)

        mu_k_prev = r_num[k-1, k] / ortho_sdot_num[k-1]
        ortho_k_sdot = float(ortho_sdot_num[k]) / ortho_den[k]**2
        ortho_prev_sdot = float(ortho_sdot_num[k-1]) / ortho_den[k-1]**2

        # Lovász condition: ‖u'_k‖² >= (δ - μ_{k, k-1}²) ‖u'_{k-1}‖²
        if ortho_k_sdot >= (delta - mu_k_prev**2) * ortho_prev_sdot:
            k += 1
        else:
            out[:, np.array((k-1, k))] = out[:, np.array((k, k-1))]
            if k > 1:
                k -= 1
            else:
                _update_rational_gram_schmidt(out, ortho_num, ortho_den, ortho_sdot_num, r_num, 0)
    return out

def lll_reduction(
    basis: np.ndarray[tuple[int, int], int], /,
    delta: float = 0.75, *,
    attempt_acceleration: bool = ...,
    attempt_numba_acceleration: bool = NUMBA_AVAILABLE,
    attempt_numpy_acceleration: bool = not NUMBA_AVAILABLE,
    numpy_acceleration_dtype: Any = np.int64,
    cast_to_numpy_int_if_possible: bool = True,
    fail_on_acceleration_fail: bool = False,
    warn_on_acceleration_fail: bool = True,
    warn_on_numba_acceleration_fail: bool = ...,  # = warn_on_acceleration_fail
    warn_on_numpy_acceleration_fail: bool = ...,  # = warn_on_acceleration_fail
) -> np.ndarray[tuple[int, int], int]:
    """
    Lenstra–Lenstra–Lovász (LLL) lattice basis reduction algorithm.

    Exact arithmetic implementation.
    Consider using `lll_reduction_fp` if performance is preferred.

    This wrapper is a friendly interface to `lll_reduction_big_int`, `lll_reduction_numba`
    and `lll_reduction_numpy`.
    It will attempt to use the best accelerated implementation available, and
    automatically fall back to `lll_reduction_big_int` in case of overflow.

    :param basis:
        Integer 2D matrix with the lattice basis generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        The algorithm is well-defined for delta in (0.25, 1], and has polynomial-time
        complexity (O(d^5n log^3 B)) for delta in (0.25, 1) (excluding delta=1).
        The default value is 0.75.
    :param attempt_acceleration:
        If set to `False`, overrides `attempt_numba_acceleration` and `attempt_numpy_acceleration`.
    :param attempt_numba_acceleration:
        Attempts to use a numba-accelerated implementation of the LLL reduction.
        This implementation may easily encounter overflow errors since the intermediate
        Gram-Schmidt vectors can grow exponentially in size with respect to the dimension.
        If the attempt fails, the algorithm falls back to a pure-Python implementation.
        Disabled by default if numba is not available.
    :param attempt_numpy_acceleration:
        Attempts to use a numpy-accelerated implementation of the LLL reduction.
        The dtype used for intermediate results can be specified with the `numpy_acceleration_dtype` parameter.
        If the attempt fails due to overflow, the algorithm falls back to a pure-Python implementation.
        At the moment, the numpy implementation is constrained to the same format (`np.int64`) as the
        numba implementation, yielding no additional value.
        Hence, why it is disabled by default if numba is available.
    :param numpy_acceleration_dtype:
        Dtype used for intermediate results when using the numpy-accelerated implementation.
        If numpy ever supports `np.int256` or big integer formats, they could be used here.
    :param cast_to_numpy_int_if_possible:
        If no acceleration succeeded, the result is generally a `dtype=object` array.
        If this flag is set, the result will be cast to `dtype=int` if all values are integers
    :param fail_on_acceleration_fail:
        Do not attempt to recover from overflow/floating point errors in the numba or
        numpy implementations.
    :param warn_on_acceleration_fail:
        Emit a warning if the numba or numpy acceleration attempts fail due to overflow.
        Ignored if `fail_on_acceleration_fail` is set.
    :param warn_on_numba_acceleration_fail:
        Emit a warning if the numba acceleration attempts fail due to overflow.
        Overrides `warn_on_acceleration_fail` if set.
        Ignored if `fail_on_acceleration_fail` is set.
    :param warn_on_numpy_acceleration_fail:
        Emit a warning if the numpy acceleration attempts fail due to overflow.
        Overrides `warn_on_acceleration_fail` if set.
        Ignored if `fail_on_acceleration_fail` is set.
    :return:
        Reduced basis with same shape as `basis`.
    """
    if not attempt_acceleration:
        attempt_numba_acceleration = False
        attempt_numpy_acceleration = False
    if attempt_numba_acceleration or attempt_numpy_acceleration:
        if not np.issubdtype(basis.dtype, np.integer):
            raise ValueError("Accelerated LLL requires an integer array input.")
        old_settings = np.seterr(over='raise')
        try:
            if attempt_numba_acceleration:
                try:
                    return _lll_reduction_numba(basis, delta)
                except (OverflowError, FloatingPointError) as e:
                    if fail_on_acceleration_fail:
                        raise e
                    if warn_on_numba_acceleration_fail is ...:
                        warn_on_numba_acceleration_fail = warn_on_acceleration_fail
                    if warn_on_numba_acceleration_fail:
                        warn("Numba-accelerated LLL reduction failed due to overflow. Consider setting `attempt_numba_acceleration=False` if this is freqent.", RuntimeWarning)
            if attempt_numpy_acceleration:
                try:
                    return _lll_reduction_pure(basis, delta=delta, dtype=numpy_acceleration_dtype)
                except (OverflowError, FloatingPointError) as e:
                    if fail_on_acceleration_fail:
                        raise e
                    if warn_on_numpy_acceleration_fail is ...:
                        warn_on_numpy_acceleration_fail = warn_on_acceleration_fail
                    if warn_on_numpy_acceleration_fail:
                        warn("Numpy-accelerated LLL reduction failed due to overflow. Consider setting `attempt_numpy_acceleration=False` if this is freqent.", RuntimeWarning)
        finally:
            np.seterr(**old_settings)
    else:
        if not np.issubdtype(basis.dtype, np.integer) and not basis.dtype.hasobject:
            raise ValueError("Accelerated LLL requires an integer array input.")
        # We do not check if all objects are integers
    result = _lll_reduction_pure(basis, delta=delta, dtype=object)

    if cast_to_numpy_int_if_possible and np.all(result < 2**63) and np.all(result >= -2**63):
        return result.astype(int)
    return result

def lll_reduction_numba(
    basis: np.ndarray[tuple[int, int], int], /,
    delta: float = 0.75,
) -> np.ndarray[tuple[int, int], int]:
    """
    Lenstra–Lenstra–Lovász (LLL) lattice basis reduction algorithm.

    Numba-accelerated exact arithmetic implementation.
    It will easily encounter overflow errors beyond dimension 4.
    Consider using `lll_reduction_big_int`, `lll_reduction_fp` in such case.
    You may also use `lll_reduction` which will automatically fall back to
    `lll_reduction_big_int` in the case of overflow.

    :param basis:
        Integer 2D matrix with the lattice basis generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        The algorithm is well-defined for delta in (0.25, 1], and has polynomial-time
        complexity (O(d^5n log^3 B)) for delta in (0.25, 1) (excluding delta=1).
        The default value is 0.75.
    :return:
        Reduced basis with same shape as `basis`.
    """
    return _lll_reduction_numba(basis, delta)

def lll_reduction_numpy(
    basis: np.ndarray[tuple[int, int], int], /,
    delta: float = 0.75, *,
    dtype: Any = np.int64,
) -> np.ndarray[tuple[int, int], int]:
    """
    Lenstra–Lenstra–Lovász (LLL) lattice basis reduction algorithm.

    Numba-accelerated exact arithmetic implementation.
    It will easily encounter overflow errors beyond dimension 4.
    Consider using `lll_reduction_big_int`, `lll_reduction_fp` in such case.
    You may also use `lll_reduction` which will automatically fall back to
    `lll_reduction_big_int` in the case of overflow.

    :param basis:
        Integer 2D matrix with the lattice basis generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        The algorithm is well-defined for delta in (0.25, 1], and has polynomial-time
        complexity (O(d^5n log^3 B)) for delta in (0.25, 1) (excluding delta=1).
        The default value is 0.75.
    :param dtype:
        Dtype used for intermediate results when using the numpy-accelerated implementation.
        If numpy ever supports `np.int256` or big integer formats, they could be used here.
    :return:
        Reduced basis with same shape as `basis`.
    """
    return _lll_reduction_pure(basis, delta, dtype=dtype)

def lll_reduction_big_int(
    basis: np.ndarray[tuple[int, int], int], /,
    delta: float = 0.75,
) -> np.ndarray[tuple[int, int], int]:
    """
    Lenstra–Lenstra–Lovász (LLL) lattice basis reduction algorithm.

    Exact arithmetic implementation with support for integers of arbitrary size.
    Consider using `lll_reduction_fp` if performance is preferred.

    :param basis:
        Integer 2D matrix with the lattice basis generators as columns.
        May be rectangular (n×m) with m <= n.
    :param delta:
        Delta value for the Lovász condition.
        Higher values of delta lead to stronger reductions of the basis at the expense
        of harder computation.
        The algorithm is well-defined for delta in (0.25, 1], and has polynomial-time
        complexity (O(d^5n log^3 B)) for delta in (0.25, 1) (excluding delta=1).
        The default value is 0.75.
    :param dtype:
        Dtype used for intermediate results when using the numpy-accelerated implementation.
        If numpy ever supports `np.int256` or big integer formats, they could be used here.
    :return:
        Reduced basis with same shape as `basis`.
    """
    return _lll_reduction_pure(basis, delta=delta, dtype=object)
