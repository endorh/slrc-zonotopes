"""
GCD, Extended GCD, and LCM generic implementations for arbitrary types,
and vectorized/reducing implementations for integers.

Vectorized implementations rely on `numba` JIT compilation.
If `numba` is not installed, they will fall back to a slower Python implementation.
"""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np

from fractions import Fraction

from slrz.util.optional_numba import njit, prange


__all__ = [
    'gcd',
    'gcd_frac',
    'gcd_array_flat',
    'gcd_array',
    'gcd_arrays',
    'gcd_generic',

    'egcd',
    'egcd_array_flat',
    'egcd_array',
    'egcd_generic',

    'lcm',
    'lcm_array_flat',
    'lcm_array',
    'lcm_arrays',
    'lcm_generic',
]


# Type variables
T = TypeVar('T')
Shape = TypeVar('Shape', bound=tuple)
A = TypeVar('A', bound=int)
B = TypeVar('B', bound=int)


@njit
def gcd(a: int, b: int) -> int:
    """
    Euclidean algorithm in binary form for njit-compatible types (e.g., int).

    The result is always non-negative, and it is `0` when `a == b == 0`.
    """
    while b != 0:
        a, b = b, a % b
    return abs(a)

@njit
def egcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean algorithm in binary form for njit-compatible types (e.g., int).

    The resulting `gcd` is always non-negative, and it is `0` when `a == b == 0`.
    The returned Bezout coefficients correspond to the returned positive `gcd`.
    """
    s, sn, t, tn = 1, 0, 0, 1
    while b != 0:
        q, r = divmod(a, b)
        a, b = b, r
        s, t, sn, tn = sn, tn, s - q*sn, t - q*tn
    if a < 0:
        a, s, t = -a, -s, -t
    return a, s, t

@njit
def _gcd_frac(a, b, c, d) -> tuple[int, int]:
    """
    gcd(a/b, c/d)

    Euclidean algorithm in binary form for unpacked fractions.
    """
    gn = gcd(a*d, c*b)
    dn = abs(b*d)
    g = gcd(gn, dn)
    return gn // g, dn // g

def gcd_frac(a: Fraction, b: Fraction) -> Fraction:
    """
    Euclidean algorithm in binary form for `Fraction`s.
    """
    return Fraction(*_gcd_frac(a.numerator, a.denominator, b.numerator, b.denominator))

@njit
def gcd_array_flat(array: np.ndarray[tuple[int, ...], int]) -> int:
    """
    Euclidean algorithm applied to reduce all elements of an array.

    Logically equivalent to reducing the raveled array using `gcd`.
    In particular, results are always non-negative, and they are `0`
    when all inputs are `0`.
    """
    # assert np.issubdtype(array.dtype, np.integer)  # numba does not support np.issubdtype
    ravel: np.ndarray[tuple[int], int] = array.copy().ravel()
    buffer: np.ndarray[tuple[int], int] = np.zeros_like(ravel)
    n = len(ravel)
    if n == 0:
        return 0
    if n == 1:
        return abs(ravel[0])
    while n > 1:
        # Extracting this to an `_array_gcd_reduce` function is sometimes slower (don't ask me why)
        if n % 2 == 1:
            buffer[(n - 1) // 2] = ravel[n - 1]
        for i in prange(n // 2):
            ii = 2 * i
            # Inlining the call to `gcd` here is sometimes slower (don't ask me why)
            buffer[i] = gcd(ravel[ii], ravel[ii + 1])
        ravel, buffer = buffer, ravel
        n = (n + 1) // 2
    return ravel[0]

@njit
def _gcd_array_slices_into(array: np.ndarray[tuple[int, ...], int], out: np.ndarray[tuple[int, ...], int]) -> np.ndarray[tuple[int, ...], int]:
    for idx in np.ndindex(out.shape):
        slice = array[idx]
        out[idx] = gcd_array_flat(slice)
    return out

def gcd_array(
    array: np.ndarray[tuple[int, ...], int],
    axis: int | tuple[int, ...] | None = None
) -> int | np.ndarray[tuple[int, ...], int]:
    """
    Euclidean algorithm applied to reduce elements of an array
    along the given axes.
    By default, reduces across all axes.

    Logically equivalent to reducing across the given axes using `gcd`.
    In particular, results are always non-negative, and they are `0`
    when all inputs are `0`.
    """
    if axis is None:
        return gcd_array_flat(array)
    if isinstance(axis, int):
        axis = (axis,)
    if len(axis) == array.ndim:
        return gcd_array_flat(array)
    moved = np.moveaxis(array, axis, range(-1, -len(axis)-1, -1))
    out = np.zeros_like(moved, shape=moved.shape[:-len(axis)])
    return _gcd_array_slices_into(moved, out)

@njit
def egcd_array_flat(array: np.ndarray[Shape, int]) -> tuple[int, np.ndarray[Shape, int]]:
    """
    Extended Euclidean algorithm applied to reduce all elements of an array,
    computing coefficients of the Bezout identity.

    Logically equivalent to reducing the raveled array using `egcd`.
    In particular, the resulting `gcd` is always non-negative, and it is `0`
    when all inputs are `0`.
    The resulting Bezout coefficients correspond to the positive `gcd` result.
    """
    ravel: np.ndarray[tuple[int], int] = array.copy().ravel()
    buffer: np.ndarray[tuple[int], int] = np.zeros_like(ravel)
    coefs: np.ndarray[tuple[int], int] = np.ones_like(ravel)
    n = len(ravel)
    if n == 0:
        return 0, coefs.reshape(array.shape)
    if n == 1:
        if ravel[0] < 0:
            return -ravel[0], -coefs.reshape(array.shape)
        return ravel[0], coefs.reshape(array.shape)
    m = n
    c = 1
    while m > 1:
        if m % 2 == 1:
            buffer[(m - 1) // 2] = ravel[m - 1]
        for i in prange(m // 2):
            ii = 2 * i
            # Extracting the sign rectification check out of the loop may be slightly more efficient
            buffer[i], s, t = egcd(ravel[ii], ravel[ii + 1])
            coefs[c * ii:c * (ii + 1)] *= s
            coefs[c * (ii + 1):min(c * (ii + 2), n)] *= t
        ravel, buffer = buffer, ravel
        m = (m + 1) // 2
        c *= 2
    return ravel[0], coefs.reshape(array.shape)

@njit
def _egcd_array_slices_into(
        array: np.ndarray[tuple[int, ...], int],
        out_gcd: np.ndarray[tuple[int, ...], int],
        out_bezout: np.ndarray[tuple[int, ...], int]
) -> tuple[np.ndarray[tuple[int, ...], int], np.ndarray[tuple[int, ...], int]]:
    for idx in np.ndindex(out_gcd.shape):
        slice = array[idx]
        egcd_res = egcd_array_flat(slice)
        out_gcd[idx] = egcd_res[0]
        out_bezout[idx] = egcd_res[1]
    return out_gcd, out_bezout

def egcd_array(
    array: np.ndarray[Shape, int], axis: int | tuple[int, ...] | None = None
) -> tuple[int | np.ndarray[tuple[int, ...], int], np.ndarray[Shape, int]]:
    """
    Extended Euclidean algorithm applied to reduce elements of an array
    along the given axes.
    By default, reduces across all axes.

    Logically equivalent to reducing across the given axes using `egcd`.
    In particular, the resulting `gcd` is always non-negative, and it is `0`
    when all inputs are `0`.
    The resulting Bezout coefficients correspond to the positive `gcd` result.
    """
    if axis is None:
        return egcd_array_flat(array)
    if isinstance(axis, int):
        axis = (axis,)
    if len(axis) == array.ndim:
        return egcd_array_flat(array)
    moved = np.moveaxis(array, axis, range(-1, -len(axis)-1, -1))
    out_gcd = np.zeros_like(moved, shape=moved.shape[:-len(axis)])
    out_bezout = np.zeros_like(moved)
    _egcd_array_slices_into(moved, out_gcd, out_bezout)
    return out_gcd, np.moveaxis(out_bezout, range(-1, -len(axis)-1, -1), axis)

def gcd_arrays(a: int | np.ndarray[A, int], b: int | np.ndarray[B, int]) -> np.ndarray[tuple[int, ...], int]:
    """
    Element-wise `gcd` of two arrays.
    Usual broadcasting rules apply if arrays have different shapes or are scalars.
    """
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} != {b.dtype}")
    return gcd_array(np.array((a, b), dtype=a.dtype), axis=0)

@njit
def lcm(a: int, b: int) -> int:
    """
    Least Common Multiple of two integers, computed from the `gcd`.
    It is always non-negative, and it is `0` when `a` or `b` are `0`.
    """
    p = abs(a * b)
    return p // gcd(a, b) if p != 0 else 0

@njit
def lcm_array_flat(array: np.ndarray[tuple[int, ...], int]) -> int:
    """
    Reduce an array using `lcm`.

    Logically equivalent to reducing the raveled array using `lcm`.
    In particular, results are always non-negative, and they are `0`
    when any inputs are `0`.
    """
    # assert np.issubdtype(array.dtype, np.integer)  # numba does not support np.issubdtype
    ravel: np.ndarray[tuple[int], int] = array.copy().ravel()
    buffer: np.ndarray[tuple[int], int] = np.zeros_like(ravel)
    n = len(ravel)
    if n == 0:
        return 0
    if n == 1:
        return abs(ravel[0])
    while n > 1:
        if n % 2 == 1:
            buffer[(n - 1) // 2] = ravel[n - 1]
        for i in prange(n // 2):
            ii = 2 * i
            buffer[i] = lcm(ravel[ii], ravel[ii + 1])
        ravel, buffer = buffer, ravel
        n = (n + 1) // 2
    return ravel[0]

@njit
def _lcm_array_slices_into(array: np.ndarray[tuple[int, ...], int], out: np.ndarray[tuple[int, ...], int]) -> np.ndarray[tuple[int, ...], int]:
    for idx in np.ndindex(out.shape):
        slice = array[idx]
        out[idx] = lcm_array_flat(slice)
    return out

def lcm_array(
    array: np.ndarray[tuple[int, ...], int],
    axis: int | tuple[int, ...] | None = None
) -> np.ndarray[tuple[int, ...], int] | int:
    """
    Reduce an array along the given axes using `lcm`.

    Logically equivalent to reducing across the given axes using `lcm`.
    In particular, results are always non-negative, and they are `0`
    when any inputs are `0`.
    """
    if axis is None:
        return lcm_array_flat(array)
    if isinstance(axis, int):
        axis = (axis,)
    if len(axis) == array.ndim:
        return lcm_array_flat(array)
    moved = np.moveaxis(array, axis, range(-1, -len(axis)-1, -1))
    out = np.zeros_like(moved, shape=moved.shape[:-len(axis)])
    return _lcm_array_slices_into(moved, out)

def lcm_arrays(a: np.ndarray[A, int], b: np.ndarray[B, int]) -> np.ndarray[tuple[int, ...], int]:
    """
    Element-wise `lcm` of two arrays.
    Usual broadcasting rules apply if arrays have different shapes or are scalars.
    """
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} != {b.dtype}")
    return lcm_array(np.array((a, b), dtype=a.dtype), axis=0)

def gcd_generic(*args: T, default: T | None = None, abs_result: bool = True) -> T | None:
    """
    Greatest Common Divisor computed with the Euclidean algorithm in n-ary form
    for generic datatypes implementing `__mod__` (or `__divmod__`)
    (e.g.: `int`, `Fraction`).

    The GCD of the empty set is :param:default, which defaults to `None`, unlike
    most implementations, which return `0` for convenience.

    The result will be in absolute value unless :param:abs_result is set to `False`.
    """
    if not args:
        return default
    g, args = args[0], args[1:]
    while args:
        b, args = args[0], args[1:]
        while b != 0:
            g, b = b, g % b
    return abs(g) if abs_result else g

def lcm_generic(*args: T, zero: T | Literal[0] = 0) -> T | Literal[0]:
    """
    Least Common Multiple computed from the GCD in n-ary form
    for generic datatypes implementing `__floordiv__` and `__mod__`
    (or `__divmod__`) (e.g.: `int`, `Fraction`).

    The LCM of the empty set is :param:zero, which defaults to `0`, but may be
    substituted with the proper `0` value for the given datatypes.
    Behavior is undefined if :param:zero is not a null ring element.
    """
    if not args:
        return zero
    m, args = args[0], args[1:]
    while args:
        b, args = args[0], args[1:]
        p = m * b
        if p == zero:
            return zero
        g = m
        while b != zero:
            g, b = b, g % b
        m = p // g
    return m

def egcd_generic(*args: T, default = None, abs_result: bool = True) -> tuple[T | None, tuple[int, ...]]:
    """
    Extended Euclidean algorithm in n-ary form for generic datatypes
    implementing `__abs__` and `__divmod__`.

    The GCD of the empty set is :param:default, which defaults to `None`, unlike
    most implementations, which return `0` for convenience.

    The resulting GCD will be in absolute value unless :param:abs_result is
    set to `False`.
    The Bezout coefficients are always relative to the returned `GCD` value.
    """
    if not args:
        return default, ()
    g, args = args[0], args[1:]
    cs = (1,)
    while args:  # for _ in range(len(args) + 1):
        b, args = args[0], args[1:]
        s, sn, t, tn = 1, 0, 0, 1
        while b != 0:
            q, r = divmod(g, b)
            g, b = b, r
            s, t, sn, tn = sn, tn, s - q*sn, t - q*tn
        cs = tuple(s * c for c in cs) + (t,)
    if abs_result:
        g_abs = abs(g)
        if g_abs != g:
            return g_abs, tuple(-c for c in cs)
    return g, cs
