from __future__ import annotations

from typing import Any, Sequence

import numpy as np

def min_int_type(a: np.ndarray[tuple, int]) -> Any:
    """
    Determine the smallest integer dtype that can hold all values in the matrix.
    """
    assert np.issubdtype(a.dtype, np.integer), "matrix type is not integer!"
    if np.any(a < 0):
        dtype = np.min_scalar_type(np.min(-np.abs(a) - (a >= 0)))
        assert np.issubdtype(dtype, np.signedinteger), "expected signed integer dtype!"
    else:
        dtype = np.min_scalar_type(np.max(a))
        assert np.issubdtype(dtype, np.unsignedinteger), "expected unsigned integer dtype!"
    return dtype

def squeeze_int_dtype(a: np.ndarray[tuple, int]) -> Any:
    """
    Casts the matrix to the smallest integer dtype that can hold all values in the matrix.
    Useful when saving a matrix with small entries to disk.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return a.astype(min_int_type(a))

def unsqueeze_int_dtype(a: np.ndarray[tuple, int]) -> Any:
    """
    Usually returns matrix with np.int64 dtype, as would be created with `dtype=int`.
    Useful when loading a matrix with small entries from disk, as operating with
    small dtypes may lead to `OverflowError` being raised at runtime (thankfully).
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return a.astype(int)

def flatten_inhomogeneous_int_array(
    arrays: Sequence[np.ndarray[tuple, int]],
) -> np.ndarray[tuple, int]:
    if not arrays:
        return np.zeros((0,), dtype=int)
    flat = np.zeros(
        sum(len(a) for a in arrays) + len(arrays) - 1,
        dtype=arrays[0].dtype)
    i = 0
    for arr in arrays:
        flat[i:i+len(arr)] = np.where(arr >= 0, arr+1, arr)
        i += len(arr) + 1
    return squeeze_int_dtype(flat)

def unflatten_inhomogeneous_int_array(
    flattened: np.ndarray[tuple, int],
) -> Sequence[np.ndarray[tuple, int]]:
    gaps = np.argwhere(flattened == 0)
    arrays = []
    prev = 0
    for gap in tuple(gaps.flat) + (len(flattened),):
        segment = flattened[prev:gap]
        prev = gap + 1
        arrays.append(np.where(segment >= 0, segment-1, segment))
    return arrays
