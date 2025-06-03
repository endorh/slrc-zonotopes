"""
This module conveniently exposes `numba.njit` if numba is installed,
and otherwise returns a passthrough decorator.
"""

NUMBA_CACHE_ENABLED = True

try:
    from numba import njit as _numba_njit, prange

    # Wrap decorator to fill the `cache` parameter with NUMBA_CACHE_ENABLED
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return _numba_njit(cache=NUMBA_CACHE_ENABLED)(*args)
        return _numba_njit(*args, **({
            'cache': NUMBA_CACHE_ENABLED,
        } | kwargs))


    NUMBA_AVAILABLE = True

except ImportError:
    from warnings import warn
    warn("Missing module `numba`. JIT compilation won't be available!")

    # Declare @njit as a passthrough decorator
    def njit(*args, **__):
        if args and callable(args[0]):
            return args[0]
        return lambda func: func

    # Declare prange as an alias to range
    prange = range

    NUMBA_AVAILABLE = False

__all__ = [
    'njit',
    'prange',
    'NUMBA_CACHE_ENABLED',
    'NUMBA_AVAILABLE',
]
