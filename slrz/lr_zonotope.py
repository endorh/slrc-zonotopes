from __future__ import annotations

from functools import cached_property
from typing import Iterable, Literal

import numpy as np

from slrz.gcd import gcd
from slrz.lll import lll_reduction, lll_reduction_fp, lll_reduction_big_int, lll_reduction_fp_big_int
from slrz.rational import linalg
from slrz.rational.milp import milp, Bounds, LinearConstraint
from slrz.util import Profiler

from slrz.util.optional_numba import njit


__all__ = [
    'LRZonotope',
    'lr_zonotope_from_volume_vector',
    'sLRC_primitive_volume_vectors',
    'zonotope_lattice_width',
    'get_inequalities',
    'get_centrally_symmetric_inequalities'
]


class LRZonotope:
    @staticmethod
    def from_volume_vector(
        volume_vector: tuple[int, ...] | np.ndarray[tuple[int], int], *,
        use_fp_lll: bool = False,
        delta: float = 0.75,
        profiler: Profiler=Profiler.noop,
    ):
        """
        Constructs a Lonely Runner Zonotope from a volume vector.

        The returned zonotope is unique modulo unimodular transformations.
        """
        return lr_zonotope_from_volume_vector(volume_vector, use_fp_lll=use_fp_lll, delta=delta, profiler=profiler)

    def __init__(self, generators: list[list[int]] | np.ndarray, *, _skip_checks: bool = False):
        """
        generators:  3×4 matrix
            ( | ... | )
            ( u₁... uₙ)
            ( | ... | )
        """
        if _skip_checks:
            self.generators = generators #.copy()
            # self.generators.flags.writeable = False
        else:
            if isinstance(generators, (list, tuple)):
                generators = np.array(generators, dtype=int).T
            if generators.shape != (3, 4):
                raise ValueError("generators must be a 3x4 matrix")
            if generators.dtype != int:
                int_generators = generators.astype(int)
                if np.any(int_generators.astype(generators.dtype) != generators):
                    raise ValueError("generators must be integers")
            self.generators = generators.copy()
            self.generators.flags.writeable = False
            if any(v == 0 for v in self.volume_vector):
                raise ValueError("generators are not in linear general position")

            assert np.all(self.generators @ np.array(self.volume_vector, dtype=int).reshape((4,1)) == 0), "volume_vector does not satisfy definition"

    @cached_property
    def volume_vector(self) -> tuple[int, int, int, int]:
        # noinspection PyTypeChecker
        return tuple(int(v) for v in volume_vector(self.generators))

    @property
    def volume(self) -> int:
        return sum(self.volume_vector)

    @cached_property
    def lattice_width(self) -> int:
        return zonotope_lattice_width(self.generators)

    @property
    def inequalities(self):
        return get_inequalities(self.generators)

    @property
    def centrally_symmetric_inequalities(self):
        return get_centrally_symmetric_inequalities(self.generators)

    @property
    def is_sLR_zonotope(self):
        vv = self.volume_vector
        return len(set(vv)) == len(vv)

    def __eq__(self, other):
        return isinstance(other, LRZonotope) and np.all(self.generators == other.generators)

    def __hash__(self):
        return hash(self.volume_vector)

    def __str__(self):
        return f"LRZ.from_volume_vector({self.volume_vector})"

    def __repr__(self):
        return f"LRZ({self.generators})"

def lr_zonotope_from_volume_vector(
    vol_vector: tuple[int, ...] | np.ndarray[tuple[int], int], *,
    use_fp_lll: bool = False,
    delta: float = 0.75,
    profiler: Profiler = Profiler.noop,
):
    """
    Constructs a Lonely Runner Zonotope from a volume vector.

    The returned zonotope is unique modulo unimodular transformations.
    """
    with profiler["pre_cond"]:
        if isinstance(vol_vector, tuple) or isinstance(vol_vector, list):
            if not all(v % 1 == 0 for v in vol_vector):
                raise ValueError("volume vector must be integer")
            vol_vector = np.array(vol_vector, dtype=int)

        n = len(vol_vector)

    with profiler["z_prime"]:
        Z_prime = np.concatenate((
            np.eye(n-1, dtype=int) * vol_vector[n-1],
            vol_vector[:n-1].reshape((n-1, 1))
        ), axis=1)

    # HNF basis reduction
    with profiler["hnf"]:
        H, U = linalg.col_hnf(Z_prime)
        Z_lattice_basis = H[:, :n-1]

    with profiler["inv"]:
        Z = linalg.solve_lower_triangular(Z_lattice_basis, Z_prime)
    with profiler["inv-check"]:
        try:
            Z_int = Z.astype(int)
        except OverflowError:
            Z_int = np.array([
                [round(e) for e in row] for row in Z
            ], dtype=object)
        assert np.all(Z == Z_int), "Lattice inversion is not unimodular!"

    # LLL basis reduction
    with profiler["lll"]:
        if Z_int.dtype.hasobject:
            lll = lll_reduction_fp_big_int if use_fp_lll else lll_reduction_big_int
        else:
            lll = lll_reduction_fp if use_fp_lll else lll_reduction
        Z_red = lll(Z_int.T, delta=delta).T

    with profiler["vol_test"]:
        vv = volume_vector(Z_red)

    # Correct the volume signs
    with profiler["vol_signs"]:
        neg_idx = tuple(i for i in range(4) if vv[i] < 0)
        if len(neg_idx) == 0:
            assert all(vv_actual == vv_expected for vv_actual, vv_expected in zip(vv, volume_vector)), f"constructed zonotope does not have expected volume vector! (expected: {volume_vector}) (actual: {vv})"
            Z = LRZonotope(Z_red)
        if len(neg_idx) == 1:
            Z = LRZonotope(Z_red * np.array([1 if i in neg_idx else -1 for i in range(4)], dtype=int))
        elif len(neg_idx) == 2:
            Z = LRZonotope(Z_red * np.array([-1 if i in neg_idx else 1 for i in range(4)], dtype=int))
        elif len(neg_idx) == 3:
            Z = LRZonotope(Z_red * np.array([1 if i in neg_idx else -1 for i in range(4)], dtype=int))
        elif len(neg_idx) == 4:
            Z = LRZonotope(-Z_red)

    with profiler["vol_check"]:
        vv = Z.volume_vector
        assert all(vv_actual == vv_expected for vv_actual, vv_expected in zip(vv, vol_vector)), f"constructed zonotope does not have expected volume vector! (expected: {vol_vector}) (actual: {vv})"

    return Z

@njit
def get_inequalities(G: np.ndarray[tuple[Literal[3], Literal[4]], int]) -> tuple[np.ndarray[tuple[Literal[12], Literal[3]], int], np.ndarray[tuple[Literal[12]], int]]:
    # Initialize lists to store A rows and b values
    d, g = G.shape
    f = g*(g-1)
    A = np.zeros((f, d), G.dtype)

    # For each pair of generators, compute their cross product
    r = 0
    for i in range(4):
        for j in range(i+1, 4):
            # Get the two generator columns
            g_i = G[:, i]
            g_j = G[:, j]
            
            # Compute cross product and add both directions
            cross = np.cross(g_i, g_j)

            # assert np.any(cross != 0), f"Generators {g_i} and {g_j} are proportional"
            A[r, :] = cross
            A[r+1, :] = -cross
            r += 2
    
    # Compute b as sum of max(0, dot product with each generator)
    b = np.zeros((f,), G.dtype)
    for r in range(f):
        s = 0
        for j in range(g):
            dt = 0
            for i in range(d):
                dt += A[r, i] * G[i, j]
            s += max(0, dt)
        b[r] = s

    return A, b

@njit
def get_centrally_symmetric_inequalities(G):
    """
    The returned inequalities may correspond to a half-lattice embedding
    of the zonotope, if the center of the zonotope is not a lattice point.
    """
    # Initialize lists to store A rows and b values
    d, g = G.shape
    f = g*(g-1)
    A = np.zeros((f, d), G.dtype)

    # For each pair of generators, compute their cross product
    r = 0
    for i in range(4):
        for j in range(i+1, 4):
            # Get the two generator columns
            g_i = G[:, i]
            g_j = G[:, j]

            # Compute cross product and add both directions
            cross = np.cross(g_i, g_j)

            # assert np.any(cross != 0), f"Generators {g_i} and {g_j} are proportional"
            A[r, :] = cross
            A[r+1, :] = -cross
            r += 2

    # Compute b as sum of abs(dot product with each generator)
    b = np.zeros((f,), G.dtype)
    for r in range(f):
        s = 0
        for j in range(g):
            dt = 0
            for i in range(d):
                dt += A[r, i] * G[i, j]
            s += abs(dt)
        b[r] = s

    return A*2, b

# @numba.njit()
def volume_vector(Z):
    """
    Computes the volume vector of a 4-LRZ, Z, obtained as the determinants
    of the 4 3×3 minors of the matrix of generators of Z.

    EXAMPLES:
        >>> print(volume_vector(np.array([
        >>>     [1, 0, 0],
        >>>     [0, 1, 0],
        >>>     [0, 0, 1],
        >>>     [1, 1, 1],
        >>> ], dtype=int).T))
        <<< [ 1  1  1 -1]
    """
    assert Z.shape == (3, 4), "Invalid shape"

    # Computes 6 2×2 minors, each used by two of the 4 3×3 minors,
    # premultiplied by sign(0.5 - j%2)
    #   (implicitly, we simply compute them in cyclic row order)
    signed_minors_2 = np.zeros((3, 2), dtype=Z.dtype)
    for j in range(0, 2):
        for i in range(0, 3):
            jj = 2*j
            i1 = i+1 if i < 2 else 0
            i2 = i-1 if i > 0 else 2
            signed_minors_2[i, j] = Z[i1, jj]*Z[i2, jj+1] - Z[i2, jj]*Z[i1, jj+1]

    minors_3 = np.zeros((4,), dtype=Z.dtype)
    for j in range(0, 4):
        # sign of the column
        s = 1 if j%2==0 else -1
        # complementary column, multiplied by the 2×2 minors
        jj = j + s
        jm = 1 - j//2
        for i in range(0, 3):
            minors_3[j] += s * Z[i, jj] * signed_minors_2[i, jm]

    return minors_3

def simplify_zonotope_generators(Z: np.ndarray[tuple, int]) -> np.ndarray[tuple, int]:
    """
    Simplifies the generators of a zonotope, removing collinear generators at the expense
    of translating the zonotope if necessary.
    """
    assert Z.ndim == 2, "Invalid shape"
    Z = Z.copy()
    i = 0
    while i < Z.shape[1]:
        gen = Z[:, i]
        res = gen.copy()
        deleted = []
        for j in range(i+1, Z.shape[1]):
            if not linalg.is_full_rank(Z[:, (i, j)]):
                deleted.append(j)
                res += Z[:, j] if gen.dot(Z[:, j]) > 0 else -Z[:, j]
        if deleted:
            Z[:, i] = res
            Z = np.delete(Z, deleted, axis=1)
        i += 1
    return Z

def zonotope_lattice_width(Z: tuple | list | np.ndarray[tuple, int]) -> int:
    """
    Computes the lattice width of an N-dimensional M-zonotope.

    First, collinear generators are simplified.
    Then, for each choice of signs of the remaining generators, we pose a MILP
    to find the minimum lattice width in the cone spanned by our choice of signs.
    (Due to symmetry, we can fix one of the signs.)

    The result is the minimum across the sign choices.
    """
    if isinstance(Z, (tuple, list)):
        Z = np.array(Z, dtype=int)
    assert Z.ndim == 2 and np.issubdtype(Z.dtype, np.integer)
    Z = simplify_zonotope_generators(Z)
    if not linalg.is_full_rank(Z):
        return 0

    min_lattice_width = np.sum(np.abs(Z[0, :]))  # Initial upper bound

    n = Z.shape[0]
    m = Z.shape[1]
    indices = np.arange(m-1, -1, -1, dtype=int)
    # 2**(m - 1), since half of the directions do not need to be checked due to symmetry
    for signs_mask in range(2**(m - 1)):
        signs = -1 + 2*np.mod(np.right_shift(signs_mask, indices), 2)
        # signs = np.array((1,) + tuple(1 if (signs_mask >> sh)%2 == 0 else -1 for sh in range(3)), dtype=int)
        signed_Z = Z * signs[np.newaxis, :]
        gg = signed_Z.sum(axis=1)  # = Z @ signs

        if np.all(gg == 0):
            continue  # This choice of signs is infeasible, skip MILP solver

        result = milp(
            # Minimize lattice width in directions within the quadrant determined by `signs`
            c=gg,
            # `milp` defaults to non-negative bounds
            # In our case, we could constrain one axis at most (we do not for simplicity)
            bounds=Bounds(-np.inf, np.inf),
            # All variables must be integer
            integrality=1,
            constraints=(
                # Require found direction to cross every generator non-negatively with our choice of signs
                LinearConstraint(signed_Z.T, lb=0),
                # Avoid degenerate direction
                LinearConstraint(gg[np.newaxis, :], lb=1),
            ),
        )

        if not result.success:
            # result.status:
            #   0: Optimal solution found.
            #   1: Iteration or time limit reached.
            #   2: Problem is infeasible.
            #   3: Problem is unbounded.
            #   4: Other; see message for details.
            if result.status == 2:
                # The posed MILP will be infeasible if the choice of signs defines the full space as its search cone
                #   i.e., if no single direction is positive for all signed generators.
                #   One way to check this would be checking if the origin is contained in the convex hull
                #   of our signed generators (since we already assume full rank)
                #   For the case with many generators we could find a better strategy
                #   to iterate through all feasible sign choices directly, avoiding these cases.
                continue
            raise ValueError(f"zonotope_lattice_width > milp > error:\n  {result.message}")
        else:
            candidate = int(result.fun)
            min_lattice_width = min(min_lattice_width, candidate)
            # if min_lattice_width <= 3:
            #     break
    return min_lattice_width


def sLRC_primitive_volume_vectors(
    n=4, max_volume_inclusive: int = 195,
    *,
    min_volume_inclusive: int = ...,  # 1+2+...+n
    order: Literal['grlex', 'revgrlex', 'grevlex', 'revgrevlex', 'lex'] = 'grlex',
) -> Iterable[tuple[int, int, int, int]]:
    if n < 1:
        return
    def sLRC_volume_vectors_grlex(volume_range: range):
        def rec(tup, mn, rem, g, m):
            if m == 1:
                if rem < mn:
                    return False
                if gcd(rem, g) == 1:
                    yield tup + (rem,)
            else:
                # Remaining volume split into `m` remaining entries
                # adjusted for a monotone sequence, rounded up
                mx = rem//m - (m-1)//2
                if mx < mn:
                    return False
                for v in range(mn, mx+1):
                    if not (yield from rec(tup + (v,), v+1, rem-v, gcd(v, g), m-1)):
                        break
            return True

        for vol in volume_range:
            yield from rec((), 1, vol, 0, n)

    def sLRC_volume_vectors_grevlex(volume_range: range):
        def rec(tup, ub, rem, g, m):
            if m == 1:
                if rem > ub:
                    return False
                if gcd(rem, g) == 1:
                    yield (rem,) + tup
            else:
                # Account for minimum remaining volume to be allocated
                #   1 + 2 + ... + m-1 = (m-1)*m/2
                mx = min(rem - (m-1)*m//2, ub)
                if mx < m:
                    return False
                for v in range(mx, m-1, -1):
                    if not (yield from rec((v,) + tup, v-1, rem - v, gcd(v, g), m-1)):
                        break
            return True

        for vol in volume_range:
            yield from rec((), vol, vol, 0, n)

    def sLRC_volume_vectors_lex(volume_range: range):
        if volume_range.step > 0:
            v_min = volume_range.start
            v_max = (volume_range.stop - volume_range.start) // volume_range.step * volume_range.step + volume_range.start - 1
        else:
            v_max = volume_range.start
            v_min = (volume_range.stop - volume_range.start) // volume_range.step * volume_range.step + volume_range.start + 1
        if v_max < v_min:
            return
        def rec(tup, mn, rem, g, m):
            if m == 1:
                for v in range(max(mn, v_min - (v_max - rem)), rem+1):
                    if gcd(v, g) == 1:
                        yield tup + (v,)
            else:
                # Remaining volume split into `m` remaining entries
                # adjusted for a growing sequence, rounded up
                mx = rem//m - (m-1)//2
                if mx < mn:
                    return False
                for v in range(mn, mx+1):
                    if not (yield from rec(tup + (v,), v+1, rem-v, gcd(v, g), m-1)):
                        break
            return True
        yield from rec((), 1, v_max, 0, n)

    if min_volume_inclusive is ...:
        min_volume_inclusive = n*(n+1)//2

    if order == 'grlex':
        yield from sLRC_volume_vectors_grlex(range(min_volume_inclusive, max_volume_inclusive + 1))
    elif order == 'revgrlex':
        yield from sLRC_volume_vectors_grlex(range(max_volume_inclusive, min_volume_inclusive - 1, -1))
    elif order == 'grevlex':
        yield from sLRC_volume_vectors_grevlex(range(min_volume_inclusive, max_volume_inclusive + 1))
    elif order == 'revgrevlex':
        yield from sLRC_volume_vectors_grevlex(range(max_volume_inclusive, min_volume_inclusive - 1, -1))
    elif order == 'lex':
        yield from sLRC_volume_vectors_lex(range(min_volume_inclusive, max_volume_inclusive + 1))
    elif order == 'revlex':
        yield from sLRC_volume_vectors_lex(range(min_volume_inclusive, max_volume_inclusive + 1))
    else:
        raise ValueError(f"sLRC_primitive_volume_vectors > error: unknown order '{order}', must be one of ['grlex', 'revgrlex', 'grevlex', 'revgrevlex', 'lex', 'revlex']")
