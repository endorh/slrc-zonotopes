from __future__ import annotations

import itertools

from fractions import Fraction
from typing import Literal, NamedTuple
from warnings import warn

import numpy as np

from slrz.slrc_certificate import (
    DyadicFundamentalDomain,
    ArrangedDyadicNode,
    ArrangedDyadicVoxel,
)
from slrz.rational.milp import milp, LinearConstraint, Bounds, SciPyMILPOptions
from slrz.util.optional_numba import njit

class VectorDBinaryTree:
    """
    2ᵈ-regular tree of vectors.
    """
    def __init__(self, children: np.ndarray[tuple[Literal[2], ...], VectorDBinaryTree | np.ndarray]):
        self.children = children

    def dyadic_fundamental_domain(self) -> DyadicFundamentalDomain:
        """
        Create a dyadic fundamental domain from this tree.
        """
        def convert(
            node: VectorDBinaryTree | np.ndarray
        ) -> ArrangedDyadicNode | ArrangedDyadicVoxel:
            if isinstance(node, VectorDBinaryTree):
                return ArrangedDyadicNode([
                    convert(child)
                    for child in node.children.flat
                ])
            else:
                return ArrangedDyadicVoxel(node)
        return DyadicFundamentalDomain(convert(self))

    def central_flip(self) -> VectorDBinaryTree:
        """
        Return the central mirror of this tree.
        """
        children = np.empty_like(self.children)
        for subcube in itertools.product((0, 1), repeat=3):
            flipped_subcube = 1 - np.array(subcube)
            child = self.children[(*flipped_subcube,)]
            if isinstance(child, VectorDBinaryTree):
                children[subcube] = child.central_flip()
            else:
                children[subcube] = -child
        return VectorDBinaryTree(children)

    @staticmethod
    def from_centrally_symmetric_top_half(
        top_children: np.ndarray[tuple[Literal[2], ...], VectorDBinaryTree | np.ndarray]
    ) -> VectorDBinaryTree:
        """
        Construct a full tree by mirroring the upper half.
        """
        children = np.empty_like(top_children, shape=top_children.shape + (2,))
        children[..., 1] = top_children
        for subcube_xy in itertools.product((0, 1), repeat=top_children.ndim):
            subcube = subcube_xy + (0,)
            subcube_source = 1 - np.array(subcube_xy)
            child = top_children[(*subcube_source,)]
            if isinstance(child, VectorDBinaryTree):
                children[subcube] = child.central_flip()
            else:
                children[subcube] = -child
        return VectorDBinaryTree(children)


class VectorDBinaryTreeInsertion(NamedTuple):
    """
    Describes an open insertion point in a VectorDBinaryTree, associated
    with a `center` vector in ℝ³.
    """
    parent: VectorDBinaryTree
    index: np.ndarray[tuple[Literal[3]], int]
    center: np.ndarray[tuple[Literal[3]], float]


def polytope_covering_radius_denominator_bound(
    A: np.ndarray[tuple[int, int], int],
    b: np.ndarray[tuple[int], int],
    rough=False,
) -> int:
    if rough:
        A_b = np.concatenate((A, -b[:, np.newaxis]), axis=1)
        return int(np.ceil(np.sqrt(np.abs(np.linalg.det(A_b.T @ A_b)))))
    return _polytope_covering_radius_denominator_bound(A, b)

@njit
def _polytope_covering_radius_denominator_bound(
    A: np.ndarray[tuple[int, int], int],
    b: np.ndarray[tuple[int], int],
) -> int:
    """
    Compute an upper bound for the covering radius of a polytope given by the inequalities:
        Ax <= b.

    :param A:
        Facet normals.
    :param b:
        Facet normal levels.
    :param rough:
        If `True`, use a rougher, but easier to compute, bound, derived from
        applying Cauchy-Binet.
    :return:
        An upper bound for the covering radius of a polytope given by {x ∈ R: Ax <= b}.
    """
    ans = 1
    m = A.shape[0]
    n = A.shape[1]
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    A_b = np.zeros((4, n+1), dtype=np.float64)
    for i_0 in range(m):
        A_b[0, :n] = A[i_0]
        A_b[0, n] = -b[i_0]
        for i_1 in range(i_0+1, m):
            A_b[1, :n] = A[i_1]
            A_b[1, n] = -b[i_1]
            for i_2 in range(i_1+1, m):
                A_b[2, :n] = A[i_2]
                A_b[2, n] = -b[i_2]
                for i_3 in range(i_2+1, m):
                    A_b[3, :n] = A[i_3]
                    A_b[3, n] = -b[i_3]

                    abs_det = int(np.round(np.abs(np.linalg.det(A_b))))
                    ans = max(ans, abs_det)
    return ans


def negative_rough_margin(A, b, cov_radius: Fraction) -> Fraction:
    """
    Compute a negative rough margin to simplify the MILP problems in
    tests for tightness.

    This margin cannot be used to certify an upper bound on the covering radius.

    :param A:
        Facet normals.
    :param b:
        Facet levels.
    :param cov_radius:
        Covering radius.
    """
    return Fraction(-1, cov_radius.denominator * int(np.ceil(np.sqrt(np.abs(np.linalg.det(A.T @ A))))))


def scaled_polytope_covers(
    A: np.ndarray[tuple[int, int], int],
    b: np.ndarray[tuple[int], int],
    r: Fraction, margin: Fraction = Fraction(0, 1), *,
    bounds: Bounds | tuple[float | int, float | int] = Bounds(-np.inf, np.inf),
    assume_centrally_symmetric: bool = False,
    assume_centered: bool = False,
    expect_possible_divergence: bool = False,
    min_side: float = 1e-7,
) -> DyadicFundamentalDomain | None:
    """
    Check whether a polytope {x ∈ ℝᵈ: Ax <= b}, scaled by `r + margin`, covers the entire
    space ℝᵈ by integer translations.

    To do so, it finds a dyadic fundamental domain that fits within scaled polytope.

    The implementation uses numerical methods for solving Integer Linear Programs (ILPs).
    This implies that results may not always be correct.
    In particular, this function may report false negatives if the ILP solver fails to
    encounter a valid arrangement for a voxel at any step.
    False positives should be rare, but may arise from numerical instabilities, as the
    solutions from the ILP solver, despite being checked for correctness, they're only
    checked under floating-point arithmetic.

    To verify the solution with exact arithmetic, consider wrapping in a `SLRCCertificate`
    and using its `is_valid` method.

    :param A:
        Facet normals.
    :param b:
        Facet levels.
    :param r:
        Covering radius.
    :param margin:
        Covering radius margin.
    :param bounds:
        Bounds used for the ILPs.
    :param assume_centrally_symmetric:
        Whether to assume that the polytope is centrally symmetric.
    :param assume_centered:
        Whether to assume that if a unit cube fits inside the zonotope,
        it'll be the centered unit cube.
    :param expect_possible_divergence:
        Whether not to emit a warning if the dyadic voxels become too small.
    :param min_side:
        Minimum length allowed for the side of a dyadic voxel.
        If reached, the algorithm will return None and emit a warning unless
        `expect_possible_divergence` is set to `True`.
    :return:
        A DyadicFundamentalDomain if the polytope covers the entire space, None otherwise.
    """
    if assume_centrally_symmetric:
        assume_centered = True

    rr = r + margin
    A = A.astype(np.float64) * rr.denominator
    b = b.astype(np.float64) * rr.numerator

    # Fast path: if the unit cube fits within the polytope, return directly
    #   A [-1/2, 1/2]^d <= b
    #   A [-1, 1]^d <= 2b
    #   abs(A) (1, 1)^t <= 2b
    if assume_centered:
        if np.all(np.sum(np.abs(A), axis=1) <= 2*b):
            return DyadicFundamentalDomain(ArrangedDyadicVoxel(np.zeros(3, dtype=int)))
    else:
        # Test whether the voxel can be arranged within the polytope
        #     Constraints: A·(x + [-side/2, side/2]) <= b
        #     Rearranging: A·x <= b - A·[-side/2, side/2]
        b_adjusted = b - np.sum(np.abs(A / 2), axis=1)
        result = milp(
            c=np.zeros(3),
            constraints=LinearConstraint(A, ub=b_adjusted),
            integrality=np.ones(3),
            bounds=bounds,
        )

        # Found a translation of the unit cube that fits within the zonotope
        if result.success:
            pos = result.x.astype(int)
            return DyadicFundamentalDomain(ArrangedDyadicVoxel(pos))
        if result.status == 1:
            warn("MILP solver reached iteration/time limit!")

    # Initialize root and first level of expanded voxels
    root = VectorDBinaryTree(np.empty((2,) * 3, dtype=object))
    next_voxels: list[VectorDBinaryTreeInsertion] = []
    next_side: float = 1/2
    for dz in range(1, 2) if assume_centrally_symmetric else range(2):
        for dy in range(2):
            for dx in range(2):
                index = np.array((dx, dy, dz), dtype=int)
                new_center = index/2 - 1/4
                next_voxels.append(VectorDBinaryTreeInsertion(root, index, new_center))

    neg_test_tolerance = np.array([np.linalg.norm(row) for row in A]) * 1e-5

    while next_voxels:
        # Advance to the next depth level
        voxels, next_voxels = next_voxels, []
        side, next_side = next_side, next_side/2

        # Check for maximum depth
        if side < min_side:
            if not expect_possible_divergence:
                warn(f"Voxel side became too small while building dyadic fundamental domain, aborted!")
            return None

        # Process all voxels in this depth level
        for parent, index, center in voxels:
            # Test whether the voxel can be arranged within the polytope
            #     Constraints: A·(x + center + [-side/2, side/2]) <= b
            #     Rearranging: A·x <= b - A·[-side/2, side/2] - A·center
            b_adjusted = b - np.sum(np.abs(A * side/2), axis=1) - A @ center

            # Solve the ILP, our wrapper automatically checks that the rounded solution is admissible
            result = milp(
                c=np.ones(3),
                constraints=LinearConstraint(A, ub=b_adjusted),
                integrality=np.ones(3),
                bounds=bounds,
            )

            if result.success:
                if not np.all(A @ np.round(result.x).astype(int) <= b_adjusted):
                    print(" ! FAILED SANITY CHECK")
                # Add leaf to the parent and move on to the next voxel
                parent.children[(*index,)] = result.x.astype(int)
                continue
            if result.status == 1:
                warn("MILP solver reached iteration/time limit!")

            # Try to use the center as a negative certificate (with a generous tolerance)
            #     Constraints: A·(x + center) <= b + 1e-5 |A_i|
            #     Rearranging: A·x <= b - A·center
            b_adjusted = b - A @ center + neg_test_tolerance
            result = milp(
                c=np.ones(3),
                constraints=LinearConstraint(A, ub=b_adjusted),
                integrality=np.ones(3),
                bounds=bounds,
            )

            # Negative certificate found, we just return `None`
            if not result.success:
                return None
            if result.status == 1:
                warn("MILP solver reached iteration/time limit!")

            # Add children to next_voxels
            node = VectorDBinaryTree(np.empty((2,) * 3, dtype=object))
            parent.children[(*index,)] = node
            for dz in range(2):
                for dy in range(2):
                    for dx in range(2):
                        index = np.array((dx, dy, dz), dtype=int)
                        new_center = center + (index*side/2 - side/4)
                        next_voxels.append(VectorDBinaryTreeInsertion(node, index, new_center))

    # Fill the bottom half as the top half's central mirror
    if assume_centrally_symmetric:
        root = VectorDBinaryTree.from_centrally_symmetric_top_half(root.children[:, :, 1])

    return root.dyadic_fundamental_domain()
