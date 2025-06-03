from unittest import TestCase

import numpy as np
from fractions import Fraction

from slrz.rational.linalg import *

def is_row_hnf(A):
    """
    Check if matrix A is in row Hermite normal form.
    """
    if len(A) == 0:
        return True

    m, n = A.shape
    last_pivot = -1  # Column index of the last found pivot

    for i in range(m):
        # Find first non-zero element in this row
        pivot_found = False
        for j in range(n):
            if A[i, j] != 0:
                if j <= last_pivot:
                    return False
                if A[i, j] <= 0:
                    return False

                # Check elements above pivot are smaller than pivot
                for k in range(i):
                    if A[k, j] < 0 or A[k, j] >= A[i, j]:
                        return False

                last_pivot = j
                pivot_found = True
                break

        # If row is not all zeros, we must have found a pivot
        if not np.all(A[i, :] == 0) and not pivot_found:
            return False

    return True

def is_col_hnf(A):
    """
    Check if matrix A is in column Hermite normal form.
    """
    return is_row_hnf(A.T)


class TestLinAlg(TestCase):
    def test_pLU_decomposition_and_related(self):
        """Test PLU decomposition, inverse, and adjugate on square matrices."""
        # Test with a 3x3 matrix
        a = (
            (2, 0, 2),
            (0, 2, 1),
            (2, 1, 3),
        )
        a = as_fraction_array(a)
        n = a.shape[0]

        # Test PLU decomposition
        P, L, U, det_P = pLU_decomposition(a)
        self.assertTrue(np.all(P @ L @ U == a), "PLU decomposition failed")

        # Test inverse
        ai = inverse(a)
        self.assertTrue(np.all(ai @ a == np.eye(n, dtype=int)), "Inverse failed")

        # Test adjugate
        det = determinant(a)
        adj = adjugate(a)
        self.assertTrue(np.all(a @ adj == np.eye(n, dtype=int) * det), "Adjugate failed")

        # Test with a different 3x3 matrix
        b = (
            ( 1, -1, 1),
            ( 1, -1, 0),
            (-4,  0, 9),
        )
        b = as_fraction_array(b)

        # Test PLU decomposition
        P, L, U, det_P = pLU_decomposition(b)
        self.assertTrue(np.all(P @ L @ U == b), "PLU decomposition failed")

        # Test solve
        # Single right-hand side
        rhs = as_fraction_array([[1], [2], [3]])
        x = solve(b, rhs)
        self.assertTrue(np.all(b @ x == rhs), "Solve failed for single right-hand side")

        # Multiple right-hand sides
        rhs_multiple = as_fraction_array([[1, 4], [2, 5], [3, 6]])
        x_multiple = solve(b, rhs_multiple)
        self.assertTrue(np.all(b @ x_multiple == rhs_multiple), "Solve failed for multiple right-hand sides")

        # Test inverse
        bi = inverse(b)
        self.assertTrue(np.all(bi @ b == np.eye(n, dtype=int)), "Inverse failed")

        # Test adjugate
        det = determinant(b)
        adj = adjugate(b)
        self.assertTrue(np.all(b @ adj == np.eye(n, dtype=int) * det), "Adjugate failed")

    def test_pLUq_decomposition_full_rank_square(self):
        """Test PLUq decomposition on full-rank square matrices."""
        a = (
            (2, 0, 2),
            (0, 2, 1),
            (2, 1, 3),
        )
        a = as_fraction_array(a)

        # Test PLUq decomposition
        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(a)

        # Check decomposition
        self.assertTrue(np.all(P @ L @ U @ Q == a), "PLUq decomposition failed")

        # Check rank
        self.assertEqual(rank(a), 3)
        self.assertTrue(is_full_rank(a))

        # Test another matrix
        b = (
            ( 1, 1, 1),
            (-1, 1, 1),
            (-2, 2, 2),
        )
        b = as_fraction_array(b)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(b)
        self.assertTrue(np.all(P @ L @ U @ Q == b), "PLUq decomposition failed")

    def test_pLUq_decomposition_non_square(self):
        """Test PLUq decomposition on non-square matrices."""
        # Test with wide matrices (2x4)
        a = (
            (1, 0, 2, 3),
            (2, 1, 3, 2),
        )
        a = as_fraction_array(a)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(a)
        self.assertTrue(np.all(P @ L @ U @ Q == a), "PLUq decomposition failed")
        self.assertTrue(is_full_rank(a))

        # Test with a rank-deficient wide matrix
        b = (
            (1, 0, 2, 3),
            (2, 0, 4, 6),
        )
        b = as_fraction_array(b)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(b)
        self.assertTrue(np.all(P @ L @ U @ Q == b), "PLUq decomposition failed")
        self.assertEqual(rank(b), 1)
        self.assertFalse(is_full_rank(b))

        # Test with duplicated row (3x4)
        c = (
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 2, 1),
        )
        c = as_fraction_array(c)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(c)
        self.assertTrue(np.all(P @ L @ U @ Q == c), "PLUq decomposition failed")

        # Test with tall matrix
        d = (
            (1, 2),
            (2, 3),
            (4, 5),
            (6, 7),
        )
        d = as_fraction_array(d)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(d)
        self.assertTrue(np.all(P @ L @ U @ Q == d), "PLUq decomposition failed")
        self.assertTrue(is_full_rank(d))

        # Test with rank-deficient tall matrix
        d = (
            (1, 2),
            (2, 4),
            (4, 8),
            (6, 12),
        )
        d = as_fraction_array(d)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(d)
        self.assertTrue(np.all(P @ L @ U @ Q == d), "PLUq decomposition failed")
        self.assertFalse(is_full_rank(d))

        # Test with a zero matrix
        e = (
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        e = as_fraction_array(e)

        P, L, U, Q, det_P, det_Q, det_PQ = pLUq_decomposition(e)
        self.assertTrue(np.all(P @ L @ U @ Q == e), "PLUq decomposition failed")
        self.assertEqual(rank(e), 0)
        self.assertFalse(is_full_rank(e))

    def test_determinant(self):
        """Test the determinant function."""
        # Test with a 2x2 matrix
        a = np.array([
            [1, 2],
            [3, 4]
        ], dtype=int)

        det = determinant(a)
        self.assertEqual(det, -2)

        # Test with a 3x3 matrix
        b = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=int)

        det = determinant(b)
        self.assertEqual(det, 0)  # This matrix is singular

        # Test with a non-singular 3x3 matrix
        c = np.array([
            [2, 0, 2],
            [0, 2, 1],
            [2, 1, 3]
        ], dtype=int)

        det = determinant(c)
        self.assertEqual(det, 2)

        # Test with PLU decomposition provided
        plu = pLU_decomposition(c)
        det_with_plu = determinant(c, pLU=plu)
        self.assertEqual(det_with_plu, 2)

    def test_row_hnf(self):
        """Test row Hermite normal form computation."""
        # Test with a 2x2 matrix
        a = np.array([
            [4, 7],
            [2, 3],
        ], dtype=int)
        h, u = row_hnf(a)
        print('a', a, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(u @ a == h), "Row HNF transformation failed")
        self.assertTrue(is_row_hnf(h), "Result is not in row HNF")

        # Test with a 3x3 matrix
        b = np.array([
            [2, 4, 4],
            [-6, 6, 12],
            [10, -4, -16]
        ], dtype=int)
        h, u = row_hnf(b)
        print('b', b, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(u @ b == h), "Row HNF transformation failed")
        self.assertTrue(is_row_hnf(h), "Result is not in row HNF")

        # Test with a non-square matrix (3x2)
        c = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=int)
        h, u = row_hnf(c)
        print('c', c, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(u @ c == h), "Row HNF transformation failed")
        self.assertTrue(is_row_hnf(h), "Result is not in row HNF")

        d = np.array([
            [4, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
        ], dtype=int)
        h, u = row_hnf(d)
        print('d', d, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(u @ d == h), "Row HNF transformation failed")
        self.assertTrue(is_row_hnf(h), "Result is not in row HNF")
    
    def test_col_hnf(self):
        """Test column Hermite normal form computation."""
        # Test with a 2x2 matrix
        a = np.array([
            [4, 7],
            [2, 3],
        ], dtype=int)
        h, u = col_hnf(a)
        print('a', a, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(a @ u == h), "Column HNF transformation failed")
        self.assertTrue(is_col_hnf(h), "Result is not in column HNF")

        # Test with a 3x3 matrix
        b = np.array([
            [ 2,  4,   4],
            [-6,  6,  12],
            [10, -4, -16]
        ], dtype=int)
        h, u = col_hnf(b)
        print('b', b, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(b @ u == h), "Column HNF transformation failed")
        self.assertTrue(is_col_hnf(h), "Result is not in column HNF")

        # Test with a non-square matrix (2x3)
        c = np.array([
            [1, 3, 5],
            [2, 4, 6]
        ], dtype=int)
        h, u = col_hnf(c)
        print('c', c, 'h', h, 'u', u, sep='\n')
        self.assertTrue(np.all(c @ u == h), "Column HNF transformation failed")
        self.assertTrue(is_col_hnf(h), "Result is not in column HNF")

    def test_update_inverse(self):
        """Test the Sherman-Morrison inverse matrix update."""
        # Start with a simple matrix
        A = as_fraction_array(np.array([
            [2, 0],
            [0, 3]
        ], dtype=int))

        # Its inverse
        Ai = inverse(A)

        # Rank-1 update vectors
        u = as_fraction_array(np.array([[1], [1]], dtype=int))
        v = as_fraction_array(np.array([[1], [1]], dtype=int))

        # Apply update
        Auv, Auv_i = update_inverse(A, Ai, u, v)

        # Check that Auv = A + u@v.T
        expected_Auv = A + u @ v.T
        self.assertTrue(np.all(Auv == expected_Auv))

        # Check that Auv_i is the inverse of Auv
        identity = np.eye(2, dtype=int) * Fraction(1, 1)
        self.assertTrue(np.all(Auv @ Auv_i == identity))
        self.assertTrue(np.all(Auv_i @ Auv == identity))

    def test_update_adjugate(self):
        """Test the Sherman-Morrison formula for adjugate update."""
        # Start with a simple matrix
        A = np.array([
            [2, 0],
            [0, 3]
        ], dtype=int)

        # Its adjugate
        Adj = adjugate(A)

        # Rank-1 update vectors
        u = np.array([1, 1], dtype=int)
        v = np.array([1, 1], dtype=int)

        # Apply update
        Auv, Auv_adj = update_adjugate(A, Adj, u, v)

        # Check that Auv = A + u@v.T
        expected_Auv = A + np.outer(u, v)
        self.assertTrue(np.all(Auv == expected_Auv))

        # Check that Auv_adj is the adjugate of Auv
        det_Auv = determinant(Auv)
        self.assertEqual(det_Auv, Auv[0, :].dot(Auv_adj[:, 0]))

        identity_scaled = np.eye(2, dtype=int) * det_Auv
        self.assertTrue(np.all(Auv @ Auv_adj == identity_scaled))

if __name__ == '__main__':
    import unittest
    unittest.main()
    