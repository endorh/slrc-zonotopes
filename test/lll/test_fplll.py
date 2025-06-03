from typing import Iterable

import numpy as np

from fractions import Fraction
from timeit import timeit

from slrz.lll.fplll import (
    update_fp_gram_schmidt,
    lll_reduction_fp,
    lll_reduction_fp_big_int,
)

from unittest import TestCase


class TestFPLLL(TestCase):
    def check_gram_schmidt(self, A: np.ndarray[tuple[int, int], int] | tuple[tuple[int, ...], ...]):
        def gramschmidt(v: Iterable[np.ndarray]) -> list[np.ndarray]:
            u: list[np.ndarray] = []
            for vi in v:
                ui = np.array(vi)
                for uj in u:
                    ui = ui - Fraction(uj.dot(vi), uj.dot(uj)) * uj
                if any(ui):
                    u.append(ui)
            return u

        A: np.ndarray[tuple[int, int], int] = np.array(A)
        n = A.shape[0]
        m = A.shape[1]

        ref = np.array(gramschmidt(tuple(A.T))).T
        print(f"ref:\n{ref}")

        # Numerators of the orthogonalized basis of A
        ortho: np.ndarray[tuple[int, int], float] = A.copy('F').astype(float)
        basis_fp = ortho.copy('F')

        # Self dots of the columns of ortho_num:
        ortho_sdot: np.ndarray[tuple[int], float] = np.zeros((n,), dtype=float)

        # The non-diagonal entries from the R matrix from the QR decomposition:
        r: np.ndarray[tuple[int, int], float] = np.zeros((n, m), dtype=float)

        for i in range(n):
            update_fp_gram_schmidt(basis_fp, ortho, ortho_sdot, r, i)

        print(f"ortho:\n{ortho}")

        print(f"ortho_sdot_num: {ortho_sdot}")

        self.assertTrue(
            np.allclose(ortho, ref.astype(float)),
            "Gram Schmidt implementation is not correct.")

    def test_gram_schmidt(self):
        self.check_gram_schmidt((
            ( 1, 1, 1),
            (-1, 0, 2),
            ( 3, 5, 6),
        ))

        self.check_gram_schmidt((
            (1, -1, 3),
            (1,  0, 5),
            (1,  2, 6),
        ))

    def test_lll_reduction_fp(self):
        delta = 0.75

        # 3x3 test
        A = np.array((
            (1, 1, 1),
            (-1, 0, 2),
            (3, 5, 6),
        )).T

        reduced = lll_reduction_fp(A, delta=delta)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 0, 1, 0),
            ( 1, 0, 1),
            (-1, 0, 2),
        )).T), "LLL reduction does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_fp(A, delta=delta), number=1000) / 1000
        print(f"lll_reduction_fp(3x3) took {t} s")

        # 4x4 test
        A = np.array((
            (105, 821, 404, 328),
            (881, 667, 644, 927),
            (181, 483,  87, 500),
            (893, 834, 732, 441),
        )).T

        reduced = lll_reduction_fp(A, delta=delta)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 76, -338, -317,  172),
            ( 88, -171, -229, -314),
            (269,  312, -142,  186),
            (519, -299,  470,  -73),
        )).T), "LLL reduction does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_fp(A, delta), number=1000) / 1000
        print(f"lll_reduction_fp(4x4) took {t} s")

        # 4x3 test (LRZ.from_volume_vector((1, 2, 3, 4))
        A = np.array((
            ( 4, 0, 0, 1),
            (-2, 1, 0, 0),
            (-3, 0, 1, 0),
        )).T

        reduced = lll_reduction_fp(A, delta=delta)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 1, 0,  1, 1),
            (-1, 1,  1, 1),
            ( 1, 1, -1, 0),
        )).T), "LLL reduction does not match reference implementation.")

        t = timeit(lambda: lll_reduction_fp(A, delta), number=1000) / 1000
        print(f"lll_reduction_fp(4x3) took {t} s")

    def test_lll_reduction_fp_big_int(self):
        delta = 0.75

        # 3x3 test
        A = np.array((
            ( 1, 1, 1),
            (-1, 0, 2),
            ( 3, 5, 6),
        ), dtype=object).T

        reduced = lll_reduction_fp_big_int(A, delta=delta)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 0, 1, 0),
            ( 1, 0, 1),
            (-1, 0, 2),
        )).T), "LLL reduction does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_fp_big_int(A, delta=delta), number=1000) / 1000
        print(f"lll_reduction_fp_big_int(3x3) took {t} s")

        # 4x4 test
        A = np.array((
            (105, 821, 404, 328),
            (881, 667, 644, 927),
            (181, 483,  87, 500),
            (893, 834, 732, 441),
        ), dtype=object).T

        reduced = lll_reduction_fp_big_int(A, delta=delta)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 76, -338, -317,  172),
            ( 88, -171, -229, -314),
            (269,  312, -142,  186),
            (519, -299,  470,  -73),
        )).T), "LLL reduction does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_fp_big_int(A, delta), number=1000) / 1000
        print(f"lll_reduction_fp_big_int(4x4) took {t} s")

        # 4x3 test (LRZ.from_volume_vector((1, 2, 3, 4))
        A = np.array((
            ( 4, 0, 0, 1),
            (-2, 1, 0, 0),
            (-3, 0, 1, 0),
        ), dtype=object).T

        reduced = lll_reduction_fp_big_int(A, delta=delta)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 1, 0,  1, 1),
            (-1, 1,  1, 1),
            ( 1, 1, -1, 0),
        )).T), "LLL reduction does not match reference implementation.")

        t = timeit(lambda: lll_reduction_fp_big_int(A, delta), number=1000) / 1000
        print(f"lll_reduction_fp_big_int(4x3) took {t} s")