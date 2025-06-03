from typing import Iterable

import numpy as np

from fractions import Fraction
from timeit import timeit

from slrz.rational.linalg import as_fraction_array

from slrz.lll.nlll import (
    update_rational_gram_schmidt,
    lll_reduction_big_int, lll_reduction_numba, lll_reduction_numpy
)

from unittest import TestCase


class TestNLLL(TestCase):
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

        ref = np.array(gramschmidt(tuple(A.T))).T
        print(f"ref:\n{ref.astype(float)}")

        # Numerators of the orthogonalized basis of A
        ortho_num = A.copy('F')

        # Denominators of the columns of ortho_num:
        ortho_den: np.ndarray[tuple[int], int] = np.ones_like(A, shape=(n,))

        # Self dots of the columns of ortho_num:
        #   Implicit denominator: ortho_den**2
        ortho_sdot_num: np.ndarray[tuple[int], int] = np.zeros_like(A, shape=(n,))

        # The non-diagonal entries from the R matrix from the QR decomposition:
        #   Denominator: ortho_sdot_num
        r_num: np.ndarray[tuple[int, int], int] = np.zeros_like(A)

        for i in range(n):
            update_rational_gram_schmidt(A, ortho_num, ortho_den, ortho_sdot_num, r_num, i)

        print(f"ortho_num:\n{ortho_num}")
        print(f"ortho_den:\n{ortho_den}")
        ortho = ortho_num.astype(float) / ortho_den[np.newaxis, :].astype(float)
        print(f"ortho:\n{ortho}")

        print(f"ortho_sdot_num: {ortho_sdot_num}")

        self.assertTrue(
            np.all(as_fraction_array(ortho_num) / as_fraction_array(ortho_den[np.newaxis, :]) == ref),
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

    def test_lll_reduction_numba(self):
        try:
            import numba
        except ImportError:
            self.skipTest("Numba is not installed.")

        # 3x3 test
        A = np.array((
            ( 1, 1, 1),
            (-1, 0, 2),
            ( 3, 5, 6),
        )).T

        reduced = lll_reduction_numba(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 0, 1, 0),
            ( 1, 0, 1),
            (-1, 0, 2),
        )).T), "lll_reduction_numba does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_numba(A, delta=0.75), number=100) / 100
        print(f"lll_reduction_numba(3x3) took {t} s")

        # 4x4 test
        A = np.array((
            (105, 821, 404, 328),
            (881, 667, 644, 927),
            (181, 483,  87, 500),
            (893, 834, 732, 441),
        )).T

        try:
            lll_reduction_numba(A, delta=0.75)
            self.fail("lll_reduction_numba was expected to raise an OverflowError for this input.")
        except OverflowError:
            pass

        # 4x3 test (LRZ.from_volume_vector((1, 2, 3, 4))
        A = np.array((
            ( 4, 0, 0, 1),
            (-2, 1, 0, 0),
            (-3, 0, 1, 0),
        )).T

        reduced = lll_reduction_numba(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 1, 0,  1, 1),
            (-1, 1,  1, 1),
            ( 1, 1, -1, 0),
        )).T), "lll_reduction_numba does not match reference implementation.")

        t = timeit(lambda: lll_reduction_numba(A, 0.75), number=100) / 100
        print(f"lll_reduction_numba(4x3) took {t} s")

    def test_lll_reduction_numpy(self):
        # 3x3 test
        A = np.array((
            ( 1, 1, 1),
            (-1, 0, 2),
            ( 3, 5, 6),
        )).T

        reduced = lll_reduction_numpy(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 0, 1, 0),
            ( 1, 0, 1),
            (-1, 0, 2),
        )).T), "lll_reduction_numpy does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_numpy(A, delta=0.75), number=100) / 100
        print(f"lll_reduction_numpy(3x3) took {t} s")

        # 4x4 test
        A = np.array((
            (105, 821, 404, 328),
            (881, 667, 644, 927),
            (181, 483,  87, 500),
            (893, 834, 732, 441),
        )).T

        try:
            lll_reduction_numpy(A, delta=0.75)
            self.fail("lll_reduction_numpy was expected to raise an OverflowError for this input.")
        except OverflowError:
            pass

        # 4x3 test (LRZ.from_volume_vector((1, 2, 3, 4))
        A = np.array((
            ( 4, 0, 0, 1),
            (-2, 1, 0, 0),
            (-3, 0, 1, 0),
        )).T

        reduced = lll_reduction_numpy(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 1, 0,  1, 1),
            (-1, 1,  1, 1),
            ( 1, 1, -1, 0),
        )).T), "lll_reduction_numpy does not match reference implementation.")

        t = timeit(lambda: lll_reduction_numpy(A, 0.75), number=100) / 100
        print(f"lll_reduction_numpy(4x3) took {t} s")

    def test_lll_reduction_big_int(self):
        # 3x3 test
        A = np.array((
            ( 1, 1, 1),
            (-1, 0, 2),
            ( 3, 5, 6),
        )).T

        reduced = lll_reduction_big_int(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 0, 1, 0),
            ( 1, 0, 1),
            (-1, 0, 2),
        )).T), "lll_reduction_big_int does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_big_int(A, delta=0.75), number=100) / 100
        print(f"lll_reduction_big_int(3x3) took {t} s")

        # 4x4 test
        A = np.array((
            (105, 821, 404, 328),
            (881, 667, 644, 927),
            (181, 483,  87, 500),
            (893, 834, 732, 441),
        )).T

        reduced = lll_reduction_big_int(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 76, -338, -317,  172),
            ( 88, -171, -229, -314),
            (269,  312, -142,  186),
            (519, -299,  470,  -73),
        )).T), "lll_reduction_big_int does not match reference implementation.")

        # Timing
        t = timeit(lambda: lll_reduction_big_int(A, 0.75), number=100) / 100
        print(f"lll_reduction_big_int(4x4) took {t} s")

        # 4x3 test (LRZ.from_volume_vector((1, 2, 3, 4))
        A = np.array((
            ( 4, 0, 0, 1),
            (-2, 1, 0, 0),
            (-3, 0, 1, 0),
        )).T

        reduced = lll_reduction_big_int(A, delta=0.75)
        print(f"reduced:\n{reduced}")
        self.assertTrue(np.all(reduced == np.array((
            ( 1, 0,  1, 1),
            (-1, 1,  1, 1),
            ( 1, 1, -1, 0),
        )).T), "lll_reduction_big_int does not match reference implementation.")

        t = timeit(lambda: lll_reduction_big_int(A, 0.75), number=100) / 100
        print(f"lll_reduction_big_int(4x3) took {t} s")