import unittest
import numpy as np
from fractions import Fraction

from slrz.gcd import (
    gcd, egcd, gcd_frac, gcd_array_flat, gcd_array, egcd_array_flat, egcd_array,
    gcd_arrays, lcm, lcm_array_flat, lcm_array, lcm_arrays, gcd_generic, lcm_generic, egcd_generic
)


class TestGCD(unittest.TestCase):
    def test_gcd(self):
        """Test the basic GCD function"""
        for g, expected in [
            (gcd(10, 15), 5),
            (gcd(35, 42), 7),
            (gcd(0, 5), 5),
            (gcd(5, 0), 5),
            (gcd(0, 0), 0),
            (gcd(-10, 15), 5),
            (gcd(10, -15), 5),
            (gcd(-10, -15), 5),
        ]:
            self.assertEqual(g, expected)

    def test_egcd(self):
        """Test the extended GCD function"""
        g, s, t = egcd(10, 15)
        self.assertEqual(g, 5)
        self.assertEqual(s * 10 + t * 15, 5)

        g, s, t = egcd(35, 42)
        self.assertEqual(g, 7)
        self.assertEqual(s * 35 + t * 42, 7)

        g, s, t = egcd(-10, 15)
        self.assertEqual(g, 5)
        self.assertEqual(s * (-10) + t * 15, 5)

    def test_gcd_frac(self):
        """Test GCD of fractions"""
        self.assertEqual(gcd_frac(Fraction(2, 3), Fraction(4, 5)), Fraction(2, 15))
        self.assertEqual(gcd_frac(Fraction(1, 2), Fraction(3, 4)), Fraction(1, 4))
        self.assertEqual(gcd_frac(Fraction(0, 1), Fraction(3, 7)), Fraction(3, 7))

    def test_gcd_array_flat(self):
        """Test GCD of flattened arrays"""
        self.assertEqual(gcd_array_flat(np.array([10, 15, 25], dtype=int)), 5)
        self.assertEqual(gcd_array_flat(np.array([1000, 220, 3800], dtype=int)), 20)
        self.assertEqual(gcd_array_flat(np.array([[1000, 220], [3800, 4200]], dtype=int)), 20)

    def test_gcd_array(self):
        """Test GCD of arrays with axis"""
        array = np.array([[10, 200, 30], [40, 100, 60], [25, 50, 75]], dtype=int)
        np.testing.assert_array_equal(gcd_array(array, axis=0), np.array([5, 50, 15], dtype=int))
        np.testing.assert_array_equal(gcd_array(array, axis=1), np.array([10, 20, 25], dtype=int))
        self.assertEqual(gcd_array(array, axis=(0, 1)), 5)

    def test_egcd_array_flat(self):
        """Test extended GCD of flattened arrays"""
        array = np.array([10, 15, 25], dtype=int)
        g, coefs = egcd_array_flat(array)
        self.assertEqual(g, 5)
        self.assertEqual(np.dot(coefs, array), g)

        array = np.array([20, 30, 60], dtype=int)
        g, coefs = egcd_array_flat(array)
        self.assertEqual(g, 10)
        self.assertEqual(np.dot(coefs, array), g)

    def test_egcd_array(self):
        """Test extended GCD of arrays with axis"""
        array = np.array([[10, 200, 30], [40, 100, 60], [25, 50, 75]], dtype=int)
        g, coefs = egcd_array(array, axis=0)
        np.testing.assert_array_equal(g, np.array([5, 50, 15], dtype=int))

        g, coefs = egcd_array(array, axis=1)
        np.testing.assert_array_equal(g, np.array([10, 20, 25], dtype=int))

    def test_gcd_arrays(self):
        """Test element-wise GCD of two arrays"""
        a = np.array([[10, 200, 30], [40, 100, 60], [25, 50, 75]], dtype=int)
        b = np.array([[5, 20, 15], [60, 150, 90], [12, 11, 20]], dtype=int)

        expected = np.array([
            [5, 20, 15],
            [20, 50, 30],
            [1, 1, 5]
        ], dtype=int)

        np.testing.assert_array_equal(gcd_arrays(a, b), expected)

        # Test with scalar
        np.testing.assert_array_equal(gcd_arrays(2, a), np.array([[2, 2, 2], [2, 2, 2], [1, 2, 1]], dtype=int))

    def test_lcm(self):
        """Test LCM function"""
        for l, expected in [
            (lcm(10, 15), 30),
            (lcm(4, 6), 12),
            (lcm(0, 5), 0),
            (lcm(5, 0), 0),
            (lcm(0, 0), 0),
            (lcm(-10, 15), 30),
        ]:
            self.assertEqual(l, expected)

    def test_lcm_array_flat(self):
        """Test LCM of flattened arrays"""
        array = np.array([2, 3, 4], dtype=int)
        self.assertEqual(lcm_array_flat(array), 12)

        array = np.array([10, 15, 25], dtype=int)
        self.assertEqual(lcm_array_flat(array), 150)

    def test_lcm_array(self):
        """Test LCM of arrays with axis"""
        array = np.array([[2, 3, 4], [3, 5, 7], [4, 6, 8]], dtype=int)
        np.testing.assert_array_equal(lcm_array(array, axis=0), np.array([12, 30, 56], dtype=int))
        np.testing.assert_array_equal(lcm_array(array, axis=1), np.array([12, 105, 24], dtype=int))

    def test_lcm_arrays(self):
        """Test element-wise LCM of two arrays"""
        a = np.array([[2, 3, 4], [3, 5, 7]], dtype=int)
        b = np.array([[3, 5, 6], [2, 10, 21]], dtype=int)

        expected = np.array([
            [6, 15, 12],
            [6, 10, 21]
        ], dtype=int)

        np.testing.assert_array_equal(lcm_arrays(a, b), expected)

    def test_gcd_generic(self):
        """Test generic GCD function"""
        self.assertEqual(gcd_generic(10, 15), 5)
        self.assertEqual(gcd_generic(10, 15, 20), 5)
        self.assertEqual(gcd_generic(Fraction(2, 3), Fraction(4, 5)), Fraction(2, 15))
        self.assertEqual(gcd_generic(), None)
        self.assertEqual(gcd_generic(default=0), 0)
        self.assertEqual(gcd_generic(10, -15), 5)
        self.assertEqual(gcd_generic(10, -15, abs_result=False), -5)

    def test_lcm_generic(self):
        """Test generic LCM function"""
        self.assertEqual(lcm_generic(10, 15), 30)
        self.assertEqual(lcm_generic(4, 6, 8), 24)
        self.assertEqual(lcm_generic(Fraction(1, 2), Fraction(1, 3)), Fraction(1, 1))
        self.assertEqual(lcm_generic(), 0)
        self.assertEqual(lcm_generic(zero=Fraction(0, 1)), Fraction(0, 1))

    def test_egcd_generic(self):
        """Test generic extended GCD function"""
        g, coefs = egcd_generic(10, 15)
        self.assertEqual(g, 5)
        self.assertEqual(coefs[0] * 10 + coefs[1] * 15, 5)

        g, coefs = egcd_generic(10, 15, 20)
        self.assertEqual(g, 5)
        self.assertEqual(coefs[0] * 10 + coefs[1] * 15 + coefs[2] * 20, 5)

        g, coefs = egcd_generic(Fraction(2, 3), Fraction(4, 5))
        self.assertEqual(g, Fraction(2, 15))

        g, coefs = egcd_generic()
        self.assertEqual(g, None)
        self.assertEqual(coefs, ())

        g, coefs = egcd_generic(default=0)
        self.assertEqual(g, 0)

        g, coefs = egcd_generic(10, -15)
        self.assertEqual(g, 5)

        g, coefs = egcd_generic(10, -15, abs_result=False)
        self.assertEqual(g, -5)


if __name__ == '__main__':
    unittest.main()