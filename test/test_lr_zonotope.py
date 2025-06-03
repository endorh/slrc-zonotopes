import itertools
from math import gcd
from unittest import TestCase

import numpy as np

from slrz.lr_zonotope import sLRC_primitive_volume_vectors, zonotope_lattice_width, lr_zonotope_from_volume_vector


def baseline_sLRC_primitive_volume_vectors(n=4, min_volume: int=1, max_volume: int=10):
    for v in itertools.product(range(1, max_volume+1), repeat=n):
        if not min_volume <= sum(v) <= max_volume:
            continue
        if any(b - a < 1 for a, b in zip(v[:-1], v[1:])):
            continue
        if gcd(*v) != 1:
            continue
        yield v

class TestLRZonotope(TestCase):
    def check_sLRC_primitive_volume_vectors(self, n=4, v=10):
        def print_seq(name, seq):
            print(name)
            for e in seq:
                print(f"  - {e}")

        baseline = list(baseline_sLRC_primitive_volume_vectors(n=n, max_volume=v))
        # print_seq("baseline", baseline)

        seqs = []
        for order in ['grlex', 'grevlex', 'revgrlex', 'revgrevlex', 'lex']:
            s = list(sLRC_primitive_volume_vectors(n=n, max_volume_inclusive=v, order=order))
            # print_seq(order, s)
            seqs.append((order, s))

        first = set(baseline)
        for s in seqs:
            self.assertEqual(first, set(s[1]), f"Distinct [n={n}, v={v}]: '{s[0]}'")

    def test_sLRC_primitive_volume_vectors(self):
        for n in [2, 3, 4, 5]:
            for v in [10, 11, 12]:
                self.check_sLRC_primitive_volume_vectors(n=n, v=v)

        # self.check_sLRC_primitive_volume_vectors(n=4, v=195)
        self.assertEqual(len(list(sLRC_primitive_volume_vectors(n=4, max_volume_inclusive=195, order='grlex'))), 2133561)
        self.assertEqual(len(list(sLRC_primitive_volume_vectors(n=4, max_volume_inclusive=195, order='revgrlex'))), 2133561)
        self.assertEqual(len(list(sLRC_primitive_volume_vectors(n=4, max_volume_inclusive=195, order='grevlex'))), 2133561)
        self.assertEqual(len(list(sLRC_primitive_volume_vectors(n=4, max_volume_inclusive=195, order='revgrevlex'))), 2133561)
        self.assertEqual(len(list(sLRC_primitive_volume_vectors(n=4, max_volume_inclusive=195, order='lex'))), 2133561)

    def test_lr_zonotope_from_volume_vector(self):
        for vv in [
            (1, 2, 3, 4),
            (1, 3, 4, 6),
            (1, 3, 4, 7),
        ]:
            Z = lr_zonotope_from_volume_vector(vv)
            self.assertEqual(Z.volume_vector, vv)

        for vv in sLRC_primitive_volume_vectors(n=4, max_volume_inclusive=30):
            Z = lr_zonotope_from_volume_vector(vv)
            self.assertEqual(Z.volume_vector, vv)

    def test_lattice_width(self):
        # Unit cube in 2D
        self.assertEqual(zonotope_lattice_width((
            (1, 0),
            (0, 1),
        )), 1)

        # Unit segment + orthogonal segment
        self.assertEqual(zonotope_lattice_width((
            (2, 0),
            (0, 1),
        )), 1)

        # Unit segment + orthogonal segment
        self.assertEqual(zonotope_lattice_width((
            (1, 0),
            (0, 2),
        )), 1)

        # Scaled unit cube in 2D
        self.assertEqual(zonotope_lattice_width((
            (2, 0),
            (0, 2),
        )), 2)

        # Slanted unimodular parallelogram
        self.assertEqual(zonotope_lattice_width((
            (2, 1),
            (1, 1),
        )), 1)

        # Square of area 5 and width 3
        self.assertEqual(zonotope_lattice_width((
            (2, -1),
            (1,  2),
        )), 3)

        # Dependent generators in 2D
        self.assertEqual(zonotope_lattice_width((
            (1, 0, 1),
            (0, 1, 1),
        )), 2)

        # Unit cube in 3D
        self.assertEqual(zonotope_lattice_width((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        )), 1)

        # Dependent generators in 3D
        self.assertEqual(zonotope_lattice_width((
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        )), 2)

        # Collinear generators in 2D
        self.assertEqual(zonotope_lattice_width((
            ( 1, 0, 1, -1),
            (-1, 1, 1, -1),
        )), 3)