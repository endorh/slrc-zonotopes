from __future__ import annotations

from fractions import Fraction
from typing import Sequence
from warnings import warn

import numpy as np

from slrz.lr_zonotope import LRZonotope
from slrz.decide_polytope_cov_radius import (
    polytope_covering_radius_denominator_bound,
    negative_rough_margin,
    scaled_polytope_covers,
)
from slrz.slrc_certificate import SLRCCertificate
from slrz.serialization.lr_zonotope import load_LR_zonotope_archive
from slrz.serialization.slrc_certificate import save_npz_sLRC_certificates_archive
from slrz.util import Profiler

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    warn("tqdm not installed, covering radius computations won't display a progress bar!")
    # Declare tqdm as a passthrough iterator decorator
    def tqdm(iterator, *_, **__):
        return iterator


def build_certificates(
    zonotopes: Sequence[LRZonotope],
    cov_radius: Fraction = Fraction(3, 5),
    rough_tight_test: bool = True,
    min_side: float = 1e-7,
    check_certificates: bool = True,
    profile: bool = False,
) -> tuple[list[SLRCCertificate], list[LRZonotope], list[LRZonotope]]:
    """
    Build certificates that all given `zonotopes` have covering radius at most `cov_radius`,
    that is, that when they're scaled by a factor of `cov_radius`, their integer translations
    still cover the entire space.

    See `scaled_polytope_covers` for details.

    :param zonotopes:
        Zonotopes to build certificates for.
    :param cov_radius:
        Covering radius to certify an upper bound for.
    :param rough_tight_test:
        If `True`, use a rougher bound, but easier to compute, derived from Cauchy-Binet.
    :param refine_epsilon_for_tight_cases:
        If `True`, use the non-rough bound for tight cases.
        This will likely reduce the size of the fundamental domain.
    :param refine_non_tight_domains:
        If `True`, recompute non-tight domains with an epsilon of 0,
        to reduce their depth.
    :param check_certificates:
        Whether the certificates should be checked for validity
        under exact arithmetic during generation.
    :param profile:
        Whether the generation process should be profiled.
        If `tqdm` is installed, will display running statistics of the time
        taken by different sections of the code.
    :return:
        Three lists:
        - List of certificates.
        - Zonotopes whose covering radius was determined to be more than `cov_radius`,
          or ran into numerical stability issues during generation of the certificate.
        - Zonotopes whose covering radius is exactly `cov_radius`.
    """

    certificates: list[SLRCCertificate] = []
    failed = []
    tight = []

    # Create progress bar if tqdm is installed
    zonotopes = tqdm(zonotopes, desc="Constructing sLRC certificates...")
    profiler = Profiler(zonotopes) if profile else Profiler.noop

    for zonotope in zonotopes:
        with profiler["inequalities"]:
            A, b = zonotope.inequalities
            Ac, bc = zonotope.centrally_symmetric_inequalities

        with profiler["bounds"]:
            bounds = (
                -np.sum(np.abs(zonotope.generators), axis=1) - 2,
                +np.sum(np.abs(zonotope.generators), axis=1) + 2)

        # Attempt to find a rough non-tightness certificate first
        with profiler["rough-tight-test"]:
            neg_rough_margin = negative_rough_margin(A, b, cov_radius)
            domain = scaled_polytope_covers(
                Ac, bc, cov_radius, neg_rough_margin,
                bounds=bounds,
                assume_centrally_symmetric=True,
                expect_possible_divergence=True,
                min_side=min_side,
            )
            if domain:
                epsilon = neg_rough_margin

        with profiler["epsilon"]:
            if not domain:
                denominator_bound = polytope_covering_radius_denominator_bound(A, b, rough=False)
                epsilon = Fraction(1, 2 * cov_radius.denominator * denominator_bound)

                if epsilon < 1e-14:
                    warn(f"epsilon value for {zonotope.volume_vector} is too small!")

        with profiler["covers"]:
            # Test first with negative epsilon to filter non-tight cases
            if not domain:
                domain = scaled_polytope_covers(
                    Ac, bc, cov_radius, -epsilon,
                    bounds=bounds,
                    assume_centrally_symmetric=True,
                    expect_possible_divergence=True,
                    min_side=min_side,
                )

            # If a domain was not found with the negative epsilon, the covering radius
            # is lower bounded by cov_radius, we may have found a tight case
            if not domain:
                # Test with positive epsilon
                domain = scaled_polytope_covers(
                    Ac, bc, cov_radius, epsilon,
                    bounds=bounds,
                    assume_centrally_symmetric=True,
                    expect_possible_divergence=False,
                    min_side=min_side,
                )

                # If the polytope does not cover with the positive epsilon,
                # its covering radius is strictly larger than cov_radius
                if not domain:
                    failed.append(zonotope)
                    continue

                # Add to the list of tight cases, since it did not cover with a negative epsilon
                tight.append(zonotope)

        with profiler["cert"]:
            # Use epsilon of 0 for the certificate if the domain was found with negative epsilon (saves space)
            # eps_for_cert = epsilon if zonotope in tight else Fraction(0, 1)
            eps_for_cert = epsilon
            certificate = SLRCCertificate(
                zonotope.volume_vector, zonotope.generators, domain, cov_radius, eps_for_cert)
            if check_certificates and not certificate.is_valid():
                raise ValueError(f"certificate is invalid: {zonotope.volume_vector})")
            certificates.append(certificate)

    return certificates, failed, tight

# The following block is run when this script is executed directly
if __name__ == '__main__':
    # Load zonotopes from archive
    zonotopes = load_LR_zonotope_archive('data/slr_3_zonotopes_up_to_volume_195.npz')

    # Build certificates
    certificates, failed, tight = build_certificates(
        zonotopes,
        cov_radius=Fraction(3, 5),
        rough_tight_test=True,
        min_side=1e-7,
        check_certificates=False,
        profile=False,
    )

    if failed:
        print(f"failed: {[z.volume_vector for z in failed]}")
    if tight:
        print(f"tight: {[z.volume_vector for z in tight]}")

    if certificates:
        print(f"saving certificates...")
        save_npz_sLRC_certificates_archive(
            certificates,
            f'data/slrc_5_runners_certificates.npz',
        )
        print(f"saved {len(certificates)} certificates!")
