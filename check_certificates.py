from math import gcd
from typing import Sequence

from slrz.slrc_certificate import SLRCCertificate
from slrz.serialization.slrc_certificate import load_npz_sLRC_certificates_archive

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    from warnings import warn
    warn("tqdm not installed, no progress bar will be displayed.")
    tqdm = None


def sLRC_primitive_volume_vectors_up_to_volume(max_volume_inclusive: int):
    """
    Yields all primitive, lexicographically sorted, volume vectors up to the given volume,
    that is, all positive integer vectors (v₁, v₂, v₃, v₄) with:
    - v₁ < v₂ < v₃ < v₄
    - gcd(v₁, v₂, v₃, v₄) = 1
    - v₁ + v₂ + v₃ + v₄ ≤ max_volume_inclusive

    The implementation is inefficient for the sake of clarity.
    See [slrz.lrz.sLRC_primitive_volume_vectors_upto_volume] for a proper implementation.

    :param max_volume_inclusive: Maximum volume to yield (inclusive).
    """
    min_volume_inclusive = 1 + 2 + 3 + 4
    for volume in range(min_volume_inclusive, max_volume_inclusive + 1):
        for v1 in range(1, volume):
            for v2 in range(v1 + 1, volume):
                for v3 in range(v2 + 1, volume):
                    v4 = volume - v1 - v2 - v3
                    if v4 > v3 and gcd(v1, v2, v3, v4) == 1:
                        yield v1, v2, v3, v4


def check_certificate_archive_integrity(
    file: str,
    required_volume_vectors: Sequence[tuple[int, int, int, int]],
):
    """
    Loads a list of certificates from the provided `.npz` file, using the
    `load_npz_sLRC_certificates_archive(...)` function, and ensures the list
    contains a valid certificate for each volume vector in the sequence
    `required_volume_vectors`.

    :param file:
        Path to a file containing the certificates, as serialized by the
        `save_npz_sLRC_certificates_archive(...)` function.
    :param required_volume_vectors:
        Sequence (e.g., list) of volume vectors to require certificates for.
    :return:
        Returns nothing.
        The function will only terminate normally if all required certificates
        are found and valid.
    :raises AssertionError:
        If either a required volume vector lacks an associated certificate in
        the `file`, or its associated certificate is not valid.
    """
    # Load certificates from file
    certificates: list[SLRCCertificate] = load_npz_sLRC_certificates_archive(file)

    # Index certificates by their volume vector for faster lookup
    certificate_map = {
        c.volume_vector: c
        for c in certificates
    }

    # Lists of missing volume vectors and volume vectors with failed certificates
    missing_volume_vectors = []
    failed_volume_vectors = []

    # Initialize progress bar, if tqdm is installed
    if tqdm:
        progress_bar = tqdm(
            total=len(required_volume_vectors),
            desc='Checking sLRC certificates',
            postfix='(failed: 0) (missing: 0)')
    else:
        progress_bar = None

    for volume_vector in required_volume_vectors:
        # Check a certificate exists for this volume vector
        missing = volume_vector not in certificate_map
        if missing:
            missing_volume_vectors.append(volume_vector)

            # Update progress bar
            if progress_bar:
                progress_bar.set_postfix_str(f'(failed: {len(failed_volume_vectors)}) (missing: {len(missing_volume_vectors)})')
                progress_bar.update(1)

            # Skip to the next required volume vector
            continue

        # Get certificate for the volume vector
        certificate = certificate_map[volume_vector]
        assert certificate.volume_vector == volume_vector, "Lookup mismatched!"

        # Check validity of the certificate
        if not certificate.is_valid():
            failed_volume_vectors.append(volume_vector)

            if progress_bar:
                progress_bar.set_postfix_str(
                    f'(failed: {len(failed_volume_vectors)}) '
                    f'(missing: {len(missing_volume_vectors)})')

        if progress_bar:
            progress_bar.update(1)

    # Report volume vectors missing a certificate
    if missing_volume_vectors:
        print(f"Missing certificates for {len(missing_volume_vectors)} volume vectors:")
        for volume_vector in missing_volume_vectors:
            print(f"  {volume_vector}")

    # Report volume vectors with a failed certificate
    if failed_volume_vectors:
        print(f"Failed certificates for {len(failed_volume_vectors)} volume vectors:")
        for volume_vector in failed_volume_vectors:
            print(f"  {volume_vector}")

    # Raise an error and exit with non-zero exit code if any volume vector lacked a valid certificate
    if missing_volume_vectors and not failed_volume_vectors:
        raise AssertionError(
            f"Missing certificates for {len(missing_volume_vectors)} volume vectors.")
    elif failed_volume_vectors and not missing_volume_vectors:
        raise AssertionError(
            f"Failed certificates for {len(failed_volume_vectors)} volume vectors.")
    elif missing_volume_vectors and failed_volume_vectors:
        raise AssertionError(
            f"Missing certificates for {len(missing_volume_vectors)} volume vectors "
            f"and failed certificates for {len(failed_volume_vectors)} volume vectors.")


# The following block is run when this script is executed directly
if __name__ == '__main__':
    print(f"Listing volume vectors up to volume 195...")
    volume_vectors = list(sLRC_primitive_volume_vectors_up_to_volume(195))

    print(f"Checking the certificates for {len(volume_vectors)} volume vectors...")
    check_certificate_archive_integrity(
        'data/slrc_5_runners_certificates.npz',
        volume_vectors,
    )
