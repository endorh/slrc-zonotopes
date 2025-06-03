from __future__ import annotations

from collections import Counter

from slrz.serialization.slrc_certificate import load_npz_sLRC_certificates_archive

if __name__ == '__main__':
    certificates = load_npz_sLRC_certificates_archive('data/slrc_5_runners_certificates.npz')
    depths = Counter(
        c.fundamental_domain.depth()
        for c in certificates
    )

    print(f"Amount of certificates per depth:")
    for i in sorted(depths.keys()):
        v = depths[i]
        print(f"  {i}: {v}")
