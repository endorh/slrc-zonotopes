# Covering Radii of $3$-zonotopes and the shifted Lonely Runner Conjecture for $5$ runners

This repository contains Python source code that can be used to reproduce
the results from the paper
*Covering Radii of $3$-zonotopes and the shifted Lonely Runner Conjecture*
([preprint on arXiv][preprint]).

It also contains the $2\,133\,561$ certificates from the computational
proof claimed in the paper, and a script to validate them under exact arithmetic.

- [*Strong* Lonely Runner zonotopes](#strong-lonely-runner-zonotopes)
- [Repository structure](#repository-structure)
- [Dependencies](#dependencies)
- [Data format](#data-format)
- [Validating the certificates](#validating-the-certificates)

## *Strong* Lonely Runner zonotopes
A $d$-zonotope is a polytope generated as the Minkowski sum of segments in $\mathbb{R}^d$

A Lonely Runner (LR) $d$-zonotope is an integer $d$-zonotope, that is,
with integral vertices/generators, that has $d+1$ generators.

Every LR zonotope has an associated volume vector.
The volume vector of a LR $(n-1)$-zonotope is a vector whose entries correspond
to the volumes of the $n$ parallelepipeds obtained by dropping each of the generators.
In other words, the volume vector of a LR $(n-1)$-zonotope $`Z = \sum U`$, with
$`U = \{[0, \mathbf{u}_1], \ldots [0, \mathbf{u}_n]\}`$ and
$`\mathbf{u}_i \in \mathbb{Z}^{n-1} \forall i \in [n]`$, is the vector of minors of
the $`(n-1) \times n`$ matrix $`\mathbf{U} = (\mathbf{u}_1 | \cdots | \mathbf{u}_n)`$.

If we restrict ourselves to volume vectors with coprime entries, that is,
such that $`gcd(v_1, \ldots, v_n) = 1`$, the volume vector characterizes LR zonotopes
up to unimodular equivalence, i.e., affine transformations that preserve the integer
lattice.

See the [`lr_zonotope_from_volume_vector`](/slrz/lr_zonotope.py#L106-L177)
function from the [`slrz.lr_zonotope`](/slrz/lr_zonotope.py) module
for an algorithm that constructs a sLR zonotope with *small* generators for
a given volume vector with distinct and coprime entries.

A *strong* Lonely Runner (sLR) zonotope is a LR zonotope whose volume vector has
pairwise distinct entries.

As the name may suggest, the *volume vector* of a Lonely Runner zonotope is
tightly related with the *velocity vector* of an instance of the
Lonely Runner Conjecture.

In particular, the *shifted* Lonely Runner conjecture for $n+1$ runners can be
reformulated as the statement that all sLR $(n-1)$-zonotopes have covering radius
at most $`\frac{n-1}{n+1}`$.
That is, if, when scaled by a factor of $`\frac{n-1}{n+1}`$, the union of
all integer translations of the zonotope cover $\mathbb{R}^{n-1}$, i.e.,
if $Z$ is a sLR $(n-1)$-zonotope,
$`\mathbb{R}^{n-1} \subseteq \frac{n-1}{n+1}Z + \mathbb{Z}^{n-1}`$.
See [[M. Henze, R. Malikiosis, 2017][M. Henze, R. Malikiosis, 2017]] for the details
and a similar reformulation of the *non-shifted* Lonely Runner Conjecture in terms of
zonotopes intersecting the integer lattice.

A result from [[R. Malikiosis, F. Santos, M. Schymura, 2024][R. Malikiosis, F. Santos, M. Schymura, 2024]]
shows that, if a certain condition named the *Lonely Vector Property* holds for
all natural numbers $\leq n$, then all sLR $(n-1)$-zonotopes with volumes at least
$`\binom{n+1}{2}^{n-1}`$ satisfy the conjecture for $n+1$ runners.

The Lonely Vector Property for $n$ vectors states that for any set of
non-zero vectors
$`\mathbf{P} = \{\mathbf{p}_1, \ldots, \mathbf{p}_n\} \subseteq \mathbb{R}^2`$, if we define
$`S_\mathbf{P} = \mathbf{P} \cup \{\mathbf{p}_i \pm \mathbf{p}_j : 1 \leq i < j \leq n\}`$,
then $`S_\mathbf{P}`$ contains a vector that is not parallel to any other vector of
$`S_\mathbf{P}`$.

This property has been proven in [[R. Malikiosis, F. Santos, M. Schymura, 2024][R. Malikiosis, F. Santos, M. Schymura, 2024]]
for up to $n = 4$ vectors, which means that checking the covering radius of
all sLR $3$-zonotopes with volume smaller than $1000$ is enough to prove the
*shifted* Lonely Runner Conjecture for five runners.
However, in the same article, specifically for the case of five runners,
the authors have improved the bound to $196$, which reduces the number of
sLR zonotopes to check down to $2\,133\,561$.

See the `sLRC_primitive_volume_vectors` function from the
[`slrz.lr_zonotope`](/slrz/lr_zonotope.py) module for an example of how to
enumerate the volume vectors of these zonotopes.

This repository contains an algorithm which can decide whether a bound on
the covering radius of a polytope holds, or, in other words, whether the
polytope scaled by a given factor, covers $`\mathbb{R}^d`$ by integer translations.

The algorithm finds a dyadic fundamental domain of the integer lattice, $`\mathbb{Z}^d`$,
that fits inside the scaled polytope, with a small margin in the case of the polytopes
whose covering radius is exactly the bound checked.
This margin relies on a bound for the denominator of the covering radius of a
lattice polytope, which allows us to certify the bound even in these tight cases.
See the details in the [paper][preprint].

While the algorithm we use to construct these zonotope domains uses numerical
methods, and is subject to potential instabilities, the resulting fundamental
domain can be easily checked later for correctness under exact arithmetic.

Hence, we have constructed a dyadic fundamental domains for each of the
$2\,133\,561$ sLR zonotopes that needed to be checked.
These fundamental domains, together with the representative zonotopes
they're each contained in, are the certificates of our proof.

See the [section below](#validating-the-certificates) for instructions on how
to interpret and validate these certificates.

## Repository structure
The structure of this repository is as follows:
- [`data`](/data) Computation results.
- [`slrz`](/slrz) Python module, main source code.
  - [`lll`](/slrz/lll) Implementations of the LLL algorithm.
  - [`rational`](/slrz/rational) Utilities for exact arithmetic.
    - [`linalg`](/slrz/rational/linalg.py) Rational linear algebra.
    - [`milp`](/slrz/rational/milp.py) Utility wrapper for SciPy's HiGHS wrapper,
      a MILP solver.
  - [`serialization`](/slrz/serialization) Module containing all serialization logic
    involved in the creation and reading of archives in the [`data`](/data) directory.
  - [`util`](/slrz/util) Miscellaneous
    - [`profiler.py`](/slrz/util/profiler.py) A simple Python profiler.
    - [`optional_numba.py`](/slrz/util/optional_numba.py) A thin wrapper to use [`Numba`][Numba]
      only when available.
  - [`lr_zonotope.py`](/slrz/lr_zonotope.py) Utilities to construct sLR zonotopes.
    Contains the `LRZonotope` class, and several utilities to work with LR zonotopes,
    such as a way to derive linear inequalities for a centrally symmetric translation
    of the zonotopes.
  - [`slrc_certificate.py`](/slrz/slrc_certificate.py) Contains the `SLRCCertificate` class,
    which defines the structure of a certificate for the sLRC, and methods to validate
    them under exact arithmetic.
  - [`decide_polytope_cov_radius.py`](/slrz/slrc_certificate.py) Decides the covering
    radius of a polytope, by arranging a dyadic fundamental domain inside
    the scaled polytope.
  - [`gcd.py`](/slrz/gcd.py) Efficient implementation of the Extended Euclidean Algorithm.
- [`test`](/test) Python unit tests for several features of `slrz`.
- [`build_zonotopes.py`](/build_zonotopes.py) Creates an archive with small generators 
  for all sLR 3-zonotopes up to volume 195, as required for proving the sLRC.
- [`build_certificates.py`](/build_certificates.py) Creates an archive with certificates
  that prove the sLRC holds for five runners.
- [`check_certificates.py`](/check_certificates.py) Script with minimal dependencies
  that loads the certificates archive, and validates it contains certificates that
  remain valid under exact arithmetic for all volume vectors required to prove the
  sLRC for five runners.
- [`certificate_statistics.py`](/certificate_statistics.py) A simple script that computes
  the frequencies of all certificate depths.
- [`requirements.txt`](/requirements.txt) Python dependencies, including some optional
  dependencies.
- [`minimal-requirements.txt`](/minimal-requirements.txt) Minimal set of required
  Python dependencies.

### Unit tests
This repository includes a few unit tests for a few of the submodules under `slrz`.
You may run this tests from the repository root by running `python` with the following options:

```shell
python -m unittest discover -s test -t test
```

The tests only check simple invariants for a few cases of simple algorithms, and exist mostly
for the convenience of discovering errors fast while editing the algorithms.
They have nothing to do with the validation of the certificates for the computational proof.

## Dependencies
We have tested this repository on Python 3.10, although we recommend using at least
Python 3.12 or later.

This repository relies on a few dependencies:
- [`numpy`][NumPy] for acceleration of linear algebra.
- [`scipy`][SciPy] for solving Mixed Integer Linear Programs (MILPs), as it provides a wrapper for the
  [HiGHS MILP][HiGHS] solver.

Additionally, the repository makes use of the following optional —but recommended— dependencies:
- [`numba`][Numba]: A JIT Python compiler for acceleration of several algorithms.
- [`tqdm`][tqdm]: A simple utility to provide progress bars for loops in terminal interfaces.

All these dependencies are properly listed in the [`requirements.txt`](/requirements.txt) file.
Hence, you may install them all automatically by running `pip` from the repository root with the
following options:
```shell
pip install -r requirements.txt
```

If you instead wish to only install the required dependencies, you may also do that by running
`pip` from the repository root with the following options:
```shell
pip install -r minimal-requirements.txt
```

It is likely —but not certain— that using older/newer versions of the listed dependencies
will cause no issues, as long as they're installed.

If you only need to validate the certificates from the `data` directory, you only need to
install [`NumPy`][NumPy].

## Data format
There are two binary archives in the [`data`](/data) directory:

### `slr_3_zonotopes_up_to_volume_195.npz`
This file contains small generators for all sLR zonotopes with volumes at most 195.
It can be regenerated by running the [`build_zonotopes.py`](/build_zonotopes.py) script
from the repository root.
On a standard PC, this takes around ~10 min.

The archive simply contains an integer array with shape $(2\,133\,561, 3, 4)$,
where each row is a $3\times 4$ matrix describing the $4$ generators of one
of the zonotopes.

Remarkably, the generators are so small that the array has been encoded in `i1` format,
that is, each integer coordinate only takes one byte of space, for a total of
$25602732B \approx 25 MB$ of storage.
The array has been saved with
[`np.savez_compressed`](https://numpy.org/doc/2.2/reference/generated/numpy.savez_compressed.html),
which compresses it further using ZIP_DEFLATED, for a final size of $\approx 12 MB$,
which allows us to freely include this file in a GitHub repository.

A slightly undesirable aspect of the `npz` format (a ZIP archive of `npy` arrays) is that it includes
timestamps, so it is not a stable format for version control/automated testing.
For more technical information on the `npy` and `npz` formats, you may refer to the
[NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).

To load this archive as a list of `LRZonotope` objects, you may use the
`load_LR_zonotope_archive` function from the
[`slrz.serialization.lr_zonotope`](/slrz/serialization/lr_zonotope.py) module:

```python
from slrz.serialization.lr_zonotope import load_LR_zonotope_archive

zonotopes = load_LR_zonotope_archive('data/slr_3_zonotopes_up_to_volume_195.npz')
```

### `slrc_5_runners_certificates.npz`
This file contains a list of serialized `SLRCCertificate` instances.
It can be regenerated by runing the [`build_certificates.py`](/build_certificates.py)
script from the repository root.
This script expects the `slr_3_zonotopes_up_to_volume_195.npz` file to exist.

On a standard PC, this takes around ~15 min.
Note that, since the zonotopes are sorted by volume, the first cases are
the hardest, so the progress bar is pessimistic in its time estimation.
If you have numba installed, the first run of the script will also encounter
an initial overhead due to JIT compilation, in exchange for improved performance.

In order to compress these certificates to a reasonable size,
the format is slightly more complex.

You may use the `load_npz_sLRC_certificates_archive` function from the
[`slrz.serialization.slrc_certificate.py`](/slrz/serialization/slrc_certificate.py)
module to load this archive as a list of `SLRCCertificate` instances:

```python
from slrz.serialization.slrc_certificate import load_npz_sLRC_certificates_archive

certificates = load_npz_sLRC_certificates_archive('data/slrc_5_runners_certificates.npz')
```

The archive is an `npz` (ZIP) archive that contains five integer arrays in `npy` format:
- `cov_radius.npy`: An array of shape $(2)$, that is, a vector with two entries,
  in order, the numerator and the denominator of the covering radius certified
  by all the certificates in the archive, that is, $\frac{3}{5}$.
- `volume_vectors.npy`: An array with shape $(2\,133\,561, 4)$ containing the
  list of volume vectors certified by each certificate.
  While this list could be computed from the generators, it is included
  for ease of lookup.
- `generators.npy`: An array with shape $(2\,133\,561, 3, 4)$, like that of
  the `slr_3_zonotopes_up_to_volume_195.npz` file, which describes small
  generators for a representative sLR zonotope for each volume vector.
- `epsilons.npy`: An array with shape $(2\,133\,561, 2)$, where each row
  contains the numerator and denominator of the epsilon associated with
  a certificate.
  Except for three of them, they are all negative, proving that
  only three instances are tight.
- `dyadic_fundamental_domains.npy`: A flat array that encodes the dyadic fundamental
  domains of each certificate.
  The domains for different certificates are separated by a $0$ byte,
  and for each domain, all non-negative bytes have been incremented by
  one for disambiguation.
  Each domain is described as a regular $8$-tree of vectors.
  A byte with value $0$ (after decrement) indicates that a node is not a leaf,
  and its $8$ children follow.
  Each leaf is defined by $3$ bytes, one for each coordinate, where
  its non-negative coordinates have been incremented by one for disambiguation.

This format, albeit slightly convoluted, allows us to encode all certificates under
$24 MB$, and include them in this repository.

## Validating the certificates
You may load the certificates in Python as a list of `SLRCCertificate` objects
using the `load_npz_sLRC_certificates_archive` function from the
[`slrz.serialization.slrc_certificate.py`](/slrz/serialization/slrc_certificate.py)
module.

Suggested steps for certifying these certificates are described in the
[`check_certificates.py`](/check_certificates.py), which we have used to
validate the certificates ourselves.

Running the [`check_certificates.py`](/check_certificates.py) script on the repository root will
check that the [`data/slrc_5_runners_certificates.npz`](/data/slrc_5_runners_certificates.npz) file
contains certificates that remain valid under exact arithmetic for all volume vectors required
to prove the *shifted* Lonely Runner Conjecture for five runners.

```shell
python check_certificates.py
```

On a standard PC, this takes around ~30 min.

This script is mostly self-contained, depending only on the following files/dependencies:
- [`slrz/slrc_certificate.py`](/slrz/slrc_certificate.py), which defines the `SLRCCertificate` class
  that describes a certificate, and contains the code to validate them.
- [`slrz/serialization/slrc_certificate.py`](/slrz/serialization/slrc_certificate.py), which contains exclusively the
  code necessary to load `SLRCCertificate` instances, and depends on
  [`slrz/serialization/npz_utils.py`](/slrz/serialization/npz_utils.py)
- [`NumPy`][NumPy], for both, loading the certificates in `npz` format, and basic integer linear algebra.


[//]: # (References)
[preprint]: https://arxiv.org/abs/2506.13379
[M. Henze, R. Malikiosis, 2017]: https://doi.org/10.1007/s00010-016-0458-3
[R. Malikiosis, F. Santos, M. Schymura, 2024]: https://doi.org/10.48550/arXiv.2411.06903
[NumPy]: https://numpy.org
[SciPy]: https://scipy.org
[HiGHS]: https://highs.dev
[Numba]: https://numba.pydata.org
[tqdm]: https://github.com/tqdm/tqdm
