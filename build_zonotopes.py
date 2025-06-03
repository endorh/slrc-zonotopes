from __future__ import annotations

from typing import Literal

from slrz.lr_zonotope import sLRC_primitive_volume_vectors, LRZonotope, lr_zonotope_from_volume_vector
from slrz.serialization.lr_zonotope import save_LR_zonotope_archive

from slrz.util import Profiler

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    from warnings import warn
    warn("tqdm not installed, no progress bar will be displayed.")
    # Declare tqdm as a passthrough iterator decorator
    def tqdm(iterator, *_, **__):
        return iterator


def build_zonotopes(
    volume_vectors: list[tuple[int, ...]],
    profile: bool = False,
) -> list[LRZonotope]:
    """
    Builds zonotopes for a list of requested volume vectors.
    :param volume_vectors:
        List of volume vectors to build zonotopes for.
    :param profile:
        Whether to profile the build process.
        If `tqdm` is installed, will display running statistics of the
        time taken by different sections of the code.
    :return:
        A list of LRZonotope objects, corresponding to the `volume_vectors`.
    """

    # Wrap with a progress bar if tqdm is installed (otherwise do nothing)
    volume_vectors = tqdm(volume_vectors, desc="Building zonotopes",)

    # Create profiler if requested
    profiler = Profiler(volume_vectors) if profile else Profiler.noop

    return [
        lr_zonotope_from_volume_vector(vv, profiler=profiler)
        for vv in volume_vectors
    ]

def build_zonotopes_archive(
    file: str,
    n=4, min_volume_inclusive=0, max_volume_inclusive=195,
    *,
    order: Literal['grlex', 'revgrlex', 'grevlex', 'revgrevlex', 'lex'] = 'grlex',
    compress: bool = True, profile: bool = False,
):
    """
    Builds zonotopes for all `n`-dimensional volume vectors with volume between
    `min_volume_inclusive` and `max_volume_inclusive` (inclusive), and saves them
    into the requested `file`, using the format provided by
    `save_LR_zonotope_archive(...)` and `load_LR_zonotope_archive(...)`.

    :param file:
        File to save the zonotopes to.
    :param n:
        Dimension of the volume vectors.
        One more than the dimension of the zonotopes.
    :param min_volume_inclusive:
        Minimum volume to build zonotopes for.
    :param max_volume_inclusive:
        Maximum volume to build zonotopes for.
    :param order:
        Order of enumeration of the volume vectors.
    :param compress:
        Whether to compress the generated file with ZIP_DEFLATE.
        See `numpy.savez_compressed(...)` for details.
    :param profile:
        Whether to profile the build process.
        If `tqdm` is installed, will display running statistics of the
        time taken by different sections of the code.
    :return:
    """

    # Enumerate volume vectors
    volume_vectors = sLRC_primitive_volume_vectors(
        n=n, max_volume_inclusive=max_volume_inclusive,
        min_volume_inclusive=min_volume_inclusive, order=order
    )

    # Wrap with a progress bar if tqdm is installed, otherwise do nothing
    volume_vectors = tqdm(
        volume_vectors,
        desc="Enumerating volume vectors",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    # Convert generator to list (forcing the eager enumeration of all volume vectors)
    volume_vectors = list(volume_vectors)

    # Build zonotopes for the volume vectors
    zonotopes = build_zonotopes(volume_vectors, profile=profile)

    # Save the built zonotopes in a `npz` archive
    save_LR_zonotope_archive(zonotopes, file, compress=compress)


# The following block is run when this script is executed directly
if __name__ == '__main__':
    build_zonotopes_archive(
        'data/slr_3_zonotopes_up_to_volume_195.npz',
        n=4,
        min_volume_inclusive=0,
        max_volume_inclusive=195,
        profile=False,
    )
