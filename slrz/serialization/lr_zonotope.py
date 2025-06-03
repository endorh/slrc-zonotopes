from __future__ import annotations

from typing import Sequence

import numpy as np

from slrz.lr_zonotope import LRZonotope
from slrz.serialization.npz_utils import squeeze_int_dtype, unsqueeze_int_dtype

try:
    from tqdm import tqdm
except ImportError:
    from warnings import warn
    warn("tqdm not installed, no progress bars will be displayed for file serialization operations.")
    # Declare tqdm as passthrough iterable decorator
    def tqdm(iterable, *_, **__):
        return iterable


__all__ = [
    'save_LR_zonotope_archive',
    'load_LR_zonotope_archive',
]


def save_LR_zonotope_archive(zonotopes: Sequence[LRZonotope], file: str, *, compress: bool = True):
    """
    Save a list of LRZs to an `npz` archive which can be loaded with `load_LR_zonotope_archive`.

    LRZs generators are stored in a single (len(zonotopes), 3, 4) array, as this is
    orders of magnitude faster than storing each generator array separately.

    To save space, the arrays are stored with the minimum possible dtype possible.

    Furthermore, unless `compress` is set to `False`, `np.savez_compressed` is used to compress
    the file using ZIP_DEFLATE, which results in a ~50% file size reduction at the expense
    of minimal save and load overhead.

    Unfortunately, both `np.savez` and `np.savez_compressed` are not deterministic across platforms.
    """
    for Z in zonotopes:
        if not np.issubdtype(Z.generators.dtype, np.integer):
            raise ValueError(f"zonotope_enumeration.LRZ.save_archive > error: non-integer generators: {Z.generators.dtype}\n  {Z.generators}")
    generators_array = np.array(tuple(Z.generators for Z in zonotopes))
    generators_array = squeeze_int_dtype(generators_array)
    save_fun = np.savez_compressed if compress else np.savez
    save_fun(file, generators=generators_array)

def load_LR_zonotope_archive(file: str) -> tuple[LRZonotope, ...]:
    """
    Load a list of LRZs from an archive created with `save_LR_zonotope_archive`.
    """
    built: list[LRZonotope] = []
    archive = np.load(file)
    generators = archive['generators']
    generators.flags.writeable = False
    for Z_generators in tqdm(generators, desc=f"Loading LR zonotopes from archive '{file}'"):
        built.append(LRZonotope(unsqueeze_int_dtype(Z_generators), _skip_checks=True))
    return tuple(built)