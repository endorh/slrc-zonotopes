from __future__ import annotations

import json
import re
from fractions import Fraction
from typing import Sequence, Any

import numpy as np

from slrz.serialization.npz_utils import (
    squeeze_int_dtype,
    unsqueeze_int_dtype,
    flatten_inhomogeneous_int_array,
    unflatten_inhomogeneous_int_array,
)
from slrz.slrc_certificate import (
    ArrangedDyadicNode,
    ArrangedDyadicVoxel,
    DyadicFundamentalDomain,
    SLRCCertificate,
)

try:
    from tqdm import tqdm
except ImportError:
    from warnings import warn
    warn("tqdm not installed, no progress bars will be displayed for file serialization operations.")
    # Declare tqdm as passthrough iterable decorator
    def tqdm(iterable, *_, **__):
        return iterable


__all__ = [
    'save_npz_sLRC_certificates_archive',
    'load_npz_sLRC_certificates_archive',
    'save_json_sLRC_certificates_archive',
    'load_json_sLRC_certificates_archive',
]


def flatten_int_vector_tree(tree: ArrangedDyadicNode | ArrangedDyadicVoxel) -> np.ndarray[tuple[int], int]:
    flat: list[int] = []
    def shift(elem: int):
        return elem + 1 if elem >= 0 else elem
    def _flatten_tree(node: ArrangedDyadicNode | ArrangedDyadicVoxel):
        nonlocal flat
        if isinstance(node, ArrangedDyadicVoxel):
            flat += [shift(coord) for coord in node.lattice_translation]
        elif isinstance(node, ArrangedDyadicNode):
            flat.append(0)
            for child in node.children:
                _flatten_tree(child)
    _flatten_tree(tree)
    return np.array(flat, dtype=int)

def unflatten_int_vector_tree(flat: np.ndarray[tuple[int], int]) -> ArrangedDyadicNode | ArrangedDyadicVoxel:
    it = iter(flat)
    def unshift(elem: int) -> int:
        return elem - 1 if elem > 0 else elem
    def _unflatten_leaf_node(first: int):
        return ArrangedDyadicVoxel(np.array((
            unshift(first), unshift(next(it)), unshift(next(it))
        ), dtype=int))
    def _unflatten_nonleaf_node():
        return ArrangedDyadicNode([
            _unflatten_node(next(it))
            for _ in range(8)])
    def _unflatten_node(first: int):
        return _unflatten_nonleaf_node() if first == 0 else _unflatten_leaf_node(first)

    try:
        node = _unflatten_node(next(it))
    except StopIteration:
        raise ValueError("Too few elements in flattened tree array!")
    try:
        next(it)
        raise ValueError("Too many elements in flattened tree array!")
    except StopIteration:
        pass

    return node


def save_npz_sLRC_certificates_archive(
    certificates: Sequence[SLRCCertificate],
    file: str,
    compress: bool = True,
):
    assert len(set(c.cov_radius for c in certificates)) == 1, "all certificates must have the same cov_radius"

    cov_radius = next(iter(certificates)).cov_radius
    cov_radius_array = squeeze_int_dtype(np.array(
        (cov_radius.numerator, cov_radius.denominator), dtype=int))
    volume_vectors_array = squeeze_int_dtype(np.array(tuple(
        c.volume_vector for c in certificates)))
    generators_array = squeeze_int_dtype(np.array(tuple(
        c.generators for c in certificates)))
    flat_domains = [
        flatten_int_vector_tree(c.fundamental_domain.root)
        for c in certificates]
    flattened_domains = flatten_inhomogeneous_int_array(flat_domains)
    epsilon_array = squeeze_int_dtype(np.array(tuple(
        (c.epsilon.numerator, c.epsilon.denominator)
        for c in certificates
    ), dtype=int))

    save_fun = np.savez_compressed if compress else np.savez
    save_fun(
        file,
        volume_vectors=volume_vectors_array,
        generators=generators_array,
        dyadic_fundamental_domains=flattened_domains,
        cov_radius=cov_radius_array,
        epsilons=epsilon_array,
    )

def load_npz_sLRC_certificates_archive(file: str) -> list[SLRCCertificate]:
    """
    Loads a collection of SLRCCertificate objects from
    a .npz file previously saved with save_archive(...).
    """

    with np.load(file, allow_pickle=True) as data:
        volume_vectors = data["volume_vectors"]
        generators = data["generators"]
        flattened_domains = data["dyadic_fundamental_domains"]
        cov_radius_num, cov_radius_den = data["cov_radius"]
        epsilon_array = data["epsilons"]

    num_certificates = len(volume_vectors)

    # Unflatten flattened inhomogeneous arrays
    flat_domains = unflatten_inhomogeneous_int_array(flattened_domains)

    # Construct the shared covering radius (Fraction)
    cov_radius = Fraction(int(cov_radius_num), int(cov_radius_den))

    # Check that the sizes match
    if len(flat_domains) != num_certificates or len(generators) != num_certificates:
        raise ValueError(f"Mismatched array sizes!")

    # Inflate the certificate instances
    certificates = []
    for i in tqdm(range(num_certificates), desc=f"Loading sLRC certificates from archive '{file}'..."):
        # Inflate the dyadic fundamental domain node tree
        tree = flat_domains[i]
        root = unflatten_int_vector_tree(tree)

        fundamental_domain = DyadicFundamentalDomain(root)

        # Create the epsilon Fraction for this certificate
        eps_num, eps_den = epsilon_array[i]
        epsilon = Fraction(int(eps_num), int(eps_den))

        # Create the certificate instance
        cert = SLRCCertificate(
            volume_vector=tuple(int(x) for x in volume_vectors[i]),
            generators=unsqueeze_int_dtype(generators[i]),
            fundamental_domain=fundamental_domain,
            cov_radius=cov_radius,
            epsilon=epsilon)
        certificates.append(cert)

    return certificates


def save_json_sLRC_certificates_archive(
    certificates: Sequence[SLRCCertificate],
    file: str,
    pretty: bool = False,
):
    def serialize_node(node: ArrangedDyadicNode | ArrangedDyadicVoxel):
        """
        Serializes a dyadic tree node as a list of coordinates/serialized children.
        """
        if isinstance(node, ArrangedDyadicVoxel):
            return [
                int(coord)
                for coord in node.lattice_translation
            ]
        elif isinstance(node, ArrangedDyadicNode):
            return [
                serialize_node(child)
                for child in node.children
            ]
        raise ValueError(f"Invalid node: {node}")
    # Build JSON-friendly list of dictionaries of lists/str
    serialized = [
        dict((
            ('volume_vector', certificate.volume_vector),
            ('generators', list(list(int(e) for e in g) for g in certificate.generators.T)),
            ('dyadic_fundamental_domain', serialize_node(certificate.fundamental_domain.root)),
            ('cov_radius', f'{certificate.cov_radius.numerator}/{certificate.cov_radius.denominator}'),
        ) + ((
            ('epsilon', f'{certificate.epsilon.numerator}/{certificate.epsilon.denominator}'),
        ) if certificate.epsilon != 0 else ()))
        for certificate in certificates
    ]
    # Serialize list as JSON
    with open(file, 'w') as f:
        json.dump(serialized, f, indent=4 if pretty else None, check_circular=False, allow_nan=False)


def load_json_sLRC_certificates_archive(file: str) -> list[SLRCCertificate]:
    def require_json_key(d: dict[str, Any], key: str, cls: type | None = None):
        if key not in d:
            raise ValueError(f"missing required key '{key}' in JSON dict")
        value = d[key]
        if cls is not None and not isinstance(value, cls):
            raise ValueError(f"expected the type of '{key}' to be {cls.__name__}, but found {type(value).__name__}")
        return value

    with open(file, 'r') as f:
        serialized = json.load(f)

    if not isinstance(serialized, list):
        raise ValueError("expected a JSON list at root level")

    certificates: list[SLRCCertificate] = []
    for c in serialized:
        if not isinstance(c, dict):
            raise ValueError("expected only JSON objects inside root level list")

        volume_vector = tuple(require_json_key(c, 'volume_vector', list))
        if len(volume_vector) != 4 or not all(isinstance(e, int) for e in volume_vector):
            raise ValueError("expected 'volume_vector' to be a list of 4 integers")
        volume_vector = tuple(volume_vector)

        generators = require_json_key(c, 'generators', list)
        if len(generators) != 4 or not all(
                isinstance(g, list) and len(g) == 3 and all(
                    isinstance(e, int) for e in g) for g in generators):
            raise ValueError("expected 'generators' to be a length 4 list of length 3 lists of integers")
        generators = np.array(generators, dtype=int).T

        cov_radius = require_json_key(c, 'cov_radius', str)
        if not re.match(r'^\d+/\d+$', cov_radius):
            raise ValueError(f"expected 'cov_radius' to be a string in the form 'numerator/denominator', was: \"{cov_radius}\"")
        cov_radius = Fraction(*map(int, cov_radius.split('/')))

        if 'epsilon' not in c:
            epsilon = Fraction(0, 1)
        else:
            epsilon = require_json_key(c, 'epsilon', str)
            if not re.match(r'^[+-]?\d+/\d+$', epsilon):
                raise ValueError(f"expected 'epsilon' to be a string in the form '[+-]?numerator/denominator', was: \"{epsilon}\"")
            epsilon = Fraction(*map(int, epsilon.split('/')))

        serialized_dyadic_fundamental_domain = require_json_key(c, 'dyadic_fundamental_domain', list)
        def deserialize_node(lst: list[Any]):
            if not isinstance(lst, list) or not lst:
                raise ValueError("expected a non-empty JSON list inside 'voxelized_fundamental_domain' list")
            first = lst[0]
            if isinstance(first, int):
                if len(lst) == 3 and all(isinstance(e, int) for e in lst):
                    return ArrangedDyadicVoxel(np.array(lst, dtype=int))
                raise ValueError("expected a JSON list of length 3 containing integers inside 'voxelized_fundamental_domain' list")
            elif isinstance(first, list):
                if len(lst) == 8 and all(isinstance(e, list) for e in lst):
                    return ArrangedDyadicNode([
                        deserialize_node(sub)
                        for sub in lst])
                raise ValueError("expected a JSON list of length 8 containing JSON lists inside 'voxelized_fundamental_domain' list")
            raise ValueError("expected a JSON list of length 3 containing integers or a JSON list of length 8 containing JSON lists inside 'voxelized_fundamental_domain' list")

        dyadic_fundamental_domain = DyadicFundamentalDomain(
            deserialize_node(serialized_dyadic_fundamental_domain))

        certificates.append(SLRCCertificate(
            volume_vector,
            generators,
            dyadic_fundamental_domain,
            cov_radius,
            epsilon,
        ))

    return certificates
