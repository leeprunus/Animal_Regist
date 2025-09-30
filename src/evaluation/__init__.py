"""
Evaluation Module

Core MGE (Mean Geodesic Error) computation functions for mesh correspondence evaluation.
These functions are used by alignment.py to compute final correspondence quality.
"""

from .mge import (
    compute_mge_direct_correspondence_with_hungarian,
    save_correspondences,
    find_bijective_correspondences,
    compute_all_pairs_geodesic_distances_cached,
    build_adjacency_matrix_with_caching,
    convert_numpy_types
)

__all__ = [
    'compute_mge_direct_correspondence_with_hungarian',
    'save_correspondences', 
    'find_bijective_correspondences',
    'compute_all_pairs_geodesic_distances_cached',
    'build_adjacency_matrix_with_caching',
    'convert_numpy_types'
]
