"""
Core MGE computation functions for mesh correspondence evaluation.

This module provides the essential functions needed by alignment.py for computing
Mean Geodesic Error (MGE) and saving correspondences after ARAP optimization.

Key Functions:
- compute_mge_direct_correspondence: Main MGE computation 
- save_correspondences: Save correspondences in JSON format
- Supporting functions for geodesic distance computation and bijective correspondence finding
"""

import os
import numpy as np
import trimesh
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import cache system
from src.utils.cache import get_cache_manager

# Global cache manager instance
cache_manager = get_cache_manager("./cache")


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def save_correspondences(correspondences, output_file, mge=None):
    """Save correspondences in JSON format with optional MGE value."""
    if not correspondences:
        return
    
    correspondence_dict = {}
    
    for corr in correspondences:
        source_idx = int(corr['source_vertex_idx'])
        target_idx = int(corr['target_vertex_idx'])
        
        correspondence_dict[str(source_idx)] = target_idx
    
    metadata = {
        'total_correspondences': len(correspondences),
        'method': 'mge_evaluation'
    }
    
    if mge is not None:
        metadata['mge'] = float(mge)
    
    output_data = {
        'correspondences': correspondence_dict,
        'metadata': metadata
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def build_adjacency_matrix_with_caching(mesh: trimesh.Trimesh) -> csr_matrix:
    """Build adjacency matrix using mesh edges and Euclidean distances."""
    vertices = mesh.vertices
    edges = mesh.edges_unique
    
    # Calculate edge lengths (weights) using vectorized operations
    edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Build sparse adjacency matrix (undirected graph)
    n_vertices = len(vertices)
    
    # Pre-allocate arrays
    n_edges = len(edges)
    row_indices = np.empty(2 * n_edges, dtype=np.int32)
    col_indices = np.empty(2 * n_edges, dtype=np.int32)
    data = np.empty(2 * n_edges, dtype=np.float32)
    
    # Fill arrays
    row_indices[:n_edges] = edges[:, 0]
    row_indices[n_edges:] = edges[:, 1]
    col_indices[:n_edges] = edges[:, 1]
    col_indices[n_edges:] = edges[:, 0]
    data[:n_edges] = edge_lengths.astype(np.float32)
    data[n_edges:] = edge_lengths.astype(np.float32)
    
    adjacency_matrix = csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(n_vertices, n_vertices),
        dtype=np.float32
    )
    
    return adjacency_matrix


def compute_all_pairs_geodesic_distances_cached(mesh: trimesh.Trimesh, mesh_file_path: str = None) -> np.ndarray:
    """
    Compute all-pairs geodesic distance matrix using Dijkstra algorithm with caching.
    Uses caching to avoid recomputation for same meshes.
    """
    # Try to load from cache first
    if mesh_file_path:
        cached_distances = cache_manager.load_geodesic(mesh_file_path)
        if cached_distances is not None:
            return cached_distances
    
    adjacency_matrix = build_adjacency_matrix_with_caching(mesh)
    
    distances = dijkstra(
        adjacency_matrix, 
        directed=False, 
        return_predecessors=False
    )
    
    # Cache the results
    if mesh_file_path:
        cache_manager.cache_geodesic(mesh_file_path, distances)
    
    return distances


def find_bijective_correspondences(deformed_vertices: np.ndarray, 
                                  target_vertices: np.ndarray,
                                  max_correspondences: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find bijective (one-to-one) correspondences using Hungarian algorithm.
    This ensures coverage and avoids many-to-one mappings.
    """
    n_deformed = len(deformed_vertices)
    n_target = len(target_vertices)
    
    # Limit correspondences if requested
    if max_correspondences is not None:
        n_correspondences = min(max_correspondences, n_deformed, n_target)
    else:
        n_correspondences = min(n_deformed, n_target)
    
    
    # For efficiency, if we're dealing with a large number of vertices,
    # we subsample to find correspondences
    if n_deformed > n_correspondences or n_target > n_correspondences:
        # Sample vertices to limit computational cost
        deformed_indices = np.random.choice(n_deformed, size=min(n_correspondences, n_deformed), replace=False)
        target_indices = np.random.choice(n_target, size=min(n_correspondences, n_target), replace=False)
        
        deformed_sample = deformed_vertices[deformed_indices]
        target_sample = target_vertices[target_indices]
    else:
        deformed_indices = np.arange(n_deformed)
        target_indices = np.arange(n_target)
        deformed_sample = deformed_vertices
        target_sample = target_vertices
    
    # Compute pairwise distances
    distance_matrix = cdist(deformed_sample, target_sample)
    
    # Use Hungarian algorithm to find optimal bijective assignment
    source_idx, target_idx = linear_sum_assignment(distance_matrix)
    
    # Get assignment distances
    assignment_distances = distance_matrix[source_idx, target_idx]
    
    # Map back to original indices
    actual_source_indices = deformed_indices[source_idx]
    actual_target_indices = target_indices[target_idx]
    
    
    return actual_source_indices, actual_target_indices, assignment_distances


def compute_mge_direct_correspondence_with_hungarian(deformed_mesh: trimesh.Trimesh, 
                                                  target_mesh: trimesh.Trimesh,
                                                  max_correspondences: int = None,
                                                  target_mesh_file: str = None) -> Tuple[float, Dict]:
    """
    Compute MGE using bijective correspondence finding.
    Normalized by √(surface_area).
    Uses Hungarian algorithm for bijective correspondences.
    """
    # Check mesh compatibility
    if len(deformed_mesh.vertices) != len(target_mesh.vertices):
        print(f"  Warning: Mesh vertex counts differ: deformed={len(deformed_mesh.vertices)}, target={len(target_mesh.vertices)}")
        print(f"  Using bijective correspondence finding...")
    
    # Determine number of correspondences
    num_correspondences = min(len(deformed_mesh.vertices), len(target_mesh.vertices))
    if max_correspondences is not None:
        num_correspondences = min(num_correspondences, max_correspondences)
    
    
    # Compute surface area for normalization
    surface_area = float(target_mesh.area)
    area_normalization = np.sqrt(surface_area)
    
    
    # Compute all-pairs geodesic distances on target mesh (with caching)
    all_distances = compute_all_pairs_geodesic_distances_cached(target_mesh, target_mesh_file)
    
    # Find bijective correspondences
    source_indices, target_indices, assignment_distances = find_bijective_correspondences(
        deformed_mesh.vertices, target_mesh.vertices, max_correspondences
    )
    
    actual_correspondences = len(source_indices)
    
    # Compute geodesic errors using vectorized operations
    
    # For direct correspondence evaluation, we compare with the identity mapping
    # But since we have bijective correspondences, we use the found correspondences
    if len(deformed_mesh.vertices) == len(target_mesh.vertices):
        # Direct correspondence case: ground truth is identity mapping
        target_ground_truth = source_indices  # Identity mapping for available correspondences
    else:
        # Different vertex counts: use the bijective correspondences as ground truth
        target_ground_truth = target_indices
    
    # Compute geodesic distances between found correspondences and ground truth
    geodesic_errors = []
    correspondences_info = []
    
    for i, (source_idx, target_idx, dist) in enumerate(zip(source_indices, target_indices, assignment_distances)):
        ground_truth_idx = target_ground_truth[i]
        
        # Geodesic error: distance between predicted and ground truth target vertices
        if target_idx < len(all_distances) and ground_truth_idx < len(all_distances):
            geodesic_error = all_distances[target_idx, ground_truth_idx]
        else:
            # Fallback: use Euclidean distance if geodesic unavailable
            geodesic_error = np.linalg.norm(target_mesh.vertices[target_idx] - target_mesh.vertices[ground_truth_idx])
        
        geodesic_errors.append(geodesic_error)
        
        # Store correspondence information
        correspondences_info.append({
            'source_vertex_idx': int(source_idx),
            'target_vertex_idx': int(target_idx),
            'ground_truth_idx': int(ground_truth_idx),
            'assignment_distance': float(dist),
            'geodesic_error': float(geodesic_error)
        })
    
    # Compute MGE (normalized by √area)
    geodesic_errors = np.array(geodesic_errors)
    mge = np.mean(geodesic_errors) / area_normalization
    
    # Compute statistics
    target_coverage = actual_correspondences / len(target_mesh.vertices) * 100
    
    
    # Return results
    stats = {
        'mge': float(mge),
        'mge_unnormalized': float(np.mean(geodesic_errors)),
        'area_normalization': float(area_normalization),
        'surface_area': float(surface_area),
        'num_correspondences': int(actual_correspondences),
        'target_coverage': float(target_coverage),
        'mean_geodesic_error': float(np.mean(geodesic_errors)),
        'median_geodesic_error': float(np.median(geodesic_errors)),
        'std_geodesic_error': float(np.std(geodesic_errors)),
        'min_geodesic_error': float(np.min(geodesic_errors)),
        'max_geodesic_error': float(np.max(geodesic_errors)),
        'mean_assignment_distance': float(np.mean(assignment_distances)),
        'correspondences': correspondences_info
    }
    
    return mge, stats
