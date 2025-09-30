"""
Common Utilities

Shared classes and functions used across multiple modules.
Consolidates previously redundant definitions.
"""

import os
import csv
import numpy as np
import trimesh
from collections import defaultdict
import heapq
from typing import Dict, Any, List, Optional


class GlobalResourceManager:
    """Singleton class to manage expensive resources across all pair processing"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._dino_matcher = None
            self._base_configs = {}
            self._kdtrees = {}
            self._gpu_buffers = []
            self._initialized = True
    
    def get_base_config(self, config_path):
        if config_path not in self._base_configs:
            import yaml
            with open(config_path, 'r') as f:
                self._base_configs[config_path] = yaml.safe_load(f)
        return self._base_configs[config_path]
    
    def get_dino_matcher(self, **kwargs):
        if self._dino_matcher is None:
            from ..features.matching import DinoMatcher
            self._dino_matcher = DinoMatcher(**kwargs)
        return self._dino_matcher
    
    def get_kdtree(self, vertices, cache_key):
        if cache_key not in self._kdtrees:
            from scipy.spatial import KDTree
            self._kdtrees[cache_key] = KDTree(vertices)
        return self._kdtrees[cache_key]
    
    def get_gpu_buffer(self, required_size):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for buffer in self._gpu_buffers:
            if len(buffer) >= required_size:
                return buffer[:required_size]
        
        new_buffer = torch.zeros(required_size, 3, device=device)
        self._gpu_buffers.append(new_buffer)
        return new_buffer
    
    def clear_caches(self):
        self._base_configs.clear()
        self._kdtrees.clear()
        self._gpu_buffers.clear()


def find_model_files(models_dir, model_name):
    """
    Find model files using cache system.
    
    Args:
        models_dir: Path to the models directory (or cache directory)
        model_name: Name of the model (without extension)
        
    Returns:
        Dictionary with file paths (not loaded data)
    """
    from ..utils.cache import CacheManager
    cache_manager = CacheManager()
    
    model_files = {'obj': None, 'npy': None, 'dino_features': None}
    
    # Find model file to compute cache hash
    # First check if we have a direct path in the models_dir
    mesh_path_for_cache = None
    
    # Try different possible paths
    possible_paths = [
        os.path.join(models_dir, f"{model_name}.obj"),  # Direct path in provided directory
        f"./models/{model_name}_simplified.obj",        # Simplified in models folder
        f"./models/{model_name}.obj"                     # Original in models folder
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            mesh_path_for_cache = path
            break
    
    if mesh_path_for_cache is None:
        raise ValueError(f"Model file not found: {model_name}.obj (searched: {possible_paths})")
    
    # Get cached file paths using the appropriate mesh path
    normalized_mesh_path = cache_manager.get_normalization_cache_path(mesh_path_for_cache)
    keypoints_file = cache_manager.get_keypoints_cache_path(mesh_path_for_cache)
    dino_features_file = cache_manager.get_dino_cache_path(mesh_path_for_cache)
    
    # Set paths if files exist
    if os.path.exists(normalized_mesh_path):
        model_files['obj'] = normalized_mesh_path
    else:
        model_files['obj'] = original_mesh_path
        
    if os.path.exists(keypoints_file):
        model_files['npy'] = keypoints_file
        
    if os.path.exists(dino_features_file):
        model_files['dino_features'] = dino_features_file
    
    return model_files


def find_all_models(models_dir: str) -> List[Dict[str, str]]:
    """
    Finds all .obj files in the specified directory and returns their paths and names.
    
    Args:
        models_dir: The directory to search for .obj files.
        
    Returns:
        A list of dictionaries, each containing 'name' (stem of the file) and 'path' (full path).
    """
    from pathlib import Path
    models = []
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".obj"):
                model_path = Path(root) / file
                model_name = model_path.stem
                models.append({'name': model_name, 'path': str(model_path), 'class': 'unknown'})
    return models


def compute_geodesic_distances(mesh, source_vertices):
    """Compute geodesic distances from source vertices to all vertices using Dijkstra's algorithm"""
    # Build vertex adjacency graph
    vertex_adjacency = defaultdict(set)
    for face in mesh.faces:
        vertex_adjacency[face[0]].add(face[1])
        vertex_adjacency[face[0]].add(face[2])
        vertex_adjacency[face[1]].add(face[0])
        vertex_adjacency[face[1]].add(face[2])
        vertex_adjacency[face[2]].add(face[0])
        vertex_adjacency[face[2]].add(face[1])
    
    vertices = mesh.vertices
    n_vertices = len(vertices)
    
    # Initialize distances
    distances = np.full(n_vertices, np.inf)
    
    # Set source vertices distance to 0
    for src in source_vertices:
        distances[src] = 0.0
    
    # Priority queue: (distance, vertex_index)
    pq = [(0.0, src) for src in source_vertices]
    heapq.heapify(pq)
    
    visited = set()
    
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
            
        visited.add(current_vertex)
        
        # Explore neighbors
        for neighbor in vertex_adjacency[current_vertex]:
            if neighbor not in visited:
                edge_length = np.linalg.norm(vertices[current_vertex] - vertices[neighbor])
                new_dist = current_dist + edge_length
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return distances


def label_vertices_by_keypoints(mesh, keypoints):
    """Label each vertex based on its nearest keypoint using geodesic distance"""
    from scipy.spatial import KDTree
    from tqdm import tqdm
    
    vertices = mesh.vertices
    n_vertices = len(vertices)
    num_keypoints = len(keypoints)
    
    # Detect keypoint structure
    if num_keypoints == 17:
        keypoint_types = ["pose"] * 17
    elif num_keypoints == 18:
        keypoint_types = ["pose"] * 17 + ["tail"]
    elif num_keypoints > 18:
        num_spine_points = num_keypoints - 18
        keypoint_types = ["pose"] * 17 + ["tail"] + ["spine"] * num_spine_points
    else:
        keypoint_types = ["custom"] * num_keypoints
    
    # Create KDTree for reference
    kp_tree = KDTree(keypoints)
    
    # For geodesic-based labeling, we'll use geodesic distance from keypoint regions
    # Find the closest vertex to each keypoint in Euclidean space (as starting point)
    keypoint_vertices = []
    for i, keypoint in enumerate(keypoints):
        distances_to_kp = [np.linalg.norm(vertex - keypoint) for vertex in vertices]
        closest_vertex = np.argmin(distances_to_kp)
        keypoint_vertices.append(closest_vertex)
    
    # Compute geodesic distances from all keypoint vertices
    all_geodesic_distances = []
    
    with tqdm(total=len(keypoint_vertices), desc="Computing geodesic distances", unit="keypoint", leave=True) as pbar:
        for i, kp_vertex in enumerate(keypoint_vertices):
            pbar.set_postfix({"Keypoint": f"{i} ({keypoint_types[i]})"})
            geodesic_dist = compute_geodesic_distances(mesh, [kp_vertex])
            all_geodesic_distances.append(geodesic_dist)
            pbar.update(1)
    
    # Assign each vertex to the keypoint with minimum geodesic distance
    vertex_labels = np.zeros(n_vertices, dtype=int)
    for v_idx in range(n_vertices):
        min_dist = float('inf')
        best_label = 0
        
        for kp_idx, geodesic_dist in enumerate(all_geodesic_distances):
            if geodesic_dist[v_idx] < min_dist:
                min_dist = geodesic_dist[v_idx]
                best_label = kp_idx
        
        vertex_labels[v_idx] = best_label
    
    # MODIFICATION: Treat first and second keypoints as the same label
    # Convert keypoint index 1 to label 0 (same as keypoint index 0)
    vertex_labels[vertex_labels == 1] = 0
    
    # Shift all labels > 1 down by 1 to maintain consecutive labeling
    for label_idx in range(2, num_keypoints):
        vertex_labels[vertex_labels == label_idx] = label_idx - 1
    
    return vertex_labels, kp_tree


# Global instance
resource_manager = GlobalResourceManager()
