#!/usr/bin/env python3
"""
ARAP Registration for alignment pipeline.

This script processes model pairs from CSV file, computes DINO correspondences,
and performs ARAP-based alignment between source and target meshes.

Reuses heavy objects across pairs for computational performance.
"""

import numpy as np
import trimesh
import os
import time
import json
import csv
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.linalg import svd, det
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Set deterministic seeds for reproducible results
np.random.seed(42)
import math
import torch
import yaml
import copy  # Add missing import
from typing import Tuple

# Import DINO functionality - using relative import from within src
from ..features.matching import DinoMatcher

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# CORRESPONDENCE UTILITIES
# ========================================

def find_bijective_correspondences(deformed_vertices: np.ndarray, 
                                  target_vertices: np.ndarray,
                                  max_correspondences: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find bijective (one-to-one) correspondences using Hungarian algorithm.
    This ensures coverage and avoids many-to-one mappings.
    Adapted from evaluation.py for use in alignment.py.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    
    n_deformed = len(deformed_vertices)
    n_target = len(target_vertices)
    
    # Limit correspondences if requested
    if max_correspondences is not None:
        n_correspondences = min(max_correspondences, n_deformed, n_target)
    else:
        n_correspondences = min(n_deformed, n_target)
    
    
    # Use subset for efficiency if needed
    if n_correspondences < n_deformed:
        # Sample uniformly from deformed vertices
        deformed_indices = np.random.choice(n_deformed, n_correspondences, replace=False)
        deformed_subset = deformed_vertices[deformed_indices]
    else:
        deformed_indices = np.arange(n_deformed, dtype=np.int32)
        deformed_subset = deformed_vertices
    
    if n_correspondences < n_target:
        # Sample uniformly from target vertices
        target_indices = np.random.choice(n_target, n_correspondences, replace=False)
        target_subset = target_vertices[target_indices]
    else:
        target_indices = np.arange(n_target, dtype=np.int32)
        target_subset = target_vertices
    
    # Compute pairwise Euclidean distances
    distance_matrix = cdist(deformed_subset, target_subset, metric='euclidean')
    
    # Solve assignment problem using Hungarian algorithm
    deformed_assign_indices, target_assign_indices = linear_sum_assignment(distance_matrix)
    
    # Map back to original indices
    source_indices = deformed_indices[deformed_assign_indices]
    target_indices_mapped = target_indices[target_assign_indices]
    assignment_distances = distance_matrix[deformed_assign_indices, target_assign_indices]
    
    
    return source_indices, target_indices_mapped, assignment_distances

# Import global resource manager
from ..utils.common import resource_manager

# ========================================
# PAIR PROCESSING FUNCTIONS
# ========================================



# Import shared functions
from ..utils.common import find_model_files



def compute_dino_correspondences_from_precomputed(models_dir, source_model, target_model, output_file, pair_folder):
    """
    Compute DINO correspondences using pre-computed DINO features.
    
    Args:
        models_dir: Path to the models directory containing pre-computed DINO features
        source_model: Source model name
        target_model: Target model name
        output_file: Path to save correspondences
        pair_folder: Path to pair directory
        
    Returns:
        True if successful, False otherwise
    """
    
    # Set default values
    
    # Find source and target model files
    source_files = find_model_files(models_dir, source_model)
    target_files = find_model_files(models_dir, target_model)
    
    # CHECK FOR AUTORIG DEFORMED MESH FIRST
    autorig_coarse_dir = os.path.join(pair_folder, "coarse")
    autorig_deformed_mesh = os.path.join(autorig_coarse_dir, f"{source_model}_deformed_to_{target_model}.obj")
    
    if os.path.exists(autorig_deformed_mesh):
        source_files['obj'] = autorig_deformed_mesh
        print(f"    Using AutoRig deformed mesh: {os.path.basename(autorig_deformed_mesh)}")
    else:
        print(f"    No coarse output found, using original mesh: {os.path.basename(source_files['obj'])}")
    
    # Check if all required files exist
    missing_files = []
    if not source_files['obj'] or not os.path.exists(source_files['obj']):
        missing_files.append(f"source mesh: {source_model}.obj")
    if not target_files['obj'] or not os.path.exists(target_files['obj']):
        missing_files.append(f"target mesh: {target_model}.obj")
    if not source_files['dino_features'] or not os.path.exists(source_files['dino_features']):
        missing_files.append(f"source DINO features: {source_model}_dino_features.npy")
    if not target_files['dino_features'] or not os.path.exists(target_files['dino_features']):
        missing_files.append(f"target DINO features: {target_model}_dino_features.npy")
    
    if missing_files:
        raise ValueError(f"Missing files: {', '.join(missing_files)}")
    
    
    # Check for keypoints
    source_keypoints = source_files['npy'] if source_files['npy'] and os.path.exists(source_files['npy']) else None
    target_keypoints = target_files['npy'] if target_files['npy'] and os.path.exists(target_files['npy']) else None
    
    
    # Load meshes
    source_mesh = trimesh.load(source_files['obj'])
    target_mesh = trimesh.load(target_files['obj'])
    
    # Load pre-computed DINO features
    source_dino_features = np.load(source_files['dino_features'])
    target_dino_features = np.load(target_files['dino_features'])
    
    # Convert to torch tensors
    source_features_tensor = torch.tensor(source_dino_features, dtype=torch.float32, device=device)
    target_features_tensor = torch.tensor(target_dino_features, dtype=torch.float32, device=device)
    source_vertices_tensor = torch.tensor(source_mesh.vertices, dtype=torch.float32, device=device)
    target_vertices_tensor = torch.tensor(target_mesh.vertices, dtype=torch.float32, device=device)
    
    # Normalize features
    source_features_normalized = torch.nn.functional.normalize(source_features_tensor, p=2, dim=1)
    target_features_normalized = torch.nn.functional.normalize(target_features_tensor, p=2, dim=1)
    
    # Find correspondences using lightweight correspondence finder
    correspondences = find_correspondences_lightweight(
        source_features_normalized, target_features_normalized,
        source_vertices_tensor, target_vertices_tensor,
        source_mesh, target_mesh,
        source_keypoints, target_keypoints,
        bidirectional_consistency=False,  # Set to False for coverage
        use_global_optimization=True
    )
    
    # Create features dictionaries for saving
    features1 = {
        'dino': source_features_normalized,
        'vertices': source_vertices_tensor,
        'dino_coverage': 1.0,  # Pre-computed features have full coverage
        'dino_consistency_loss': 0.0
    }
    features2 = {
        'dino': target_features_normalized,
        'vertices': target_vertices_tensor,
        'dino_coverage': 1.0,  # Pre-computed features have full coverage
        'dino_consistency_loss': 0.0
    }
    
    # Save correspondences using lightweight saver
    save_correspondences_lightweight(correspondences, features1, features2, output_file)
    
    return True

def find_correspondences_lightweight(features1, features2, vertices1, vertices2, 
                                   mesh1, mesh2, keypoints1_path, keypoints2_path,
                                   bidirectional_consistency=False, use_global_optimization=False):
    """
    Lightweight correspondence finder that doesn't load DINO model.
    Uses pre-computed features and optional keypoint labeling.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from sklearn.neighbors import NearestNeighbors
    from collections import defaultdict
    
    
    features1_cpu = features1.cpu().numpy()
    features2_cpu = features2.cpu().numpy()
    vertices1_cpu = vertices1.cpu().numpy()
    vertices2_cpu = vertices2.cpu().numpy()
    
    # Check if we should use keypoint label constraints
    use_keypoint_labels = keypoints1_path and keypoints2_path
    
    if use_keypoint_labels:
        
        # Load keypoints and compute vertex labels
        keypoints1 = np.load(keypoints1_path)
        keypoints2 = np.load(keypoints2_path)
        
        # Use the same labeling function from dino_match.py
        vertex_labels1, _ = label_vertices_by_keypoints(mesh1, keypoints1)
        vertex_labels2, _ = label_vertices_by_keypoints(mesh2, keypoints2)
        
        # Create vertex indices (all vertices)
        vertex_indices1 = list(range(len(features1_cpu)))
        vertex_indices2 = list(range(len(features2_cpu)))
        
        # Use label-constrained correspondence finding
        correspondences = find_vertex_correspondences_by_labels(
            features1_cpu, vertex_indices1, vertex_labels1,
            features2_cpu, vertex_indices2, vertex_labels2,
            vertices1_cpu, vertices2_cpu,
            bidirectional_consistency, use_global_optimization
        )
        
    else:
        # Use standard matching methods without label constraints
        print(f"    Using standard matching without keypoint constraints")
        
        if use_global_optimization:
            # Use global optimization for optimal bijective assignment
            print(f"    Using global optimization")
            cost_matrix = cdist(features1_cpu, features2_cpu, metric='euclidean')
            
            # Handle different sizes
            n1, n2 = cost_matrix.shape
            if n1 != n2:
                max_cost = cost_matrix.max() * 3
                if n1 < n2:
                    cost_matrix = np.vstack([cost_matrix, np.full((n2 - n1, n2), max_cost)])
                else:
                    cost_matrix = np.hstack([cost_matrix, np.full((n1, n1 - n2), max_cost)])
            
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            correspondences = []
            for i, j in zip(row_indices, col_indices):
                if i < len(features1_cpu) and j < len(features2_cpu):
                    distance = float(np.linalg.norm(features1_cpu[i] - features2_cpu[j]))
                    correspondences.append({
                        'vertex1_idx': int(i), 'vertex2_idx': int(j),
                        'feature_distance': distance,
                        'euclidean_distance': float(np.linalg.norm(vertices1_cpu[i] - vertices2_cpu[j]))
                    })
        else:
            # Use nearest neighbor matching
            print(f"    Using nearest neighbor matching ({'bidirectional' if bidirectional_consistency else 'unidirectional'})")
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(features2_cpu)
            distances_ab, indices_ab = nn_model.kneighbors(features1_cpu)
            distances_ab, indices_ab = distances_ab.flatten(), indices_ab.flatten()
            
            correspondences = []
            
            if bidirectional_consistency:
                nn_model_reverse = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_model_reverse.fit(features1_cpu)
                distances_ba, indices_ba = nn_model_reverse.kneighbors(features2_cpu)
                indices_ba = indices_ba.flatten()
                
                for i in range(len(features1_cpu)):
                    j, distance = indices_ab[i], distances_ab[i]
                    if indices_ba[j] == i:
                        correspondences.append({
                            'vertex1_idx': int(i), 'vertex2_idx': int(j),
                            'feature_distance': float(distance),
                            'euclidean_distance': float(np.linalg.norm(vertices1_cpu[i] - vertices2_cpu[j]))
                        })
            else:
                for i in range(len(features1_cpu)):
                    j, distance = indices_ab[i], distances_ab[i]
                    correspondences.append({
                        'vertex1_idx': int(i), 'vertex2_idx': int(j),
                        'feature_distance': float(distance),
                        'euclidean_distance': float(np.linalg.norm(vertices1_cpu[i] - vertices2_cpu[j]))
                    })
    
    
    return correspondences

def save_correspondences_lightweight(correspondences, features1_dict, features2_dict, output_file):
    """Lightweight correspondence saver that doesn't depend on DinoMatcher."""
    vertices1 = features1_dict['vertices']
    vertices2 = features2_dict['vertices']
    
    if torch.is_tensor(vertices1):
        vertices1 = vertices1.cpu().numpy()
    if torch.is_tensor(vertices2):
        vertices2 = vertices2.cpu().numpy()
    
    correspondence_dict = {str(corr['vertex1_idx']): corr['vertex2_idx'] for corr in correspondences}
    
    output_data = {
        'correspondences': correspondence_dict,
        'metadata': {
            'total_correspondences': len(correspondences),
            'method': 'DINO_precomputed_features_lightweight',
            'parameters': {
                'uses_precomputed_features': True,
                'lightweight_processing': True
            },
            'statistics': {
                'avg_feature_distance': float(np.mean([c['feature_distance'] for c in correspondences])) if correspondences else 0,
                'avg_euclidean_distance': float(np.mean([c['euclidean_distance'] for c in correspondences])) if correspondences else 0,
                'dino_consistency_loss_mesh1': float(features1_dict.get('dino_consistency_loss', 0)),
                'dino_consistency_loss_mesh2': float(features2_dict.get('dino_consistency_loss', 0)),
                'dino_coverage_mesh1': float(features1_dict.get('dino_coverage', 0)),
                'dino_coverage_mesh2': float(features2_dict.get('dino_coverage', 0)),
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    

# Import shared functions  
from ..utils.common import compute_geodesic_distances, label_vertices_by_keypoints

def find_vertex_correspondences_by_labels(features1, vertex_indices1, vertex_labels1,
                                          features2, vertex_indices2, vertex_labels2,
                                          mesh1_vertices=None, mesh2_vertices=None,
                                          bidirectional_consistency=False, use_global_optimization=False):
    """Find bijective correspondences between vertices with the same labels using DINO features"""
    from collections import defaultdict
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from sklearn.neighbors import NearestNeighbors
    
    
    # Group vertices by their labels
    label_groups1 = defaultdict(list)
    label_groups2 = defaultdict(list)
    feature_groups1 = defaultdict(list)
    feature_groups2 = defaultdict(list)
    
    for i, vertex_idx in enumerate(vertex_indices1):
        label = vertex_labels1[vertex_idx]
        label_groups1[label].append(vertex_idx)
        feature_groups1[label].append(features1[i])
    
    for i, vertex_idx in enumerate(vertex_indices2):
        label = vertex_labels2[vertex_idx]
        label_groups2[label].append(vertex_idx)
        feature_groups2[label].append(features2[i])
    
    # Find common labels
    common_labels = set(label_groups1.keys()).intersection(set(label_groups2.keys()))
    
    all_correspondences = []
    
    # Process each label group separately
    for label in common_labels:
        vertices1 = label_groups1[label]
        vertices2 = label_groups2[label]
        features1_label = np.array(feature_groups1[label])
        features2_label = np.array(feature_groups2[label])
        
        if len(vertices1) == 0 or len(vertices2) == 0:
            continue
        
        
        # Find correspondences within this label group using the specified matching strategy
        label_correspondences = find_vertex_correspondences_within_label(
            features1_label, vertices1,
            features2_label, vertices2,
            label, mesh1_vertices, mesh2_vertices,
            bidirectional_consistency, use_global_optimization
        )
        
        all_correspondences.extend(label_correspondences)
    
    
    # Verify bijective property across all labels
    source_vertices = set(c['vertex1_idx'] for c in all_correspondences)
    target_vertices = set(c['vertex2_idx'] for c in all_correspondences)
    
    if len(target_vertices) != len(all_correspondences):
        print("    ERROR: Global bijective property violated - some target vertices are used multiple times!")
    
    return all_correspondences

def find_vertex_correspondences_within_label(features1, vertices1, features2, vertices2, label, 
                                            mesh1_vertices=None, mesh2_vertices=None,
                                            bidirectional_consistency=False, use_global_optimization=False):
    """Find bijective correspondences between vertices within a single label group using DINO features"""
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from sklearn.neighbors import NearestNeighbors
    
    if len(features1) == 0 or len(features2) == 0:
        return []
    
    # Compute Euclidean distance matrix for DINO features
    euclidean_distance_matrix = cdist(features1, features2, metric='euclidean')
    
    # Choose matching strategy based on parameters
    if use_global_optimization:
        # Use Hungarian algorithm for optimal bijective assignment (minimize Euclidean distance)
        if len(features1) != len(features2):
            # Handle different numbers of vertices
            if len(features1) < len(features2):
                # More target vertices than source vertices
                max_cost = euclidean_distance_matrix.max() * 2 if euclidean_distance_matrix.size > 0 else 1.0
                padding = np.full((len(features2) - len(features1), len(features2)), max_cost)
                padded_matrix = np.vstack([euclidean_distance_matrix, padding])
                row_indices, col_indices = linear_sum_assignment(padded_matrix)
                # Filter out dummy assignments
                valid_assignments = row_indices < len(features1)
                row_indices = row_indices[valid_assignments]
                col_indices = col_indices[valid_assignments]
            else:
                # More source vertices than target vertices
                max_cost = euclidean_distance_matrix.max() * 2 if euclidean_distance_matrix.size > 0 else 1.0
                padding = np.full((len(features1), len(features1) - len(features2)), max_cost)
                padded_matrix = np.hstack([euclidean_distance_matrix, padding])
                row_indices, col_indices = linear_sum_assignment(padded_matrix)
                # Filter out dummy assignments
                valid_assignments = col_indices < len(features2)
                row_indices = row_indices[valid_assignments]
                col_indices = col_indices[valid_assignments]
        else:
            # Equal numbers - direct assignment
            row_indices, col_indices = linear_sum_assignment(euclidean_distance_matrix)
        
        # Create correspondences from the optimal assignment
        correspondences = []
        used_target_vertices = set()
        
        for i, j in zip(row_indices, col_indices):
            # Ensure we don't exceed array bounds
            if i >= len(vertices1) or j >= len(vertices2):
                continue
                
            vertex1_idx = vertices1[i]
            vertex2_idx = vertices2[j]
            
            # Double-check bijective property
            if vertex2_idx in used_target_vertices:
                print(f"        Warning: Target vertex {vertex2_idx} already used in label {label}, skipping...")
                continue
            used_target_vertices.add(vertex2_idx)
            
            # Get feature distance
            feature_distance = euclidean_distance_matrix[i, j] if euclidean_distance_matrix.size > 0 else 0.0
            
            # Compute 3D Euclidean distance if vertex positions are provided
            euclidean_3d_distance = feature_distance  # Default to feature distance
            if mesh1_vertices is not None and mesh2_vertices is not None:
                euclidean_3d_distance = float(np.linalg.norm(mesh1_vertices[vertex1_idx] - mesh2_vertices[vertex2_idx]))
            
            correspondences.append({
                'vertex1_idx': int(vertex1_idx),
                'vertex2_idx': int(vertex2_idx),
                'feature_distance': float(feature_distance),
                'euclidean_distance': float(euclidean_3d_distance),
                'label': int(label)
            })
        
    else:
        # Use nearest neighbor matching with optional bidirectional consistency
        print(f"      Using nearest neighbor matching ({'bidirectional' if bidirectional_consistency else 'unidirectional'}) for label {label}")
        
        # Find nearest neighbors from features1 to features2
        nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_model.fit(features2)
        distances_ab, indices_ab = nn_model.kneighbors(features1)
        distances_ab, indices_ab = distances_ab.flatten(), indices_ab.flatten()
        
        correspondences = []
        
        if bidirectional_consistency:
            # Find nearest neighbors from features2 to features1 for consistency check
            nn_model_reverse = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model_reverse.fit(features1)
            distances_ba, indices_ba = nn_model_reverse.kneighbors(features2)
            indices_ba = indices_ba.flatten()
            
            # Only keep correspondences that are mutually nearest neighbors
            for i in range(len(features1)):
                j = indices_ab[i]
                distance = distances_ab[i]
                
                # Check if j->i is also a nearest neighbor relationship
                if indices_ba[j] == i:
                    vertex1_idx = vertices1[i]
                    vertex2_idx = vertices2[j]
                    
                    # Compute 3D Euclidean distance if vertex positions are provided
                    euclidean_3d_distance = distance  # Default to feature distance
                    if mesh1_vertices is not None and mesh2_vertices is not None:
                        euclidean_3d_distance = float(np.linalg.norm(mesh1_vertices[vertex1_idx] - mesh2_vertices[vertex2_idx]))
                    
                    correspondences.append({
                        'vertex1_idx': int(vertex1_idx),
                        'vertex2_idx': int(vertex2_idx),
                        'feature_distance': float(distance),
                        'euclidean_distance': float(euclidean_3d_distance),
                        'label': int(label)
                    })
        else:
            # Unidirectional matching - just use nearest neighbors
            for i in range(len(features1)):
                j = indices_ab[i]
                distance = distances_ab[i]
                
                vertex1_idx = vertices1[i]
                vertex2_idx = vertices2[j]
                
                # Compute 3D Euclidean distance if vertex positions are provided
                euclidean_3d_distance = distance  # Default to feature distance
                if mesh1_vertices is not None and mesh2_vertices is not None:
                    euclidean_3d_distance = float(np.linalg.norm(mesh1_vertices[vertex1_idx] - mesh2_vertices[vertex2_idx]))
                
                correspondences.append({
                    'vertex1_idx': int(vertex1_idx),
                    'vertex2_idx': int(vertex2_idx),
                    'feature_distance': float(distance),
                    'euclidean_distance': float(euclidean_3d_distance),
                    'label': int(label)
                })
    
    
    return correspondences

def compute_dino_correspondences(models_dir, source_model, target_model, output_file, pair_folder):
    """
    Compute DINO correspondences between source and target models.
    First tries to use pre-computed DINO features, falls back to full pipeline if not available.
    
    Args:
        models_dir: Path to the models directory containing DINO features
        source_model: Source model name
        target_model: Target model name
        output_file: Path to save correspondences
        use_hungarian: Whether to use Hungarian algorithm for bijective correspondences
        use_keypoint_labels: Whether to use keypoint labeling for correspondence constraints
        
    Returns:
        True if successful, False otherwise
    """
    
    # First try to use pre-computed DINO features
    source_files = find_model_files(models_dir, source_model)
    target_files = find_model_files(models_dir, target_model)
    
    # Check if pre-computed DINO features are available
    if (source_files['dino_features'] and os.path.exists(source_files['dino_features']) and
        target_files['dino_features'] and os.path.exists(target_files['dino_features'])):
        
        return compute_dino_correspondences_from_precomputed(models_dir, source_model, target_model, output_file, pair_folder)
    
    else:
        print(f"    Pre-computed DINO features not found, falling back to full pipeline...")
        print(f"    Pre-computed DINO features not found, please run DINO feature extraction first")
        return False


def create_pair_config(base_config, pair_folder, source_model, target_model, models_dir):
    """
    Create a config for a specific pair by updating paths in the base config.
    
    Args:
        base_config: Base configuration dictionary
        pair_folder: Path to the pair directory
        source_model: Source model name
        target_model: Target model name
        models_dir: Path to models directory
        
    Returns:
        Updated config dictionary
    """
    # Create a copy of the base config
    config = copy.deepcopy(base_config)
    
    # Find source and target model files in models directory
    source_files = find_model_files(models_dir, source_model)
    target_files = find_model_files(models_dir, target_model)
    
    # CHECK FOR AUTORIG DEFORMED MESH FOR ARAP ALIGNMENT
    autorig_coarse_dir = os.path.join(pair_folder, "coarse")
    autorig_deformed_mesh = os.path.join(autorig_coarse_dir, f"{source_model}_deformed_to_{target_model}.obj")
    
    if os.path.exists(autorig_deformed_mesh):
        source_files['obj'] = autorig_deformed_mesh
    
    # Update paths for this specific pair
    dino_result_dir = os.path.join(pair_folder, "dino_result")
    dense_dir = os.path.join(pair_folder, "dense")
    
    # Ensure paths section exists
    if 'paths' not in config:
        config['paths'] = {}
        
    # Update config paths to use models from models directory (or AutoRig if available)
    config['paths']['source_mesh'] = source_files['obj']
    config['paths']['target_mesh'] = target_files['obj']
    config['paths']['correspondence'] = os.path.join(dino_result_dir, f"{source_model}_to_{target_model}_correspondences.json")
    config['paths']['log_dir'] = dense_dir
    
    # Ensure output section exists
    if 'output' not in config:
        config['output'] = {}
    config['output']['final_mesh'] = os.path.join(dense_dir, f"{source_model}_aligned_to_{target_model}.obj")
    
    return config

def process_single_pair(pair_id, source_model, target_model, pair_folder, base_config, models_dir):
    """
    Process a single pair with DINO correspondence computation and ARAP alignment.
    
    Args:
        pair_id: Pair ID
        source_model: Source model name
        target_model: Target model name
        pair_folder: Path to pair directory
        base_config: Base configuration dictionary
        models_dir: Path to models directory
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Create output directories
        dino_result_dir = os.path.join(pair_folder, "dino_result")
        dense_dir = os.path.join(pair_folder, "dense")
        os.makedirs(dino_result_dir, exist_ok=True)
        os.makedirs(dense_dir, exist_ok=True)
        
        # Set up output files
        output_file = os.path.join(dense_dir, f"{source_model}_aligned_to_{target_model}.obj")
        done_marker = os.path.join(pair_folder, ".alignment_done")
        
        # Step 1: Compute DINO correspondences
        correspondence_file = os.path.join(dino_result_dir, f"{source_model}_to_{target_model}_correspondences.json")
        
        # Check if DINO correspondences already exist
        if not os.path.exists(correspondence_file):
            success = compute_dino_correspondences(models_dir, source_model, target_model, correspondence_file, pair_folder)
            if not success:
                raise ValueError("Failed to compute DINO correspondences")
        
        # Verify correspondences file exists after processing
        if not os.path.exists(correspondence_file):
            raise ValueError(f"DINO correspondences not found: {correspondence_file}")
        
        # Step 2: Create config for this pair (using pre-loaded base config)
        pair_config = create_pair_config(base_config, pair_folder, source_model, target_model, models_dir)
        
        # Verify required files exist
        source_obj = pair_config['paths']['source_mesh']
        target_obj = pair_config['paths']['target_mesh']
        
        if not os.path.exists(source_obj):
            raise ValueError(f"Source mesh not found: {source_obj}")
        if not os.path.exists(target_obj):
            raise ValueError(f"Target mesh not found: {target_obj}")
        if not os.path.exists(correspondence_file):
            raise ValueError(f"DINO correspondences not found: {correspondence_file}")
        
        
        # Create registration object with the pair-specific config
        registration = ARAPRegistration(pair_config, from_dict=True)
        
        # Run optimization
        vertices = registration.optimize()
        
        # Update deformed vertices
        registration.deformed_vertices = vertices
        
        # Save result (the final mesh should already be saved by optimize method)
        # But also ensure it's saved to the expected location
        if not os.path.exists(output_file):
            # Try to find the latest iteration file as fallback
            import glob
            iteration_files = glob.glob(os.path.join(dense_dir, "iteration_*.obj"))
            if iteration_files:
                # Sort by iteration number and get the latest
                def extract_iteration_number(filename):
                    try:
                        # Extract iteration number from filename like "iteration_004.obj"
                        basename = os.path.basename(filename)
                        if not isinstance(basename, str):
                            return 0
                        parts = basename.split('_')
                        if len(parts) >= 2:
                            number_part = parts[-1].split('.')[0]
                            if number_part.isdigit():
                                return int(number_part)
                        return 0
                    except (ValueError, IndexError, AttributeError):
                        return 0
                
                # Filter out any non-string entries and sort
                valid_files = [f for f in iteration_files if isinstance(f, str)]
                if valid_files:
                    valid_files.sort(key=extract_iteration_number)
                    latest_iteration = valid_files[-1]
                    print(f"    Copying latest iteration file to final output: {latest_iteration} -> {output_file}")
                    import shutil
                    shutil.copy2(latest_iteration, output_file)
            else:
                # Fallback to save_result method
                result_mesh = registration.save_result(output_file)
        else:
            pass
        
        # Step 3: Generate final correspondences (moved from evaluation.py)
        try:
            # Create dense_result directory for final correspondences
            dense_result_dir = os.path.join(pair_folder, "dense_result")
            os.makedirs(dense_result_dir, exist_ok=True)
            
            # Load aligned mesh and target mesh
            import trimesh
            aligned_mesh = trimesh.load(output_file)
            
            # Find target mesh in models directory
            target_files = find_model_files(models_dir, target_model)
            target_obj = target_files['obj']
            if not target_obj or not os.path.exists(target_obj):
                raise ValueError(f"Target mesh not found: {target_model}")
            
            target_mesh = trimesh.load(target_obj)
            
            # Compute final correspondences using bijective assignment
            from src.evaluation import compute_mge_direct_correspondence_with_hungarian, save_correspondences
            from src.utils.config import ConfigManager
            
            # Get max_correspondences from config
            config_manager = ConfigManager()
            evaluation_config = config_manager.get_evaluation_config()
            max_correspondences = evaluation_config['max_correspondences']
            
            mge, stats = compute_mge_direct_correspondence_with_hungarian(
                aligned_mesh, target_mesh, max_correspondences=max_correspondences, target_mesh_file=target_obj
            )
            
            # Save final correspondences with MGE
            final_correspondences_file = os.path.join(dense_result_dir, f"{source_model}_to_{target_model}_eval_correspondences.json")
            save_correspondences(stats['correspondences'], final_correspondences_file, mge=mge)
            
            
        except Exception as e:
            print(f"    Warning: Failed to compute final correspondences: {e}")
            # Continue with the rest of the process even if correspondence computation fails
        
        # Create completion marker
        with open(done_marker, 'w') as f:
                f.write(f"ARAP alignment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {source_model}\n")
                f.write(f"Target: {target_model}\n")
                f.write(f"Aligned mesh: {os.path.basename(output_file)}\n")
                f.write(f"Correspondences: {os.path.basename(correspondence_file)}\n")
        
        
        return True
        
    except Exception as e:
        print(f"Process single pair failed: {e}")
        raise e


# ========================================
# ARAP REGISTRATION CLASS
# ========================================

class ARAPRegistration:
    def __init__(self, config_or_path, from_dict=False):
        """
        Initialize with parameters from a YAML config file or dictionary.
        
        Args:
            config_or_path: Path to YAML configuration file OR config dict
            from_dict: If True, config_or_path is treated as a dictionary
        """
        # Load configuration
        if from_dict:
            self.config = config_or_path
        else:
            with open(config_or_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Use paths directly from config
        source_mesh_path = self.config['paths']['source_mesh']
        target_mesh_path = self.config['paths']['target_mesh']
        correspondence_path = self.config['paths']['correspondence']
        
        # Note: confidence is no longer used in the pipeline
        
        # Create log directory if it doesn't exist
        self.log_dir = self.config['paths']['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Extract weights
        self.arap_weight = self.config['weights']['arap']
        self.correspondence_weight = self.config['weights']['correspondence']
        self.smoothness_weight = self.config['weights']['smoothness']
        
        # Initialize config manager for accessing configuration values
        from ..utils.config import ConfigManager
        self.config_manager = ConfigManager()
        
        # Extract algorithm parameters
        self.filter_correspondences = True  # Always use correspondence filtering
        
        # Extract correspondence end iteration parameter
        self.correspondence_end_iteration = self.config['parameters'].get('correspondence_end_iteration', -1)
        
        # Extract correspondence update parameters
        self.correspondence_update_frequency = self.config['parameters'].get('correspondence_update_frequency', 5)
        self.correspondence_update_start_iteration = self.config['parameters'].get('correspondence_update_start_iteration', 1)
        
        # Extract filter threshold parameters with defaults
        self.filter_params = self.config.get('filter_thresholds', {})
        self.consistency_threshold = self.filter_params.get('consistency_threshold', 0.3)
        
        # Load meshes
        self.source_mesh = trimesh.load(source_mesh_path)
        self.target_mesh = trimesh.load(target_mesh_path)
        
        # Store source model name for final alignment output
        self.source_model_name = os.path.splitext(os.path.basename(source_mesh_path))[0]
        
        
        # Load correspondences
        with open(correspondence_path, 'r') as f:
            correspondence_data = json.load(f)
        
        # Check if this is the vertex_correspondence.py format
        if 'correspondences' in correspondence_data:
            # Format from vertex_correspondence.py
            self.correspondences = correspondence_data['correspondences']
        else:
            # Direct correspondence format
            self.correspondences = correspondence_data
        
        # Convert string keys to integers
        self.correspondences = {int(k): int(v) for k, v in self.correspondences.items()}
        
        
        # Initialize correspondence data
        # These values were previously set in prepare_rigid_alignment
        self.correspondence_indices = np.array(list(self.correspondences.keys()), dtype=np.int32)
        self.source_vertices = self.source_mesh.vertices[self.correspondence_indices]
        target_indices = [self.correspondences[idx] for idx in self.correspondence_indices]
        self.target_vertices = self.target_mesh.vertices[target_indices]
        self.target_correspondence_points = self.target_mesh.vertices[target_indices]
        self.correspondence_weights = np.ones(len(self.correspondence_indices))
        
        # Skip rigid alignment and use source vertices directly
        self.aligned_vertices = self.source_mesh.vertices.copy()
        
        # Filter outlier correspondences if enabled
        if self.filter_correspondences:
            self.filter_outlier_correspondences()
        
        # Now build neighborhood and Laplacian for ARAP optimization
        self.build_laplacian()
        
        # Initialize correspondence weights for adaptive reweighting
        self.update_correspondence_weights = {}
        for idx in self.correspondence_indices:
            self.update_correspondence_weights[idx] = 1.0
    
    def filter_outlier_correspondences(self):
        """
        Filter outlier correspondences using neighborhood consistency checking.
        Keep correspondences that are consistent with their local neighborhood.
        """
        original_count = len(self.correspondence_indices)
        
        # Compute local consistency scores
        consistency_scores = np.ones(len(self.correspondence_indices))
        
        # Build KD-tree for nearest neighbor lookup in source mesh (with caching)
        source_mesh_hash = hash(self.source_mesh.vertices.tobytes())
        source_tree = resource_manager.get_kdtree(self.source_mesh.vertices, f"source_{source_mesh_hash}")
        
        # For each correspondence, check geometric consistency with neighbors
        k_neighbors = min(20, len(self.source_mesh.vertices) // 100)
        
        for i, (source_idx, target_idx) in enumerate(zip(self.correspondence_indices, 
                                                       [self.correspondences[idx] for idx in self.correspondence_indices])):
            # Find k nearest neighbors in source mesh
            _, source_neighbor_indices = source_tree.query(self.source_mesh.vertices[source_idx], k=k_neighbors)
            
            # Check how many of these neighbors have consistent correspondences
            consistent_count = 0
            total_checked = 0
            
            for neighbor_idx in source_neighbor_indices[1:]:  # Skip the point itself
                if neighbor_idx in self.correspondences:
                    total_checked += 1
                    neighbor_target_idx = self.correspondences[neighbor_idx]
                    
                    # Check if this correspondence preserves distance relatively well
                    source_dist = np.linalg.norm(
                        self.source_mesh.vertices[source_idx] - self.source_mesh.vertices[neighbor_idx]
                    )
                    target_dist = np.linalg.norm(
                        self.target_mesh.vertices[target_idx] - self.target_mesh.vertices[neighbor_target_idx]
                    )
                    
                    # Calculate distance ratio (should be close to 1 for similar shapes)
                    if source_dist > 1e-6 and target_dist > 1e-6:
                        ratio = min(source_dist/target_dist, target_dist/source_dist)
                        if ratio > 0.5:  # Fixed reasonable threshold for distance preservation
                            consistent_count += 1
            
            # Calculate consistency score
            if total_checked > 0:
                consistency_scores[i] = consistent_count / total_checked
            else:
                consistency_scores[i] = 0.5  # Neutral score if no neighbors to check
        
        # Filter based on consistency threshold only
        valid_indices = []
        
        for i, (source_idx, consistency) in enumerate(zip(self.correspondence_indices, consistency_scores)):
            if consistency > self.consistency_threshold:
                valid_indices.append(i)
        
        # Update correspondence data
        self.correspondence_indices = self.correspondence_indices[valid_indices]
        self.source_vertices = self.source_vertices[valid_indices]
        self.target_vertices = self.target_vertices[valid_indices]
        self.correspondence_weights = np.ones(len(valid_indices))
        self.target_correspondence_points = self.target_mesh.vertices[[
            self.correspondences[idx] for idx in self.correspondence_indices
        ]]
        

    def build_laplacian(self):
        """
        Build the cotangent Laplacian matrix and neighborhood structure with GPU acceleration.
        Uses pre-allocated GPU buffers and batching.
        """
        vertices = self.aligned_vertices
        faces = self.source_mesh.faces
        
        # Get mesh edges for neighborhood structure
        edges = self.source_mesh.edges
        
        # Store vertex neighborhood
        n_vertices = len(vertices)
        self.vertex_neighbors = [[] for _ in range(n_vertices)]
        
        # Create edge-to-face lookup for faster processing
        edge_to_faces = {}
        
        # For each face, add it to the edge-to-face lookup
        for face_idx, face in enumerate(faces):
            # Get the three edges of the triangle
            edges_in_face = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            for i, j in edges_in_face:
                edge = tuple(sorted([i, j]))  # Canonicalize edge orientation
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Move vertices to GPU for computation (use pre-allocated buffer)
        vertices_tensor = resource_manager.get_gpu_buffer(n_vertices)
        vertices_tensor[:] = torch.tensor(vertices, dtype=torch.float32, device=device)
        
        # Initialize cotangent weights dictionary
        cotangent_weights = {}
        
        # Process edges in batches to utilize GPU
        edge_list = list(edges)
        batch_size = 10000  # Process edges in batches
        
        for batch_start in range(0, len(edge_list), batch_size):
            batch_end = min(batch_start + batch_size, len(edge_list))
            edge_batch = edge_list[batch_start:batch_end]
            
            for i, j in edge_batch:
                # Ensure canonical edge ordering
                edge = tuple(sorted([i, j]))
                
                # Update neighborhood information
                self.vertex_neighbors[i].append(j)
                self.vertex_neighbors[j].append(i)
                
                # Get faces that contain this edge
                edge_faces = edge_to_faces.get(edge, [])
                
                # Skip if no adjacent faces (boundary edge)
                if not edge_faces:
                    cotangent_weights[(i, j)] = cotangent_weights[(j, i)] = 1e-6
                    continue
                
                # Compute cotangent weight for this edge
                weight = 0
                
                for face_idx in edge_faces:
                    face = faces[face_idx]
                    # Find the third vertex in the face
                    third_vertex = [v for v in face if v != i and v != j][0]
                    
                    # Use GPU for faster vector operations
                    vi = vertices_tensor[i]
                    vj = vertices_tensor[j]
                    vk = vertices_tensor[third_vertex]
                    
                    # Compute edge vectors
                    e1 = vj - vk
                    e2 = vi - vk
                    
                    # Compute cotangent using dot product and cross product
                    dot_product = torch.dot(e1, e2)
                    cross_product = torch.cross(e1, e2)
                    cross_product_norm = torch.norm(cross_product)
                    
                    # Add cotangent value (ensure numerical stability)
                    if cross_product_norm > 1e-10:
                        cot = dot_product / cross_product_norm
                        weight += 0.5 * cot.item()
                
                # Store weight with minimum threshold to ensure stability
                cotangent_weights[(i, j)] = cotangent_weights[(j, i)] = max(weight, 1e-6)
        
        # Construct Laplacian matrix in COO format
        rows = []
        cols = []
        data = []
        
        for i in range(n_vertices):
            neighbors = self.vertex_neighbors[i]
            if not neighbors:
                # If vertex has no neighbors, add small diagonal element
                rows.append(i)
                cols.append(i)
                data.append(1e-6)
                continue
                
            weight_sum = 0
            for j in neighbors:
                weight = cotangent_weights.get((i, j), 0)
                weight_sum += weight
                rows.append(i)
                cols.append(j)
                data.append(-weight)
            
            rows.append(i)
            cols.append(i)
            data.append(weight_sum)
        
        # Create sparse Laplacian matrix
        self.L = sp.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
        self.cotangent_weights = cotangent_weights
        

    def update_correspondences_bijective(self, V_current, iteration, max_iterations):
        """
        Update correspondences using bijective correspondence finding from evaluation.py.
        This finds new one-to-one correspondences between the current deformed mesh and target mesh.
        """
        
        # Use all vertices for correspondence finding
        max_correspondences = len(V_current)
        
        # Find new bijective correspondences between current deformed vertices and target vertices
        source_indices, target_indices, assignment_distances = find_bijective_correspondences(
            V_current, self.target_mesh.vertices, max_correspondences
        )
        
        # Update correspondence data structures
        self.correspondence_indices = source_indices
        self.target_correspondence_points = self.target_mesh.vertices[target_indices]
        
        # Create new correspondences dictionary
        new_correspondences = {}
        
        for i, (src_idx, tgt_idx, dist) in enumerate(zip(source_indices, target_indices, assignment_distances)):
            new_correspondences[int(src_idx)] = int(tgt_idx)
        
        # Update internal data structures
        self.correspondences = new_correspondences
        
        # Update correspondence weights array
        self.correspondence_weights = np.ones(len(self.correspondence_indices))
        
        # Reset adaptive weights
        self.update_correspondence_weights = {}
        for idx in self.correspondence_indices:
            self.update_correspondence_weights[idx] = 1.0
        
        
        # Save updated correspondences to log directory if available
        if hasattr(self, 'log_dir') and self.log_dir:
            correspondence_file = os.path.join(self.log_dir, f'correspondences_iteration_{iteration:03d}.json')
            correspondence_data = {
                'correspondences': new_correspondences,
                'metadata': {
                    'iteration': iteration,
                    'total_correspondences': len(new_correspondences),
                    'method': 'bijective_hungarian_algorithm',
                    'mean_assignment_distance': float(np.mean(assignment_distances))
                }
            }
            
            try:
                with open(correspondence_file, 'w') as f:
                    json.dump(correspondence_data, f, indent=2)
            except Exception as e:
                print(f"    Warning: Failed to save correspondences: {e}")

    def update_adaptive_weights(self, V_current, iteration, max_iterations):
        """
        Update correspondence weights adaptively during optimization based on
        how well each correspondence is being satisfied
        """
        if not self.filter_correspondences:
            return
        
        
        # Compute current distance between each source vertex and its target
        distances = []
        for i, source_idx in enumerate(self.correspondence_indices):
            current_point = V_current[source_idx]
            target_point = self.target_correspondence_points[i]
            
            # Compute Euclidean distance
            dist = np.linalg.norm(current_point - target_point)
            distances.append(dist)
        
        # Convert to numpy array
        distances = np.array(distances)
        
        # Get statistics for scaling
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))  # Median absolute deviation
        
        # Normalize distances with scaling
        if mad > 1e-6:
            normalized_distances = (distances - median_dist) / (1.4826 * mad)
        else:
            normalized_distances = distances / (median_dist + 1e-6)
        
        # Analyze local neighborhood consistency for each correspondence
        # This helps identify points that are locally inconsistent with their neighborhood
        neighborhood_consistency = np.ones(len(self.correspondence_indices))
        
        # Create KD-tree for the current deformed vertices (with caching)
        current_vertices_hash = hash(V_current.tobytes())
        current_tree = resource_manager.get_kdtree(V_current, f"current_{current_vertices_hash}")
        
        # Check each correspondence's consistency with its local neighborhood
        k_neighbors = min(20, len(V_current) // 100)
        
        for i, source_idx in enumerate(self.correspondence_indices):
            # Find k nearest neighbors in the current mesh
            dists, neigh_indices = current_tree.query(V_current[source_idx], k=k_neighbors+1)
            neigh_indices = neigh_indices[1:]  # Skip the point itself
            
            # Check which of these neighbors are correspondence points
            valid_neighbors = []
            
            for j, n_idx in enumerate(neigh_indices):
                if n_idx in self.correspondence_indices:
                    valid_neighbors.append(n_idx)
            
            # If enough valid neighbors, compute neighborhood consistency
            if len(valid_neighbors) >= 3:
                neighbor_distances = []
                for n_idx in valid_neighbors:
                    # Get its target correspondence
                    n_target_idx = self.correspondences[n_idx]
                    
                    # Compute distance between targets
                    source_dist = np.linalg.norm(V_current[source_idx] - V_current[n_idx])
                    target_dist = np.linalg.norm(
                        self.target_mesh.vertices[self.correspondences[source_idx]] - 
                        self.target_mesh.vertices[n_target_idx]
                    )
                    
                    # Store distance ratio
                    if source_dist > 1e-6 and target_dist > 1e-6:
                        ratio = min(source_dist/target_dist, target_dist/source_dist)
                        neighbor_distances.append(ratio)
                
                if neighbor_distances:
                    # Calculate neighborhood consistency as average of distance ratios
                    neighborhood_consistency[i] = min(1.0, sum(neighbor_distances) / len(neighbor_distances))
                else:
                    neighborhood_consistency[i] = 0.5  # Neutral score if no valid neighbors
        
        # Update weights based on how well correspondence is being satisfied
        # Use Tukey's biweight function for downweighting with added neighborhood consistency
        c = 4.685  # Tukey's constant
        
        for i, source_idx in enumerate(self.correspondence_indices):
            # Use uniform weight
            orig_confidence = 1.0
            
            # Compute weight based on normalized distance
            nd = normalized_distances[i]
            if abs(nd) <= c:
                # Tukey weight: (1 - (r/c)²)²
                weight_scale = (1 - (nd/c)**2)**2
            else:
                # Zero weight for outliers
                weight_scale = 0.0
            
            # Factor in neighborhood consistency
            weight_scale *= neighborhood_consistency[i]
            
            # Update weight (blend original confidence with adaptive weight)
            # Early iterations: trust original confidence more
            # Later iterations: trust adaptive weights more
            if max_iterations == 1:
                blend_factor = 1.0  # Use full adaptive weight for single iteration
            else:
                blend_factor = min(1.0, iteration / (max_iterations * 0.5))
            new_weight = (1 - blend_factor) * orig_confidence + blend_factor * weight_scale
            
            # Apply additional penalization to questionable correspondences (points far from their targets)
            # This penalty increases with iterations to progressively filter out bad correspondences
            if distances[i] > 3.0 * median_dist:
                # Gradually increase penalty for outliers
                if max_iterations == 1:
                    penalty = 0.9  # Use maximum penalty for single iteration
                else:
                    penalty = min(0.9, 0.5 + (0.4 * iteration / max_iterations))
                new_weight *= (1.0 - penalty)
            
            # Store updated weight
            self.update_correspondence_weights[source_idx] = new_weight

    def local_step(self, V_current):
        """
        Local step: Compute optimal rotations for each vertex using GPU acceleration.
        Uses pre-allocated GPU buffers and batching.
        """
        n_vertices = len(V_current)
        rotations = np.zeros((n_vertices, 3, 3))
        
        # Move data to GPU (use pre-allocated buffers)
        vertices_gpu = resource_manager.get_gpu_buffer(n_vertices)
        vertices_gpu[:] = torch.tensor(self.aligned_vertices, dtype=torch.float32, device=device)
        
        current_gpu = resource_manager.get_gpu_buffer(n_vertices)
        current_gpu[:] = torch.tensor(V_current, dtype=torch.float32, device=device)
        
        # Process vertices in batches
        alignment_config = self.config_manager.get_alignment_config()
        batch_size = min(alignment_config['optimization']['batch_size'], n_vertices)
        
        for batch_start in range(0, n_vertices, batch_size):
            batch_end = min(batch_start + batch_size, n_vertices)
            batch_indices = list(range(batch_start, batch_end))
            
            for i in batch_indices:
                # Skip if no neighbors
                if not self.vertex_neighbors[i]:
                    rotations[i] = np.eye(3)
                    continue
                
                # Compute covariance matrix
                S = torch.zeros((3, 3), dtype=torch.float32, device=device)
                for j in self.vertex_neighbors[i]:
                    w_ij = self.cotangent_weights.get((i, j), 0)
                    d_ij = vertices_gpu[i] - vertices_gpu[j]  # Original difference
                    d_ij_prime = current_gpu[i] - current_gpu[j]  # Current difference
                    
                    # Outer product on GPU
                    S += w_ij * torch.outer(d_ij, d_ij_prime)
                
                # Move to CPU for SVD
                S_cpu = S.cpu().numpy()
                
                # SVD decomposition
                U, _, Vt = svd(S_cpu)
                
                # Compute rotation (with determinant check)
                R = Vt.T @ U.T
                
                # Ensure no reflection
                if det(R) < 0:
                    U[:, 2] *= -1
                    R = Vt.T @ U.T
                
                rotations[i] = R
        
        return rotations


    def global_step(self, rotations, iteration=0, max_iterations=None):
        """
        Global step: Solve for new vertex positions with GPU pre-computation.
        Uses cached KDTrees and pre-allocated GPU buffers.
        
        The total energy function is:
        
        E_total = arap_weight * E_ARAP + 
                  correspondence_weight * E_correspondence + 
                  smoothness_weight * E_smoothness
        """
        # Get max_iterations from config if not provided
        if max_iterations is None:
            alignment_config = self.config_manager.get_alignment_config()
            max_iterations = alignment_config['optimization']['global_max_iterations']
            
        n_vertices = len(self.aligned_vertices)
        
        # Create a copy of the Laplacian for modification
        L_prime = self.L.copy()
        
        # Right-hand side of the equation (b vector)
        b = np.zeros((n_vertices, 3))
        
        # Move aligned vertices to GPU (use pre-allocated buffer)
        vertices_gpu = resource_manager.get_gpu_buffer(n_vertices)
        vertices_gpu[:] = torch.tensor(self.aligned_vertices, dtype=torch.float32, device=device)
        
        # Compute the right-hand side vector
        # ARAP term with explicit weight
        arap_weight = self.arap_weight
        
        alignment_config = self.config_manager.get_alignment_config()
        batch_size = min(alignment_config['optimization']['batch_size'], n_vertices)
        for batch_start in range(0, n_vertices, batch_size):
            batch_end = min(batch_start + batch_size, n_vertices)
            batch_indices = list(range(batch_start, batch_end))
            
            for i in batch_indices:
                for j in self.vertex_neighbors[i]:
                    w_ij = self.cotangent_weights.get((i, j), 0)
                    d_ij = (vertices_gpu[i] - vertices_gpu[j]).cpu().numpy()
                    R_avg = 0.5 * (rotations[i] + rotations[j])
                    # Apply ARAP weight to the right-hand side
                    b[i] += arap_weight * w_ij * (R_avg @ d_ij)
        
        # Add correspondence constraints with adaptive weighting
        # Check if correspondence term should be applied based on iteration
        if self.correspondence_end_iteration < 0 or iteration <= self.correspondence_end_iteration:
            # Progressive correspondence weight scheme
            # Gradually increase influence of correspondences with iterations
            if max_iterations == 1:
                correspondence_scale = 1.0  # Use full weight for single iteration
            else:
                correspondence_scale = min(1.0, (iteration + 1) / (max_iterations * 0.3))
            adjusted_lambda = self.correspondence_weight * correspondence_scale
            
            for idx, source_idx in enumerate(self.correspondence_indices):
                # Use adaptive weights if filtering is enabled
                if self.filter_correspondences and iteration > 0:
                    # Apply both adaptive weighting AND uncertainty-based confidence
                    confidence = self.update_correspondence_weights.get(source_idx, self.correspondence_weights[idx])
                    
                    # Apply sigmoid-like function to downweight less confident correspondences more aggressively
                    if confidence < 0.8:
                        confidence = confidence * 0.5  # Stronger penalty for less confident points
                    
                    weight = adjusted_lambda * confidence
                else:
                    weight = adjusted_lambda * self.correspondence_weights[idx]
                
                L_prime[source_idx, source_idx] += weight
                b[source_idx] += weight * self.target_correspondence_points[idx]
        else:
            print(f"Correspondence term disabled (iteration {iteration} > end_iteration {self.correspondence_end_iteration})")
            
        # Add smoothness energy term
        if self.smoothness_weight > 0:
            # Add Laplacian smoothness term to the linear system
            # This penalizes large differences between neighboring vertices
            L_prime += self.smoothness_weight * self.L
        
        # Solve the linear system for each coordinate
        V_new = np.zeros((n_vertices, 3))
        for dim in range(3):
            V_new[:, dim] = spsolve(L_prime, b[:, dim])
        
        # Store current vertices for next iteration's regularization
        self.prev_vertices = V_new.copy()
        
        return V_new

    def compute_energy(self, vertices, iteration=0, max_iterations=None):
        """
        Compute the total energy and individual energy components
        
        Returns:
            dict: Dictionary containing total energy and individual components
        """
        # Get max_iterations from config if not provided
        if max_iterations is None:
            alignment_config = self.config_manager.get_alignment_config()
            max_iterations = alignment_config['optimization']['global_max_iterations']
            
        energy_dict = {}
        
        # 1. ARAP Energy
        arap_energy = 0.0
        for i in range(len(vertices)):
            for j in self.vertex_neighbors[i]:
                w_ij = self.cotangent_weights.get((i, j), 0)
                
                # Original edge vector
                d_ij_orig = self.aligned_vertices[i] - self.aligned_vertices[j]
                # Current edge vector
                d_ij_curr = vertices[i] - vertices[j]
                
                # Compute optimal rotation for this edge
                # For energy computation, we use identity rotation as approximation
                # to avoid recomputing all rotations
                diff = d_ij_curr - d_ij_orig
                arap_energy += 0.5 * w_ij * np.dot(diff, diff)
        
        energy_dict['arap'] = self.arap_weight * arap_energy
        
        # 2. Correspondence Energy
        correspondence_energy = 0.0
        if self.correspondence_end_iteration < 0 or iteration <= self.correspondence_end_iteration:
            if max_iterations == 1:
                correspondence_scale = 1.0  # Use full weight for single iteration
            else:
                correspondence_scale = min(1.0, (iteration + 1) / (max_iterations * 0.3))
            adjusted_lambda = self.correspondence_weight * correspondence_scale
            
            for idx, source_idx in enumerate(self.correspondence_indices):
                # Use adaptive weights if filtering is enabled
                if self.filter_correspondences and iteration > 0:
                    confidence = self.update_correspondence_weights.get(source_idx, self.correspondence_weights[idx])
                    if confidence < 0.8:
                        confidence = confidence * 0.5
                    weight = adjusted_lambda * confidence
                else:
                    weight = adjusted_lambda * self.correspondence_weights[idx]
                
                diff = vertices[source_idx] - self.target_correspondence_points[idx]
                correspondence_energy += weight * np.dot(diff, diff)
        
        energy_dict['correspondence'] = correspondence_energy
        
        # 3. Smoothness Energy
        smoothness_energy = 0.0
        if self.smoothness_weight > 0:
            # Compute smoothness energy as quadratic form: 0.5 * v^T * L * v
            # This measures how much the mesh deviates from being smooth
            for i in range(len(vertices)):
                laplacian_i = 0.0
                for j in self.vertex_neighbors[i]:
                    w_ij = self.cotangent_weights.get((i, j), 0)
                    laplacian_i += w_ij * (vertices[i] - vertices[j])
                
                # Add contribution to smoothness energy
                smoothness_energy += 0.5 * np.dot(vertices[i], laplacian_i)
        
        energy_dict['smoothness'] = self.smoothness_weight * smoothness_energy
        
        # Total energy
        energy_dict['total'] = energy_dict['arap'] + energy_dict['correspondence'] + energy_dict['smoothness']
        
        return energy_dict

    def optimize(self, max_iterations=None, convergence_tol=None):
        """
        Perform ARAP optimization with the given parameters
        """
        # Use parameters from config if not provided
        if max_iterations is None:
            max_iterations = self.config['optimization']['max_iterations']
        if convergence_tol is None:
            convergence_tol = self.config['optimization']['convergence_tolerance']
        
        # Get save frequency from config
        save_frequency = self.config['output'].get('save_frequency', 10)
        
        
        # Initialize with aligned vertices
        V_current = self.aligned_vertices.copy()
        
        start_time = time.time()
        
        # Main optimization loop
        from tqdm import tqdm
        
        optimization_progress = tqdm(total=max_iterations, desc="ARAP optimization", unit="iter", leave=True)
        
        for iteration in range(max_iterations):
            
            # Update correspondences using bijective method (always enabled)
            if (iteration >= self.correspondence_update_start_iteration and
                iteration % self.correspondence_update_frequency == 0):
                self.update_correspondences_bijective(V_current, iteration, max_iterations)
            
            # Update correspondence weights adaptively - do this every iteration now
            if iteration > 0 and self.filter_correspondences:
                self.update_adaptive_weights(V_current, iteration, max_iterations)
            
            # Local step: compute optimal rotations
            rotations = self.local_step(V_current)
            
            # Global step: solve for new positions
            V_new = self.global_step(rotations, iteration, max_iterations)
            
            # Smoothness is now handled as an energy term in the optimization
            # No need for post-processing smoothing
            
            # Compute and display energy information
            energy_info = self.compute_energy(V_new, iteration, max_iterations)
            
            
            # Compute error for convergence check
            error = np.max(np.linalg.norm(V_new - V_current, axis=1))
            
            
            # Update progress bar with energy and displacement info
            optimization_progress.set_postfix({
                'Energy': f"{energy_info['total']:.3f}", 
                'Displacement': f"{error:.4f}"
            })
            optimization_progress.update(1)
            
            # Check for convergence
            if error < convergence_tol:
                break
            
            # Update current vertices
            V_current = V_new
            
            # Save intermediate result based on save_frequency
            if iteration % save_frequency == 0 or iteration == max_iterations - 1:
                intermediate_mesh = trimesh.Trimesh(
                    vertices=V_current,
                    faces=self.source_mesh.faces,
                    process=False
                )
                intermediate_path = os.path.join(self.log_dir, f'iteration_{iteration:03d}.obj')
                intermediate_mesh.export(intermediate_path)
        
        optimization_progress.close()
        
        elapsed_time = time.time() - start_time
        
        # Final alignment step using nearest neighbor matching
        V_final_aligned = self.final_nearest_neighbor_alignment(V_current)
        
        # Store the final result
        self.deformed_vertices = V_current
        self.final_aligned_vertices = V_final_aligned
        
        # Save the final mesh with the expected filename
        if hasattr(self, 'config') and 'output' in self.config and 'final_mesh' in self.config['output']:
            final_output_path = self.config['output']['final_mesh']
            final_mesh = trimesh.Trimesh(
                vertices=V_current,
                faces=self.source_mesh.faces,
                process=False
            )
            final_mesh.export(final_output_path)
        
        return V_current
    
    def final_nearest_neighbor_alignment(self, vertices):
        """
        Perform final alignment using nearest neighbor matching to target mesh.
        This step directly moves each vertex to its nearest neighbor on the target mesh.
        
        Args:
            vertices: Current vertex positions
            
        Returns:
            Final aligned vertices
        """
        
        # Build KD-tree for target mesh vertices (with caching)
        target_mesh_hash = hash(self.target_mesh.vertices.tobytes())
        target_kdtree = resource_manager.get_kdtree(self.target_mesh.vertices, f"target_final_{target_mesh_hash}")
        
        # Find nearest neighbor for each vertex
        distances, indices = target_kdtree.query(vertices, k=1)
        
        # Get the nearest target vertices
        final_aligned_vertices = self.target_mesh.vertices[indices]
        
        # Compute alignment statistics
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        
        
        # Save the final aligned mesh
        final_aligned_mesh = trimesh.Trimesh(
            vertices=final_aligned_vertices,
            faces=self.source_mesh.faces,
            process=False
        )
        
        # Use stored source model name
        model_name = self.source_model_name
        
        final_output_path = os.path.join(self.log_dir, f"{model_name}_final_align.obj")
        final_aligned_mesh.export(final_output_path)
        
        
        return final_aligned_vertices

def process_single_pair_alignment(models_dir, pair_dir, source_model, target_model, config):
    """
    Process a single pair directly without CSV file.
    
    Args:
        models_dir: Path to the models directory containing all normalized meshes and features
        pair_dir: Path to the pair directory where results will be stored
        source_model: Source model name (without .obj extension)
        target_model: Target model name (without .obj extension)
        config: Configuration object
        
    Returns:
        1 if successful, 0 if failed
    """
    try:
        # Get alignment config from config
        base_config = config.get_alignment_config()
        
        pair_id = 1
        
        print(f"Processing pair {pair_id}: {source_model} -> {target_model}")
        
        # Create or find pair directory
        pair_folder = pair_dir
        os.makedirs(pair_folder, exist_ok=True)
        
        # Process the pair using shared base config
        success = process_single_pair(pair_id, source_model, target_model, pair_folder, 
                                    base_config, models_dir)
        
        if success:
            print(f"Successfully processed pair: {source_model} -> {target_model}")
            return 1
        else:
            raise ValueError("DINO + ARAP pipeline failed")
        
    except Exception as e:
        print(f"Error processing pair {source_model} -> {target_model}: {e}")
        import traceback
        traceback.print_exc()
        return 0
        

# This module is now used only as a library by correspondence.py
# The main() function has been removed since this script is no longer called directly
