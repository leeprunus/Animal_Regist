"""
Mesh normalization functions.

This module contains functions for normalizing meshes for alignment pipeline.
The normalization uses keypoint-based alignment with procrustes alignment.
"""

import numpy as np
import open3d as o3d
import copy
import os
import glob
from pathlib import Path
from tqdm import tqdm
from src.utils.cache import get_cache_manager


def rotation_matrix_from_vectors(vec1, vec2):
    """Compute rotation matrix that rotates vec1 to vec2"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    cross = np.cross(a, b)
    dot = np.dot(a, b)
    if np.linalg.norm(cross) < 1e-8:
        return np.eye(3, dtype=np.float64)
    skew = np.array([[0, -cross[2], cross[1]],
                     [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross)**2))
    return R


def normalize_keypoints(keypoints):
    """
    Normalize keypoints:
    - Keypoint 3 (index 3, 0-based) as origin
    - Distance from keypoint 3 to keypoint 4 (index 4) for scaling
    - Rotate so keypoint 4 (index 4) aligns to [1, 0, 0]
    """
    keypoints = keypoints.astype(np.float64)
    origin = keypoints[3]
    kp_translated = keypoints - origin
    v = kp_translated[4]
    scale = np.linalg.norm(v)
    if scale == 0:
        raise ValueError("Keypoints 3 and 4 are at the same position, cannot normalize.")
    kp_scaled = kp_translated / scale
    desired_direction = np.array([1.0, 0, 0], dtype=np.float64)
    current_direction = kp_scaled[4]
    R = rotation_matrix_from_vectors(current_direction, desired_direction)
    kp_normalized = (R @ kp_scaled.T).T
    return kp_normalized, R, scale, origin


def refine_alignment_with_additional_keypoints(keypoints, source_keypoints, primary_indices=[3, 4]):
    """
    Refine alignment using additional keypoints after primary alignment.
    This helps achieve overall anatomical alignment.
    """
    keypoints = keypoints.astype(np.float64)
    source_keypoints = source_keypoints.astype(np.float64)
    
    # Define keypoint groups for different anatomical regions
    head_indices = [0, 1, 2]  # Nose, eyes, ears
    front_leg_indices = [5, 6, 7, 8]  # Front legs
    back_leg_indices = [9, 10, 11, 12]  # Back legs
    body_indices = [13, 14, 15, 16]  # Body/torso points
    spine_indices = [17, 18] if len(keypoints) > 18 else []  # Spine points if available
    
    # Combine all secondary keypoints (excluding already aligned ones)
    secondary_indices = []
    for group in [head_indices, front_leg_indices, back_leg_indices, body_indices, spine_indices]:
        for idx in group:
            if idx < len(keypoints) and idx not in primary_indices:
                secondary_indices.append(idx)
    
    if len(secondary_indices) < 3:
        print("    Not enough secondary keypoints for refinement. Skipping additional alignment.")
        return keypoints, np.eye(3, dtype=np.float64)
    
    # Extract secondary keypoints
    source_secondary = keypoints[secondary_indices]
    target_secondary = source_keypoints[secondary_indices]
    
    # Compute refinement rotation using secondary keypoints
    R_refine, t_refine = procrustes_alignment(source_secondary, target_secondary)
    
    # Apply only the rotation part
    refined_keypoints = (R_refine @ keypoints.T).T
    
    # Verify that primary keypoints are still properly aligned
    primary_error = np.linalg.norm(refined_keypoints[primary_indices] - source_keypoints[primary_indices])
    
    if primary_error > 0.1:  # If primary alignment is significantly disturbed
        print(f"    Warning: Refinement disturbed primary alignment (error: {primary_error:.6f}). Using original alignment.")
        return keypoints, np.eye(3, dtype=np.float64)
    
    # Calculate improvement in secondary keypoint alignment
    original_error = np.mean(np.linalg.norm(keypoints[secondary_indices] - source_keypoints[secondary_indices], axis=1))
    refined_error = np.mean(np.linalg.norm(refined_keypoints[secondary_indices] - source_keypoints[secondary_indices], axis=1))
    
    
    return refined_keypoints, R_refine


def normalize_vertices(vertices, R, scale, origin):
    """Apply the same normalization to vertices as was applied to keypoints"""
    return (R @ ((vertices - origin).T / scale)).T


def procrustes_alignment(source_points, target_points):
    """Compute rigid transformation (R, t) to align source to target points"""
    # Convert to float64 for numerical stability
    source_points = np.array(source_points, dtype=np.float64)
    target_points = np.array(target_points, dtype=np.float64)
    
    # Check for NaN or infinite values
    if not (np.isfinite(source_points).all() and np.isfinite(target_points).all()):
        print("Warning: Non-finite values detected in keypoints. Using identity transformation.")
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    
    # Check if we have enough points
    if source_points.shape[0] < 3 or target_points.shape[0] < 3:
        print("Warning: Not enough points for alignment. Using identity transformation.")
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    
    # Check for degenerate cases
    if np.allclose(source_centered, 0) or np.allclose(target_centered, 0):
        print("Warning: Degenerate point configuration. Using identity rotation.")
        t = centroid_target - centroid_source
        return np.eye(3, dtype=np.float64), t
    
    H = source_centered.T @ target_centered
    
    # Try SVD with error handling
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError as e:
        print(f"Warning: SVD failed ({e}). Trying with regularization...")
        H_reg = H + np.eye(H.shape[0]) * 1e-8
        try:
            U, S, Vt = np.linalg.svd(H_reg)
        except np.linalg.LinAlgError:
            print("Warning: SVD still failed. Using identity transformation.")
            t = centroid_target - centroid_source
            return np.eye(3, dtype=np.float64), t
    
    R = Vt.T @ U.T
    
    # Ensure proper rotation matrix (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Verify that R is a valid rotation matrix
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
        print("Warning: Invalid rotation matrix computed. Using identity transformation.")
        t = centroid_target - centroid_source
        return np.eye(3, dtype=np.float64), t
    
    t = centroid_target - R @ centroid_source
    return R, t


def load_obj_and_keypoints(obj_file, cache_manager=None):
    """
    Load an OBJ file and its corresponding keypoints from cache.
    """
    try:
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(obj_file)
        
        if not mesh.has_triangles():
            print(f"Warning: {obj_file} has no triangles.")
            return None, None
        
        # Load keypoints from cache
        if cache_manager is not None:
            keypoints_data = cache_manager.load_keypoints(obj_file)
            if keypoints_data is None:
                print(f"Warning: Keypoints for {obj_file} not found in cache.")
                return None, None
            
            keypoints = keypoints_data[0]  # keypoints_data is (keypoints, metadata)
        else:
            # Direct file access if cache manager not provided
            keypoints_file = obj_file.replace('.obj', '.npy')
            if not os.path.exists(keypoints_file):
                print(f"Warning: Keypoints file {keypoints_file} not found.")
                return None, None
            keypoints = np.load(keypoints_file)
            
        return mesh, keypoints
        
    except Exception as e:
        print(f"Error loading {obj_file} or its keypoints: {e}")
        return None, None


def align_to_source(mesh, keypoints, source_mesh, source_keypoints):
    """
    Align a model directly to the source using Procrustes alignment.
    """
    # Compute rigid alignment between keypoints
    R_rigid, t_rigid = procrustes_alignment(keypoints, source_keypoints)
    
    # Apply alignment to mesh vertices
    vertices = np.asarray(mesh.vertices)
    aligned_vertices = (R_rigid @ vertices.T).T + t_rigid
    
    # Apply alignment to keypoints
    aligned_keypoints = (R_rigid @ keypoints.T).T + t_rigid
    
    # Create aligned mesh
    aligned_mesh = copy.deepcopy(mesh)
    aligned_mesh.vertices = o3d.utility.Vector3dVector(aligned_vertices)
    
    return aligned_mesh, aligned_keypoints


def find_mesh_files(models_dir, extensions=('.obj',)):
    """
    Find all mesh files in the given directory and its subdirectories
    """
    mesh_files = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise ValueError(f"Models directory {models_dir} does not exist")
    
    for ext in extensions:
        # Use rglob to search recursively through all subdirectories
        mesh_files.extend(list(models_path.rglob(f"*{ext}")))
    
    return sorted(mesh_files)


def check_keypoint_alignment(keypoints_list, model_names):
    """
    Check alignment quality for all keypoints across models
    """
    if len(keypoints_list) < 2:
        return
    
    print(f"\n===== Keypoint Alignment Analysis =====")
    
    source_keypoints = keypoints_list[0]
    num_keypoints = source_keypoints.shape[0]
    
    # Calculate alignment errors for each keypoint
    keypoint_errors = []
    
    for kp_idx in range(num_keypoints):
        errors_for_this_kp = []
        source_kp = source_keypoints[kp_idx]
        
        for i, keypoints in enumerate(keypoints_list[1:]):
            if kp_idx < keypoints.shape[0]:
                error = np.linalg.norm(keypoints[kp_idx] - source_kp)
                errors_for_this_kp.append(error)
        
        if errors_for_this_kp:
            avg_error = np.mean(errors_for_this_kp)
            max_error = np.max(errors_for_this_kp)
            keypoint_errors.append((kp_idx, avg_error, max_error))
    
    # Sort by average error (worst first)
    keypoint_errors.sort(key=lambda x: x[1], reverse=True)
    
    print("Keypoint alignment quality (sorted by average error):")
    print("Keypoint | Avg Error | Max Error | Quality")
    print("-" * 45)
    
    for kp_idx, avg_err, max_err in keypoint_errors:
        if avg_err < 0.01:
            quality = "Excellent"
        elif avg_err < 0.05:
            quality = "Good"
        elif avg_err < 0.1:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"   {kp_idx:2d}    | {avg_err:8.6f} | {max_err:8.6f} | {quality}")
    
    # Summary statistics
    all_avg_errors = [err[1] for err in keypoint_errors]
    overall_avg = np.mean(all_avg_errors)
    well_aligned_count = sum(1 for err in all_avg_errors if err < 0.05)
    
    print("-" * 45)
    print(f"Overall average error: {overall_avg:.6f}")
    print(f"Well-aligned keypoints (error < 0.05): {well_aligned_count}/{len(keypoint_errors)}")
    print(f"Alignment success rate: {(well_aligned_count / len(keypoint_errors) * 100):.1f}%")
    print("=" * 50)


def process_all_meshes(models_dir, extensions=('.obj',), backup_originals=False, cache_manager=None):
    """
    Process all meshes in the models directory:
    1. Find all mesh files
    2. Use first mesh as source
    3. Normalize all meshes to the source
    4. Save normalized meshes to cache
    """
    # Find all mesh files
    try:
        mesh_files = find_mesh_files(models_dir, extensions)
    except ValueError as e:
        print(f"Error: {e}")
        return 0
    
    if not mesh_files:
        print(f"No mesh files found in {models_dir} with extensions {extensions}")
        return 0
    
    
    # Load all meshes and keypoints
    meshes = []
    keypoints_list = []
    model_names = []
    valid_files = []
    
    for i, mesh_file in enumerate(mesh_files):
        # Get relative path for display
        try:
            relative_path = mesh_file.relative_to(Path(models_dir))
            display_name = str(relative_path.with_suffix(''))  # Remove extension
        except ValueError:
            display_name = mesh_file.stem
        
        
        mesh, keypoints = load_obj_and_keypoints(str(mesh_file), cache_manager)
        if mesh is not None and keypoints is not None:
            meshes.append(mesh)
            keypoints_list.append(keypoints)
            model_names.append(display_name)
            valid_files.append(mesh_file)
        else:
            pass
    
    if len(meshes) == 0:
        print("No valid meshes found!")
        return 0
    
    
    # Use first mesh as source
    source_mesh = meshes[0]
    source_keypoints = keypoints_list[0]
    source_name = model_names[0]
    
    # Align all models to source
    aligned_meshes = [source_mesh]  # Source remains unchanged
    aligned_keypoints = [source_keypoints]
    
    for i, (mesh, keypoints, name) in enumerate(zip(meshes[1:], keypoints_list[1:], model_names[1:]), 1):
        try:
            aligned_mesh, aligned_kp = align_to_source(mesh, keypoints, source_mesh, source_keypoints)
            aligned_meshes.append(aligned_mesh)
            aligned_keypoints.append(aligned_kp)
        except Exception as e:
            # Use original mesh/keypoints if alignment fails
            aligned_meshes.append(mesh)
            aligned_keypoints.append(keypoints)
    
    # Normalize all aligned models
    normalized_meshes = []
    normalized_keypoints = []
    
    # First normalize the source to establish reference
    try:
        source_normalized_kp, source_R, source_scale, source_origin = normalize_keypoints(source_keypoints)
        source_normalized_vertices = normalize_vertices(np.asarray(source_mesh.vertices), source_R, source_scale, source_origin)
        
        source_normalized_mesh = copy.deepcopy(source_mesh)
        source_normalized_mesh.vertices = o3d.utility.Vector3dVector(source_normalized_vertices)
        
        normalized_meshes.append(source_normalized_mesh)
        normalized_keypoints.append(source_normalized_kp)
    except Exception as e:
        normalized_meshes.append(source_mesh)
        normalized_keypoints.append(source_keypoints)
    
    # Normalize all other aligned models
    for i, (mesh, keypoints, name) in enumerate(zip(aligned_meshes[1:], aligned_keypoints[1:], model_names[1:]), 1):
        
        try:
            # Apply primary normalization
            normalized_kp, R_primary, scale, origin = normalize_keypoints(keypoints)
            
            # Refine alignment using additional keypoints
            refined_kp, R_refine = refine_alignment_with_additional_keypoints(
                normalized_kp, source_normalized_kp
            )
            
            # Apply combined transformation to mesh vertices
            vertices = np.asarray(mesh.vertices)
            vertices_primary = normalize_vertices(vertices, R_primary, scale, origin)
            vertices_refined = (R_refine @ vertices_primary.T).T
            
            # Create final normalized mesh
            normalized_mesh = copy.deepcopy(mesh)
            normalized_mesh.vertices = o3d.utility.Vector3dVector(vertices_refined)
            
            normalized_meshes.append(normalized_mesh)
            normalized_keypoints.append(refined_kp)
            
        except Exception as e:
            # Use original mesh/keypoints if normalization fails
            normalized_meshes.append(mesh)
            normalized_keypoints.append(keypoints)
    
    # Cache normalized meshes
    successful_saves = 0
    
    # Use provided cache manager or initialize one if not provided
    if cache_manager is None:
        cache_manager = get_cache_manager("./cache")
    
    
    for i, (mesh, keypoints, name, mesh_file) in enumerate(zip(normalized_meshes, normalized_keypoints, model_names, valid_files)):
        
        try:
            # Save normalized mesh to cache
            normalized_mesh_path = cache_manager.save_normalized_mesh(mesh_file, mesh)
            
            # Update keypoints in cache with normalized positions 
            # First, try to load existing keypoint info
            existing_keypoints = cache_manager.load_keypoints(mesh_file)
            if existing_keypoints:
                _, existing_info = existing_keypoints
                # Update the info with normalization data
                existing_info.update({
                    'normalized': True,
                    'normalization_applied': True,
                    'source_file': str(mesh_file)
                })
                info = existing_info
            else:
                # Create new info if none exists
                info = {
                    'num_keypoints': len(keypoints),
                    'normalized': True,
                    'normalization_applied': True,
                    'source_file': str(mesh_file),
                    'model_name': name
                }
            
            # Cache the normalized keypoints
            cache_manager.cache_keypoints(mesh_file, keypoints, info)
            
            successful_saves += 1
            
        except Exception as e:
            pass
    
    
    return successful_saves
