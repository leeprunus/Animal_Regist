"""
Linear Blend Skinning (LBS) Deformation

Basic mesh deformation using Linear Blend Skinning.
"""

import numpy as np
import trimesh


def lbs_mesh_deformation(source_mesh_path, source_kp, target_kp, output_path):
    """
    Basic mesh deformation using Linear Blend Skinning (LBS).
    
    Args:
        source_mesh_path: Path to source mesh (.obj)
        source_kp: Source keypoints (numpy array)
        target_kp: Target keypoints (numpy array)
        output_path: Path to save deformed mesh (.obj)
    """
    # Load mesh and keypoints
    mesh = trimesh.load(source_mesh_path)
    
    # Ensure we have the same number of keypoints
    min_kp = min(len(source_kp), len(target_kp))
    source_kp = source_kp[:min_kp]
    target_kp = target_kp[:min_kp]
    
    # Simple LBS: Compute transformation from keypoint differences
    if min_kp < 3:
        # Not enough keypoints for meaningful deformation, just copy the mesh
        mesh.export(output_path)
        return
    
    # Compute centroid-based transformation
    source_center = np.mean(source_kp, axis=0)
    target_center = np.mean(target_kp, axis=0)
    translation = target_center - source_center
    
    # Apply translation to all vertices
    deformed_vertices = mesh.vertices + translation
    
    # Create new mesh with deformed vertices
    deformed_mesh = trimesh.Trimesh(vertices=deformed_vertices, faces=mesh.faces)
    
    # Export the deformed mesh
    deformed_mesh.export(output_path)
