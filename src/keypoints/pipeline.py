#!/usr/bin/env python3
"""
End-to-End 3D Keypoints Pipeline

This script combines the functionality of hks.py, predict2d.py, and predict3d.py 
into a single end-to-end pipeline that takes a 3D model as input and produces 
3D keypoints as output.

Usage:
    python keypoints_pipeline.py --input path/to/model.obj --output path/to/keypoints.npy
"""

import os
import sys
import json
import math
import copy
import shutil
import tempfile
import argparse
import glob
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import cv2
import torch
import torch.sparse as torch_sparse
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
from tqdm import tqdm
import warnings
import logging

# Warning suppression
warnings.filterwarnings('ignore')
logging.getLogger('mmengine').setLevel(logging.ERROR)  
logging.getLogger('mmpose').setLevel(logging.ERROR)
logging.getLogger('mmdet').setLevel(logging.ERROR)
logging.getLogger('mmcv').setLevel(logging.ERROR)

# Import MMPose for 2D keypoint detection
from mmpose.apis.inferencers import MMPoseInferencer


class KeypointsPipeline:
    """End-to-end pipeline for 3D keypoint detection from 3D models"""
    
    def __init__(self, temp_dir=None, device=None, num_views=16):
        """
        Initialize the pipeline
        
        Args:
            temp_dir: Temporary directory for intermediate files (None for auto)
            device: PyTorch device ('cuda', 'cpu', or None for auto-detection)
            num_views: Number of camera views for multi-view rendering
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix='keypoints_pipeline_')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.num_views = num_views
        
        
        # Create subdirectories
        self.render_dir = os.path.join(self.temp_dir, 'renders')
        self.vis_dir = os.path.join(self.temp_dir, 'vis_results')
        self.pred_dir = os.path.join(self.temp_dir, 'pred_results')
        
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
    
    
    def detect_tips(self, mesh_file, num_eigenvalues=80, time_samples=40, 
                   curvature_threshold=0.8, hks_threshold=0.0, min_distance_ratio=0.1,
                   max_vertices=50000):
        """
        Step 1: Detect tips using Heat Kernel Signature with fallback
        
        Args:
            mesh_file: Path to input mesh file
            num_eigenvalues: Number of eigenvalues to compute
            time_samples: Number of time scales for HKS
            curvature_threshold: Curvature percentile threshold (0-1, lower is more permissive)
            hks_threshold: HKS percentile threshold (0-1)
            min_distance_ratio: Minimum distance between tips as ratio of bounding box diagonal
            max_vertices: Maximum number of vertices (mesh will be decimated if larger)
            
        Returns:
            tips_positions: numpy array of tip positions (num_tips, 3)
        """
        
        # Load mesh
        mesh = trimesh.load(mesh_file)
        
        if not hasattr(mesh, 'vertices'):
            raise ValueError("Could not load mesh or mesh has no vertices")
        
        # Simplify mesh if it's too large
        if len(mesh.vertices) > max_vertices:
            mesh = self._simplify_mesh(mesh, target_vertices=max_vertices)
        
        # Try HKS detection with progressively more permissive parameters
        tips_positions = None
        original_curvature_threshold = curvature_threshold
        original_min_distance_ratio = min_distance_ratio
        
        # Initialize tip detector
        from src.geometry.hks import TipDetector
        
        detector = TipDetector(
            mesh=mesh,
            num_eigenvalues=num_eigenvalues,
            time_samples=time_samples,
            device=self.device
        )
        
        # Try different parameter settings to ensure we find tips
        threshold_attempts = [
            (curvature_threshold, min_distance_ratio),  # Original parameters
            (max(0.6, curvature_threshold - 0.2), min_distance_ratio * 0.7),  # More permissive
            (max(0.4, curvature_threshold - 0.4), min_distance_ratio * 0.5),  # Even more permissive
            (0.3, min_distance_ratio * 0.3),  # Very permissive
        ]
        
        for attempt, (curv_thresh, dist_ratio) in enumerate(threshold_attempts):
            
            # Detect tips
            tips = detector.detect_tips(
                curvature_threshold=curv_thresh,
                hks_threshold=hks_threshold,
                min_distance_ratio=dist_ratio
            )
            
            if len(tips) > 0:
                # Save tips to temporary file
                tips_file = os.path.join(self.temp_dir, 'detected_tips.npy')
                tips_positions = detector.save_tips_npy(tips_file)
                break
        
        # If HKS completely fails, use geometric fallback
        if tips_positions is None or len(tips_positions) == 0:
            tips_positions = self._geometric_tip_detection(mesh, min_tips=6, max_tips=12)
        
        # Ensure we have sufficient tip coverage
        if len(tips_positions) < 4:
            print(f"\nToo few tips detected ({len(tips_positions)}). Augmenting with geometric extrema...")
            geometric_tips = self._geometric_tip_detection(mesh, min_tips=8, max_tips=10)
            
            # Combine HKS tips with geometric tips, removing duplicates
            if len(tips_positions) > 0:
                combined_tips = np.vstack([tips_positions, geometric_tips])
                # Remove duplicates based on distance
                bbox_diagonal = np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
                min_distance = bbox_diagonal * (original_min_distance_ratio * 0.5)
                tips_positions = self._remove_duplicate_tips(combined_tips, min_distance)
            else:
                tips_positions = geometric_tips
            
            print(f"Total tips after augmentation: {len(tips_positions)}")
        
        
        return tips_positions
    
    def _geometric_tip_detection(self, mesh, min_tips=6, max_tips=12):
        """
        Geometric fallback method for tip detection when HKS fails
        
        This method finds extremal points using:
        1. Vertices farthest from centroid in different directions
        2. Principal component analysis to find extrema along main axes
        3. Local curvature maxima
        
        Args:
            mesh: Trimesh object
            min_tips: Minimum number of tips to detect
            max_tips: Maximum number of tips to detect
            
        Returns:
            numpy array of tip positions
        """
        print("Using geometric method for tip detection...")
        
        vertices = mesh.vertices
        centroid = np.mean(vertices, axis=0)
        
        # Method 1: Extremal points based on distance from centroid
        distances_from_centroid = np.linalg.norm(vertices - centroid, axis=1)
        
        # Find vertices that are local maxima in distance from centroid
        # Use a simple local maximum detection
        bbox_diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        search_radius = bbox_diagonal * 0.05  # 5% of bounding box diagonal
        
        extremal_candidates = []
        
        # Get top candidates by distance from centroid
        far_indices = np.argsort(distances_from_centroid)[-min(len(vertices), max_tips * 3):]
        
        for idx in far_indices:
            vertex_pos = vertices[idx]
            vertex_dist = distances_from_centroid[idx]
            
            # Check if this vertex is a local maximum (farther than nearby vertices)
            nearby_distances = np.linalg.norm(vertices - vertex_pos, axis=1)
            nearby_mask = nearby_distances < search_radius
            nearby_indices = np.where(nearby_mask)[0]
            
            # Check if this vertex has the maximum distance from centroid among nearby vertices
            if len(nearby_indices) > 1:
                nearby_centroid_distances = distances_from_centroid[nearby_indices]
                if vertex_dist >= np.max(nearby_centroid_distances):
                    extremal_candidates.append((idx, vertex_dist))
        
        # Method 2: Principal component analysis extrema
        # Compute PCA to find main directions
        centered_vertices = vertices - centroid
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (largest first)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Find extrema along each principal component
        for i, direction in enumerate(eigenvectors.T):
            # Project vertices onto this direction
            projections = np.dot(centered_vertices, direction)
            
            # Find extrema (both positive and negative)
            max_proj_idx = np.argmax(projections)
            min_proj_idx = np.argmin(projections)
            
            extremal_candidates.append((max_proj_idx, distances_from_centroid[max_proj_idx]))
            extremal_candidates.append((min_proj_idx, distances_from_centroid[min_proj_idx]))
        
        # Method 3: Coordinate extrema (simple but effective)
        for axis in range(3):  # x, y, z axes
            max_idx = np.argmax(vertices[:, axis])
            min_idx = np.argmin(vertices[:, axis])
            extremal_candidates.append((max_idx, distances_from_centroid[max_idx]))
            extremal_candidates.append((min_idx, distances_from_centroid[min_idx]))
        
        # Remove duplicates and sort by distance from centroid
        unique_candidates = {}
        for idx, dist in extremal_candidates:
            if idx not in unique_candidates or dist > unique_candidates[idx]:
                unique_candidates[idx] = dist
        
        # Sort candidates by distance from centroid (farthest first)
        sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Select top candidates ensuring minimum distance between them
        selected_tips = []
        min_distance = bbox_diagonal * 0.08  # 8% of bounding box diagonal
        
        for idx, dist in sorted_candidates:
            vertex_pos = vertices[idx]
            
            # Check distance to already selected tips
            too_close = False
            for selected_pos in selected_tips:
                if np.linalg.norm(vertex_pos - selected_pos) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected_tips.append(vertex_pos)
                if len(selected_tips) >= max_tips:
                    break
        
        # Ensure we have at least min_tips
        if len(selected_tips) < min_tips:
            print(f"Only found {len(selected_tips)} well-separated tips, adding more candidates...")
            
            # Add more candidates with relaxed distance constraint
            relaxed_min_distance = min_distance * 0.6
            
            for idx, dist in sorted_candidates:
                if len(selected_tips) >= min_tips:
                    break
                    
                vertex_pos = vertices[idx]
                
                # Check if already selected
                already_selected = False
                for selected_pos in selected_tips:
                    if np.linalg.norm(vertex_pos - selected_pos) < 1e-6:
                        already_selected = True
                        break
                
                if already_selected:
                    continue
                
                # Check relaxed distance constraint
                too_close = False
                for selected_pos in selected_tips:
                    if np.linalg.norm(vertex_pos - selected_pos) < relaxed_min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    selected_tips.append(vertex_pos)
        
        tips_array = np.array(selected_tips)
        
        print(f"Geometric detection found {len(tips_array)} tips:")
        for i, tip in enumerate(tips_array):
            print(f"  Geom tip {i+1}: [{tip[0]:7.3f}, {tip[1]:7.3f}, {tip[2]:7.3f}]")
        
        return tips_array
    
    def _remove_duplicate_tips(self, tips_array, min_distance):
        """
        Remove duplicate tips that are too close to each other
        
        Args:
            tips_array: Array of tip positions
            min_distance: Minimum distance between tips
            
        Returns:
            Filtered array of tip positions
        """
        if len(tips_array) <= 1:
            return tips_array
        
        # Compute pairwise distances
        distances = cdist(tips_array, tips_array)
        
        # Use greedy selection to keep tips that are far enough apart
        selected_mask = np.zeros(len(tips_array), dtype=bool)
        
        # Start with the tip that has the maximum distance to all other tips
        total_distances = np.sum(distances, axis=1)
        start_idx = np.argmax(total_distances)
        selected_mask[start_idx] = True
        
        # Iteratively add tips that are far enough from already selected ones
        while True:
            best_idx = -1
            best_min_distance = 0
            
            for i in range(len(tips_array)):
                if selected_mask[i]:
                    continue
                
                # Find minimum distance to already selected tips
                selected_indices = np.where(selected_mask)[0]
                min_dist_to_selected = np.min(distances[i, selected_indices])
                
                # Select the tip that has the maximum minimum distance to selected tips
                if min_dist_to_selected >= min_distance and min_dist_to_selected > best_min_distance:
                    best_idx = i
                    best_min_distance = min_dist_to_selected
            
            if best_idx == -1:
                break
            
            selected_mask[best_idx] = True
        
        filtered_tips = tips_array[selected_mask]
        print(f"Removed {len(tips_array) - len(filtered_tips)} duplicate/close tips, kept {len(filtered_tips)}")
        
        return filtered_tips
    
    def generate_multiview_renders(self, mesh_file):
        """
        Step 2a: Generate multi-view renderings of the 3D model
        
        Args:
            mesh_file: Path to input mesh file
            
        Returns:
            normalization_params: Dictionary with normalization parameters
        """
        
        # Load and normalize mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if mesh.is_empty():
            raise ValueError("Model loading failed, please check the file format or file path.")
        
        # Store the original center and scale for later transformation
        original_center = mesh.get_center()
        original_scale = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
        
        # Normalize the mesh to be centered at origin with unit scale
        mesh.translate(-original_center)
        mesh.scale(1.0/original_scale, center=np.array([0, 0, 0]))
        
        # Store normalization parameters
        normalization_params = {
            "original_center": original_center.tolist(),
            "original_scale": float(original_scale)
        }
        
        normalization_file = os.path.join(self.temp_dir, 'normalization.json')
        with open(normalization_file, "w") as f:
            json.dump(normalization_params, f, indent=2)
        
        mesh.compute_vertex_normals()
        
        # Set up rendering parameters
        center = np.array([0, 0, 0])
        radius = 2.0
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        
        view_ctl = vis.get_view_control()
        
        # Generate views
        for i in range(self.num_views):
            angle = 2 * math.pi * i / self.num_views
            cam_pos = center + np.array([radius * math.cos(angle), 0, radius * math.sin(angle)])
            
            # Set camera parameters
            view_ctl.set_lookat(center)
            front_vec = center - cam_pos
            front_vec = front_vec / np.linalg.norm(front_vec)
            view_ctl.set_front(front_vec)
            view_ctl.set_up(np.array([0, 1, 0]))
            
            vis.poll_events()
            vis.update_renderer()
            
            # Get camera parameters
            cam_params = view_ctl.convert_to_pinhole_camera_parameters()
            intrinsic = cam_params.intrinsic.intrinsic_matrix
            extrinsic = cam_params.extrinsic
            
            # Render and save image
            image = vis.capture_screen_float_buffer(do_render=True)
            image_file = os.path.join(self.render_dir, f"camera_view_{i}.png")
            plt.imsave(image_file, np.asarray(image))
            
            # Save camera parameters
            params = {
                "K": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
                "view_angle": angle,
                "normalized": True
            }
            cam_params_file = os.path.join(self.temp_dir, f"camera_view_{i}_camparams.json")
            with open(cam_params_file, "w") as f:
                json.dump(params, f, indent=2)
        
        vis.destroy_window()
        return normalization_params
    
    def predict_2d_keypoints(self, pose2d="animal"):
        """
        Step 2b: Predict 2D keypoints from rendered images
        
        Args:
            pose2d: 2D pose estimation model name
        """
        
        # MMPose configuration
        POSE2D_SPECIFIC_ARGS = dict(
            yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
            rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
        )
        
        # Initialize filter parameters
        filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
        
        # Update model-specific filter parameters
        if pose2d is not None:
            for model in POSE2D_SPECIFIC_ARGS:
                if model in pose2d:
                    filter_args.update(POSE2D_SPECIFIC_ARGS[model])
                    break
        
        # Build initialization arguments
        init_args = {
            'pose2d': pose2d,
            'pose2d_weights': None,
            'scope': 'mmpose',
            'device': str(self.device),
            'det_model': None,
            'det_weights': None,
            'det_cat_ids': [0],
            'pose3d': None,
            'pose3d_weights': None,
            'show_progress': False
        }
        
        # Build call arguments
        call_args = {
            'inputs': self.render_dir,
            'vis_out_dir': None,
            'pred_out_dir': self.pred_dir,
            'bbox_thr': filter_args['bbox_thr'],
            'nms_thr': filter_args['nms_thr'],
            'pose_based_nms': filter_args['pose_based_nms'],
            'show': False,
            'draw_bbox': False,
            'draw_heatmap': False,
            'kpt_thr': 0.3,
            'tracking_thr': 0.3,
            'use_oks_tracking': False,
            'disable_norm_pose_2d': False,
            'disable_rebase_keypoint': False,
            'num_instances': 1,
            'radius': 3,
            'thickness': 1,
            'skeleton_style': 'mmpose',
            'black_background': False,
        }
        
        # Completely suppress all output during model loading and inference
        import sys
        import os
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        
        # Create buffer for suppressed output
        suppressed_output = StringIO()
        
        try:
            # Completely suppress both stdout and stderr during model loading
            with redirect_stdout(suppressed_output), redirect_stderr(suppressed_output):
                inferencer = MMPoseInferencer(**init_args)
            
            # Show progress for the 16 camera views being processed  
            image_files = [f for f in os.listdir(self.render_dir) if f.endswith('.png')]
            with tqdm(total=len(image_files), desc="2D keypoint detection", leave=True) as pbar:
                with redirect_stdout(suppressed_output), redirect_stderr(suppressed_output):
                    for _ in inferencer(**call_args):
                        pbar.update(1)
                        
        except Exception as e:
            # If something goes wrong, restore output and show the error
            print(f"Error in keypoint detection: {e}")
            raise
        
    
    def _find_geodesic_path_points(self, mesh, start_point, end_point, num_intermediate_points=5):
        """
        Find points along the geodesic path on the mesh surface between two 3D points
        
        Args:
            mesh: Trimesh object
            start_point: Starting 3D point (e.g., keypoint 4 - nose)
            end_point: Ending 3D point (e.g., tail tip)
            num_intermediate_points: Number of intermediate points to sample along the path
            
        Returns:
            path_points: Array of 3D points along the geodesic path (including start and end)
        """
        try:
            
            # Simplify mesh if needed for geodesic computation
            working_mesh = mesh
            if len(mesh.vertices) > 30000:
                print("Simplifying mesh for geodesic path computation...")
                working_mesh = self._simplify_mesh(mesh, target_vertices=20000)
            
            # Find nearest vertices to start and end points
            kdtree = KDTree(working_mesh.vertices)
            _, start_vertex_idx = kdtree.query(start_point)
            _, end_vertex_idx = kdtree.query(end_point)
            
            
            # Create adjacency matrix for geodesic computation
            adjacency_matrix = self._create_mesh_adjacency_matrix(working_mesh)
            
            # Compute geodesic distances from start vertex to all vertices
            distances, predecessors = dijkstra(
                adjacency_matrix, 
                indices=start_vertex_idx, 
                directed=False, 
                return_predecessors=True
            )
            
            # Reconstruct the geodesic path
            path_vertex_indices = []
            current_vertex = end_vertex_idx
            
            while current_vertex != start_vertex_idx:
                path_vertex_indices.append(current_vertex)
                current_vertex = predecessors[current_vertex]
                
                # Safety check to avoid infinite loops
                if current_vertex == -9999 or len(path_vertex_indices) > len(working_mesh.vertices):
                    # This can happen with complex mesh topology - use direct path as fallback
                    break
            
            path_vertex_indices.append(start_vertex_idx)
            path_vertex_indices.reverse()
            
            
            # Get 3D coordinates of path vertices
            path_vertices_3d = working_mesh.vertices[path_vertex_indices]
            
            # Compute cumulative distances along the path
            cumulative_distances = [0.0]
            for i in range(1, len(path_vertices_3d)):
                dist = np.linalg.norm(path_vertices_3d[i] - path_vertices_3d[i-1])
                cumulative_distances.append(cumulative_distances[-1] + dist)
            
            total_path_length = cumulative_distances[-1]
            
            # Sample points evenly along the path
            sampled_points = []
            target_distances = np.linspace(0, total_path_length, num_intermediate_points + 2)
            
            for target_dist in target_distances:
                # Find the segment containing the target distance
                segment_idx = 0
                for i in range(len(cumulative_distances) - 1):
                    if cumulative_distances[i] <= target_dist <= cumulative_distances[i + 1]:
                        segment_idx = i
                        break
                
                # Interpolate within the segment
                if segment_idx < len(path_vertices_3d) - 1:
                    t = (target_dist - cumulative_distances[segment_idx]) / max(
                        cumulative_distances[segment_idx + 1] - cumulative_distances[segment_idx], 1e-6
                    )
                    t = np.clip(t, 0.0, 1.0)
                    
                    interpolated_point = (
                        (1 - t) * path_vertices_3d[segment_idx] + 
                        t * path_vertices_3d[segment_idx + 1]
                    )
                    sampled_points.append(interpolated_point)
                else:
                    sampled_points.append(path_vertices_3d[-1])
            
            sampled_points = np.array(sampled_points)
            
            # Map points back to original mesh if we used a decimated mesh
            if len(mesh.vertices) != len(working_mesh.vertices):
                print("Mapping path points back to original mesh...")
                original_kdtree = KDTree(mesh.vertices)
                mapped_points = []
                
                for point in sampled_points:
                    _, nearest_idx = original_kdtree.query(point)
                    mapped_points.append(mesh.vertices[nearest_idx])
                
                sampled_points = np.array(mapped_points)
            
            
            return sampled_points
            
        except Exception as e:
            print(f"Error computing geodesic path: {e}")
            print("Falling back to linear interpolation in 3D space...")
            
            # Fallback: linear interpolation in 3D space
            t_values = np.linspace(0, 1, num_intermediate_points + 2)
            linear_points = []
            
            for t in t_values:
                interpolated_point = (1 - t) * start_point + t * end_point
                linear_points.append(interpolated_point)
            
            return np.array(linear_points)

    def triangulate_3d_keypoints(self, input_mesh, normalization_params, tips_positions, score_thr=0.5, add_spine_points=True, num_spine_points=5):
        """
        Step 3: Triangulate 3D keypoints, add tail tip, and optionally add spine points
        
        Args:
            input_mesh: Path to the original input mesh file
            normalization_params: Normalization parameters from rendering
            tips_positions: Detected tip positions
            score_thr: Score threshold for 2D keypoints
            add_spine_points: Whether to add points along the spine from nose to tail
            num_spine_points: Number of intermediate spine points to add
            
        Returns:
            keypoints_3d: Final 3D keypoints in original coordinate system
        """
        
        # Load camera parameters
        cam_params_dict = {}
        for i in range(self.num_views):
            cam_file = os.path.join(self.temp_dir, f"camera_view_{i}_camparams.json")
            if os.path.exists(cam_file):
                with open(cam_file, "r") as f:
                    cam_params_dict[str(i)] = json.load(f)
        
        # Load 2D predictions
        predictions = {}
        for filename in sorted(os.listdir(self.pred_dir)):
            if filename.endswith('.json') and filename.startswith('camera_view'):
                try:
                    view_id = filename.split('_')[-1].split('.')[0]
                except IndexError:
                    continue
                file_path = os.path.join(self.pred_dir, filename)
                with open(file_path, "r") as f:
                    data = json.load(f)
                if len(data) > 0:
                    keypoints = np.array(data[0]["keypoints"])
                    keypoint_scores = np.array(data[0]["keypoint_scores"])
                    predictions[view_id] = (keypoints, keypoint_scores)
        
        if not predictions:
            raise ValueError("No 2D prediction results found")
        
        # Get number of keypoints
        sample_keypoints, _ = next(iter(predictions.values()))
        num_keypoints = sample_keypoints.shape[0]
        
        # Triangulate each keypoint
        keypoints_3d = []
        for kp_idx in range(num_keypoints):
            pts_norm_list = []
            proj_list = []
            
            for view_id, (keypoints, scores) in predictions.items():
                if scores[kp_idx] >= score_thr:
                    if view_id not in cam_params_dict:
                        continue
                    
                    params = cam_params_dict[view_id]
                    K = np.array(params["K"])
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    
                    u, v = keypoints[kp_idx]
                    u_norm = (u - cx) / fx
                    v_norm = (v - cy) / fy
                    pts_norm_list.append([u_norm, v_norm])
                    
                    extrinsic = np.array(params["extrinsic"])[:3]
                    proj_list.append(extrinsic)
            
            if len(pts_norm_list) >= 2:
                kp3d = self._triangulate_point(pts_norm_list, proj_list)
            else:
                kp3d = np.array([np.nan, np.nan, np.nan])
            
            keypoints_3d.append(kp3d)
        
        keypoints_3d = np.array(keypoints_3d)
        
        # Denormalize keypoints
        original_center = np.array(normalization_params["original_center"])
        original_scale = normalization_params["original_scale"]
        denormalized_keypoints = keypoints_3d * original_scale + original_center
        
        # Add tail tip - the tip that is farthest from all keypoints using geodesic distance
        tail_tip_pos = None
        if tips_positions is not None and len(tips_positions) > 0:
            
            # Load the original mesh for geodesic computation
            tail_tip_pos = self._find_tail_tip_geodesic(input_mesh, tips_positions, denormalized_keypoints, normalization_params)
            
            if tail_tip_pos is not None:
                # Add tail tip as new keypoint
                denormalized_keypoints = np.vstack([denormalized_keypoints, tail_tip_pos.reshape(1, 3)])
            else:
                print("Warning: Could not compute tail tip using geodesic distance")
        
        # Add spine points along geodesic path from keypoint 4 (nose) to tail tip
        if add_spine_points and tail_tip_pos is not None and len(denormalized_keypoints) > 4:
            
            # Get keypoint 4 (nose/head) and tail tip positions
            nose_pos = denormalized_keypoints[4]  # Keypoint 4 is typically the nose
            
            # Check if nose position is valid
            if not np.isnan(nose_pos).any():
                # Load the original mesh
                mesh = trimesh.load(input_mesh)
                
                if hasattr(mesh, 'vertices'):
                    # Find geodesic path points
                    spine_path_points = self._find_geodesic_path_points(
                        mesh, nose_pos, tail_tip_pos, num_spine_points
                    )
                    
                    # Add intermediate spine points (exclude start and end as they're already in keypoints)
                    if len(spine_path_points) > 2:
                        intermediate_spine_points = spine_path_points[1:-1]  # Exclude first and last
                        
                        # Add spine points to keypoints
                        denormalized_keypoints = np.vstack([denormalized_keypoints, intermediate_spine_points])
                        
                    else:
                        print("Warning: Could not generate sufficient spine path points")
                else:
                    print("Warning: Could not load mesh for spine point computation")
            else:
                print("Warning: Keypoint 4 (nose) position is invalid, skipping spine points")
        
        return denormalized_keypoints
    
    def _triangulate_point(self, pts_list, proj_matrices):
        """Triangulate a single 3D point from multiple 2D observations"""
        A = []
        for (u, v), P in zip(pts_list, proj_matrices):
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        return X[:3]
    
    def _find_tail_tip_geodesic(self, input_mesh, tips_positions, keypoints_3d, normalization_params):
        """
        Find the tail tip as the tip that is farthest from all keypoints using geodesic distance
        
        Args:
            input_mesh: Path to the original mesh file
            tips_positions: Array of detected tip positions
            keypoints_3d: Array of detected keypoints (in original coordinate system)
            normalization_params: Parameters used for mesh normalization
            
        Returns:
            tail_tip_position: Position of the tail tip, or None if computation failed
        """
        try:
            
            # Load the original mesh
            mesh = trimesh.load(input_mesh)
            if not hasattr(mesh, 'vertices'):
                print("Warning: Could not load mesh for geodesic computation")
                return None
            
            
            # Simplify mesh if it's too large for geodesic computation
            if len(mesh.vertices) > 50000:
                print(f"Mesh too large for geodesic computation, simplifying...")
                mesh = self._simplify_mesh(mesh, target_vertices=30000)
                print(f"Decimated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Create adjacency matrix for geodesic distance computation
            adjacency_matrix = self._create_mesh_adjacency_matrix(mesh)
            
            # Find closest vertices on mesh for each tip and keypoint
            valid_keypoints = keypoints_3d[~np.isnan(keypoints_3d).any(axis=1)]
            all_points = np.vstack([valid_keypoints, tips_positions])
            
            # Create KDTree for finding nearest vertices
            kdtree = KDTree(mesh.vertices)
            
            # Find nearest vertex indices for all points
            _, nearest_vertex_indices = kdtree.query(all_points)
            
            num_keypoints = len(valid_keypoints)
            tip_vertex_indices = nearest_vertex_indices[num_keypoints:]
            keypoint_vertex_indices = nearest_vertex_indices[:num_keypoints]
            
            
            # For each tip, compute total geodesic distance to all keypoints
            # and check if it's closer to keypoint 4 than to any other keypoint
            tip_scores = []
            valid_tail_candidates = []
            
            for i, tip_vertex_idx in enumerate(tip_vertex_indices):
                # Compute geodesic distance from this tip to all keypoints
                distances = dijkstra(adjacency_matrix, indices=tip_vertex_idx, directed=False)
                
                # Get distances to all keypoints
                keypoint_distances = []
                total_geodesic_distance = 0
                
                for kp_vertex_idx in keypoint_vertex_indices:
                    geodesic_dist = distances[kp_vertex_idx]
                    if np.isinf(geodesic_dist):
                        # If geodesic distance is infinite, use a large penalty
                        geodesic_dist = 1e6
                    keypoint_distances.append(geodesic_dist)
                    total_geodesic_distance += geodesic_dist
                
                tip_scores.append(total_geodesic_distance)
                
                # Check if this tip is closer to keypoint 4 than to any other keypoint
                # Note: keypoint indices are 0-based, so keypoint 4 is at index 4
                is_closer_to_kp4 = False
                if len(keypoint_distances) > 4:  # Ensure keypoint 4 exists
                    kp4_distance = keypoint_distances[4]
                    
                    # Check if keypoint 4 is the closest keypoint to this tip
                    closest_kp_distance = min(keypoint_distances)
                    is_closer_to_kp4 = (kp4_distance == closest_kp_distance)
                    
                    # If there are ties, we still consider it valid if kp4 is among the closest
                    closest_kp_indices = [j for j, d in enumerate(keypoint_distances) if d == closest_kp_distance]
                    is_closer_to_kp4 = 4 in closest_kp_indices
                
                if is_closer_to_kp4:
                    valid_tail_candidates.append((i, total_geodesic_distance))
            
            # Select the tip with maximum total geodesic distance among valid candidates
            if valid_tail_candidates:
                # Sort by total geodesic distance (descending) and select the best one
                valid_tail_candidates.sort(key=lambda x: x[1], reverse=True)
                tail_tip_idx, max_distance = valid_tail_candidates[0]
                tail_tip_pos = tips_positions[tail_tip_idx]
                
                return tail_tip_pos
            else:
                print("Warning: No tips are closer to keypoint 4 than to other keypoints")
                # Fallback to original behavior if no valid candidates
                if tip_scores:
                    tail_tip_idx = np.argmax(tip_scores)
                    tail_tip_pos = tips_positions[tail_tip_idx]
                    print(f"Fallback: Selected tip {tail_tip_idx} as tail tip (max total geodesic distance: {tip_scores[tail_tip_idx]:.3f})")
                    return tail_tip_pos
                else:
                    print("Warning: No valid tip scores computed")
                    return None
                
        except Exception as e:
            print(f"Error computing geodesic distances: {e}")
            print("Falling back to farthest tip using Euclidean distance...")
            
            # Fallback: use tip that is farthest from all keypoints using Euclidean distance
            # but still closer to keypoint 4 than to any other keypoint
            valid_keypoints = keypoints_3d[~np.isnan(keypoints_3d).any(axis=1)]
            if len(valid_keypoints) == 0:
                return tips_positions[0] if len(tips_positions) > 0 else None
            
            tip_scores = []
            valid_tail_candidates_euclidean = []
            
            for i, tip_pos in enumerate(tips_positions):
                # Compute Euclidean distances to all keypoints
                distances = np.linalg.norm(valid_keypoints - tip_pos, axis=1)
                total_distance = np.sum(distances)
                tip_scores.append(total_distance)
                
                # Check if this tip is closer to keypoint 4 than to any other keypoint
                is_closer_to_kp4_euclidean = False
                if len(distances) > 4:  # Ensure keypoint 4 exists
                    kp4_distance = distances[4]
                    
                    # Check if keypoint 4 is the closest keypoint to this tip
                    closest_kp_distance = min(distances)
                    is_closer_to_kp4_euclidean = (kp4_distance == closest_kp_distance)
                    
                    # If there are ties, we still consider it valid if kp4 is among the closest
                    closest_kp_indices = [j for j, d in enumerate(distances) if d == closest_kp_distance]
                    is_closer_to_kp4_euclidean = 4 in closest_kp_indices
                
                if is_closer_to_kp4_euclidean:
                    valid_tail_candidates_euclidean.append((i, total_distance))
                    print(f"Fallback Tip {i}: total Euclidean distance = {total_distance:.3f} (valid - closest to keypoint 4)")
                else:
                    print(f"Fallback Tip {i}: total Euclidean distance = {total_distance:.3f} (invalid - not closest to keypoint 4)")
            
            # Select from valid candidates first
            if valid_tail_candidates_euclidean:
                # Sort by total distance (descending) and select the best one
                valid_tail_candidates_euclidean.sort(key=lambda x: x[1], reverse=True)
                tail_tip_idx, max_distance = valid_tail_candidates_euclidean[0]
                print(f"Fallback: Selected tip {tail_tip_idx} as tail tip (max Euclidean distance among valid candidates: {max_distance:.3f})")
                return tips_positions[tail_tip_idx]
            elif tip_scores:
                # Ultimate fallback: ignore the keypoint 4 constraint
                tail_tip_idx = np.argmax(tip_scores)
                print(f"Ultimate Fallback: Selected tip {tail_tip_idx} as tail tip (max Euclidean distance, ignoring keypoint 4 constraint)")
                return tips_positions[tail_tip_idx]
            
            return None
    
    def _create_mesh_adjacency_matrix(self, mesh):
        """
        Create sparse adjacency matrix for geodesic distance computation
        
        Args:
            mesh: Trimesh object
            
        Returns:
            Sparse adjacency matrix with edge weights as Euclidean distances
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create lists for sparse matrix construction
        rows = []
        cols = []
        data = []
        
        # Add edges from faces with Euclidean distances as weights
        for face in faces:
            v0, v1, v2 = face
            
            # Add edges between all pairs of vertices in the face
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            
            for i, j in edges:
                if i != j:
                    # Compute Euclidean distance between vertices
                    distance = np.linalg.norm(vertices[i] - vertices[j])
                    
                    # Add both directions (undirected graph)
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([distance, distance])
        
        # Create sparse matrix
        n_vertices = len(vertices)
        adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
        
        return adjacency_matrix
    
    def _simplify_mesh(self, mesh, target_vertices=50000):
        """
        Simplify mesh using Open3D quadric decimation
        
        Args:
            mesh: Trimesh object
            target_vertices: Target number of vertices
            
        Returns:
            Decimated trimesh object
        """
        print("Simplifying mesh using Open3D...")
        
        # Convert trimesh to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        
        # Calculate target number of triangles (roughly 2x vertices for typical meshes)
        target_triangles = min(target_vertices * 2, len(mesh.faces))
        
        # Simplify using quadric decimation
        print(f"Decimating from {len(mesh.faces)} to ~{target_triangles} triangles...")
        o3d_mesh_simplified = o3d_mesh.simplify_quadric_decimation(target_triangles)
        
        # Convert back to trimesh
        vertices_decimated = np.asarray(o3d_mesh_simplified.vertices)
        faces_decimated = np.asarray(o3d_mesh_simplified.triangles)
        
        # Create new trimesh object
        decimated_mesh = trimesh.Trimesh(vertices=vertices_decimated, faces=faces_decimated)
        
        # Ensure mesh is properly oriented and manifold
        decimated_mesh.fix_normals()
        decimated_mesh.fill_holes()
        
        reduction_ratio = len(vertices_decimated) / len(mesh.vertices)
        print(f"Mesh decimated: {reduction_ratio:.2%} vertices retained")
        
        return decimated_mesh
    
    
    def run(self, input_mesh, output_file, pose2d="animal", 
            num_eigenvalues=80, time_samples=40, 
            curvature_threshold=0.8, hks_threshold=0.0, min_distance_ratio=0.1,
            max_vertices=50000, use_hks=True, 
            add_spine_points=True, num_spine_points=5):
        """
        Run the complete pipeline
        
        Args:
            input_mesh: Path to input 3D mesh file
            output_file: Path to output keypoints file (.npy)
            pose2d: 2D pose estimation model name
            num_eigenvalues: Number of eigenvalues for HKS
            time_samples: Number of time scales for HKS
            curvature_threshold: Curvature threshold for tip detection
            hks_threshold: HKS threshold for tip detection
            min_distance_ratio: Minimum distance ratio for tip detection
            max_vertices: Maximum vertices before mesh simplification
            use_hks: Whether to use HKS for additional keypoint detection (tail tip)
            add_spine_points: Whether to add points along the spine from nose to tail
            num_spine_points: Number of intermediate spine points to add
            
        Returns:
            keypoints_3d: Final 3D keypoints array
        """
        
        try:
            # Step 1: Detect tips (only if HKS is enabled)
            if use_hks:
                tips_positions = self.detect_tips(
                    input_mesh, num_eigenvalues, time_samples,
                    curvature_threshold, hks_threshold, min_distance_ratio, max_vertices
                )
            else:
                print(f"\n{'='*60}")
                print("STEP 1: SKIPPING HKS TIP DETECTION (--no_hks enabled)")
                print(f"{'='*60}")
                print("HKS tip detection disabled - only using 2D pose estimation keypoints")
                tips_positions = None
            
            # Step 2a: Generate multi-view renderings
            normalization_params = self.generate_multiview_renders(input_mesh)
            
            # Step 2b: Predict 2D keypoints
            self.predict_2d_keypoints(pose2d)
            
            # Step 3: Triangulate 3D keypoints
            keypoints_3d = self.triangulate_3d_keypoints(
                input_mesh, normalization_params, tips_positions, 
                add_spine_points=add_spine_points, num_spine_points=num_spine_points
            )
            
            # Save results
            np.save(output_file, keypoints_3d)
            
            # Also save as JSON for inspection
            json_file = output_file.replace('.npy', '_info.json')
            
            # Count different types of keypoints
            num_original_pose = 17
            num_tail_tip = 1 if use_hks and len(keypoints_3d) > 17 else 0
            num_spine_pts = max(0, len(keypoints_3d) - num_original_pose - num_tail_tip)
            
            keypoints_info = {
                "keypoints": keypoints_3d.tolist(),
                "num_keypoints": len(keypoints_3d),
                "num_pose_keypoints": num_original_pose,
                "num_tail_tip": num_tail_tip,
                "num_spine_points": num_spine_pts,
                "num_tips": len(tips_positions) if tips_positions is not None else 0,
                "hks_enabled": use_hks,
                "spine_points_enabled": add_spine_points,
                "normalization_params": normalization_params,
                "pipeline_params": {
                    "num_views": self.num_views,
                    "pose2d_model": pose2d,
                    "use_hks": use_hks,
                    "add_spine_points": add_spine_points,
                    "num_spine_points": num_spine_points,
                    "num_eigenvalues": num_eigenvalues if use_hks else None,
                    "time_samples": time_samples if use_hks else None,
                    "curvature_threshold": curvature_threshold if use_hks else None,
                    "hks_threshold": hks_threshold if use_hks else None,
                    "min_distance_ratio": min_distance_ratio if use_hks else None
                }
            }
            
            with open(json_file, 'w') as f:
                json.dump(keypoints_info, f, indent=2)
            
            
            return keypoints_3d
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise

