"""
DINO Feature Matching Module

Contains DinoMatcher class for feature extraction and matching between 3D models.
Uses resource management for batch processing.
"""

import os
import json
import copy
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional
import warnings
import logging

# Suppress specific warnings  
warnings.filterwarnings('ignore')
logging.getLogger('mmengine').setLevel(logging.ERROR)
logging.getLogger('mmpose').setLevel(logging.ERROR)
logging.getLogger('mmdet').setLevel(logging.ERROR)
logging.getLogger('mmcv').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
# logging.getLogger('torch').setLevel(logging.ERROR)  # Can interfere with PyTorch 2.4+ logging

# Set deterministic behavior
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import global resource manager and shared functions
from ..utils.common import resource_manager, compute_geodesic_distances, label_vertices_by_keypoints

# ========================================
# KEYPOINT LABELING FUNCTIONS
# ========================================

def find_vertex_correspondences_by_labels(features1, vertex_indices1, vertex_labels1,
                                          features2, vertex_indices2, vertex_labels2,
                                          mesh1_vertices=None, mesh2_vertices=None,
                                          bidirectional_consistency=True, use_global_optimization=False):
    """Find bijective correspondences between vertices with the same labels using DINO features"""
    
    print(f"Finding label-constrained bijective correspondences using DINO features...")
    print(f"  Matching strategy: {'Global optimization' if use_global_optimization else 'Nearest neighbor'}")
    if not use_global_optimization:
        print(f"  Bidirectional consistency: {'Enabled' if bidirectional_consistency else 'Disabled'}")
    
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
    print(f"Found {len(common_labels)} common labels: {sorted(common_labels)}")
    
    all_correspondences = []
    
    # Process each label group separately
    for label in common_labels:
        vertices1 = label_groups1[label]
        vertices2 = label_groups2[label]
        features1_label = np.array(feature_groups1[label])
        features2_label = np.array(feature_groups2[label])
        
        if len(vertices1) == 0 or len(vertices2) == 0:
            continue
        
        print(f"  Processing label {label}: {len(vertices1)} vertices ↔ {len(vertices2)} vertices")
        
        # Find correspondences within this label group using the specified matching strategy
        label_correspondences = find_vertex_correspondences_within_label(
            features1_label, vertices1,
            features2_label, vertices2,
            label, mesh1_vertices, mesh2_vertices,
            bidirectional_consistency, use_global_optimization
        )
        
        all_correspondences.extend(label_correspondences)
    
    print(f"  Total correspondences found: {len(all_correspondences)}")
    
    return all_correspondences

def find_vertex_correspondences_within_label(features1, vertices1, features2, vertices2, label, 
                                            mesh1_vertices=None, mesh2_vertices=None,
                                            bidirectional_consistency=True, use_global_optimization=False):
    """Find correspondences between vertices with the same label"""
    
    if use_global_optimization:
        # Use Hungarian algorithm for optimal assignment
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        # Compute pairwise distances between features
        distance_matrix = cdist(features1, features2, metric='cosine')
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        correspondences = []
        for i, j in zip(row_indices, col_indices):
            correspondences.append((vertices1[i], vertices2[j], distance_matrix[i, j]))
        
        print(f"    Label {label} correspondences (global optimization): {len(correspondences)}")
        
    else:
        # Use nearest neighbor matching with optional bidirectional consistency
        from sklearn.neighbors import NearestNeighbors
        
        # Forward matching: mesh1 -> mesh2
        nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(features2)
        distances, indices = nbrs.kneighbors(features1)
        
        forward_matches = [(vertices1[i], vertices2[indices[i, 0]], distances[i, 0]) for i in range(len(vertices1))]
        
        if bidirectional_consistency:
            # Backward matching: mesh2 -> mesh1
            nbrs_back = NearestNeighbors(n_neighbors=1, metric='cosine').fit(features1)
            distances_back, indices_back = nbrs_back.kneighbors(features2)
            
            # Keep only mutually consistent matches
            correspondences = []
            for i, (v1, v2, dist) in enumerate(forward_matches):
                # Check if the backward match from v2 points back to v1
                v2_idx = indices[i, 0]
                if indices_back[v2_idx, 0] == i:
                    correspondences.append((v1, v2, dist))
            
            print(f"    Label {label} correspondences (bidirectional consistency): {len(correspondences)}")
        else:
            correspondences = forward_matches
            print(f"    Label {label} correspondences (forward matching): {len(correspondences)}")
    
    return correspondences


# ========================================
# DINO FEATURE EXTRACTION CLASSES
# ========================================

class DinoFeatureProcessor:
    """Handles DINO feature processing with batch optimization"""
    
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14",
                 image_size=448, pca_dim=32, device=None):
        self.repo_name = repo_name
        self.model_name = model_name
        self.image_size = image_size
        self.pca_dim = pca_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        
        # Load DINO model
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.model = torch.hub.load(repo_or_dir=self.repo_name, model=self.model_name, force_reload=False)
        self.model = self.model.to(self.device).eval()
        
        # Initialize PCA if needed
        self.pca = None
        if self.pca_dim is not None and self.pca_dim > 0:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.pca_dim)
            self.pca_fitted = False

    def prepare_images_batch(self, images_list):
        """Convert list of images to batched tensor for processing"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        batch_tensors = []
        for img in images_list:
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
        
        return torch.stack(batch_tensors).to(self.device)
    
    def extract_dense_features_batch(self, image_batch):
        """Extract DINO features from a batch of images"""
        with torch.no_grad():
            features = self.model.forward_features(image_batch)['x_norm_patchtokens']
        return features
    
    def bilinear_sample_features(self, features, pixel_coords, grid_size, resize_scale):
        """Sample features at pixel coordinates using bilinear interpolation"""
        # Implementation details for bilinear sampling
        # Convert pixel coordinates to grid coordinates for sampling
        batch_size, seq_len, feature_dim = features.shape
        h = w = int(np.sqrt(seq_len))
        
        features_2d = features.view(batch_size, h, w, feature_dim).permute(0, 3, 1, 2)
        
        # Normalize coordinates to [-1, 1] for grid_sample
        normalized_coords = pixel_coords / resize_scale * 2.0 - 1.0
        normalized_coords = normalized_coords.unsqueeze(1)  # Add height dimension
        
        sampled = F.grid_sample(features_2d, normalized_coords, mode='bilinear', align_corners=False)
        return sampled.squeeze(2).permute(0, 2, 1)


class DinoConsistencyLoss:
    """Computes consistency loss for multi-view DINO features"""
    
    def __init__(self, consistency_weight=1.0):
        self.consistency_weight = consistency_weight
    
    def compute_consistency_loss(self, vertex_features_dict):
        """Compute consistency loss across multiple views for vertices"""
        total_loss = 0.0
        num_vertices = len(vertex_features_dict)
        
        for vertex_idx, view_features in vertex_features_dict.items():
            if len(view_features) < 2:
                continue
                
            # Compute pairwise cosine similarities
            features_tensor = torch.stack([torch.tensor(feat) for feat in view_features])
            normalized_features = F.normalize(features_tensor, dim=1)
            
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # Loss is negative average similarity (encourage high similarity)
            mask = torch.eye(len(view_features)) == 0  # Exclude diagonal
            avg_similarity = similarity_matrix[mask].mean()
            total_loss += (1.0 - avg_similarity)
        
        return total_loss / num_vertices if num_vertices > 0 else 0.0
    
    def weighted_feature_aggregation(self, vertex_features_dict, vertex_counts):
        """Aggregate features across views with consistency weighting"""
        aggregated_features = {}
        
        for vertex_idx, view_features in vertex_features_dict.items():
            if not view_features:
                continue
                
            # Simple averaging for now
            features_tensor = torch.stack([torch.tensor(feat) for feat in view_features])
            aggregated = features_tensor.mean(dim=0)
            aggregated_features[vertex_idx] = aggregated.cpu().numpy()
        
        return aggregated_features 


# Utility functions for 3D to 2D projection
def project_3d_to_2d(point_3d, intrinsic, extrinsic):
    """Project 3D point to 2D image coordinates"""
    # Convert to homogeneous coordinates
    point_3d_h = np.append(point_3d, 1.0)
    
    # Apply extrinsic transformation (world to camera)
    point_cam_h = np.dot(extrinsic, point_3d_h)
    point_cam = point_cam_h[:3] / point_cam_h[3]
    
    # Apply intrinsic transformation (camera to image)
    point_2d_h = np.dot(intrinsic, point_cam)
    point_2d = point_2d_h[:2] / point_2d_h[2]
    
    return point_2d

def get_visible_vertices(vertices, intrinsic, extrinsic, image_shape, depth_buffer):
    """Determine which vertices are visible in the rendered view"""
    visible_mask = np.zeros(len(vertices), dtype=bool)
    
    for i, vertex in enumerate(vertices):
        # Project to 2D
        point_2d = project_3d_to_2d(vertex, intrinsic, extrinsic)
        
        # Check if within image bounds
        x, y = int(point_2d[0]), int(point_2d[1])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            # Check depth (simple approximation)
            if depth_buffer is not None:
                # Transform to camera space to get depth
                point_3d_h = np.append(vertex, 1.0)
                point_cam_h = np.dot(extrinsic, point_3d_h)
                depth = point_cam_h[2] / point_cam_h[3]
                
                # Compare with depth buffer (with small tolerance)
                if abs(depth - depth_buffer[y, x]) < 0.01:
                    visible_mask[i] = True
            else:
                visible_mask[i] = True
    
    return visible_mask


# ========================================
# MAIN DINO MATCHER CLASS
# ========================================

class DinoMatcher:
    """Main class for DINO-based mesh feature extraction and matching"""
    
    def __init__(self, dino_model=None, dino_image_size=None, dino_pca_dim=None, 
                 num_views=None, render_width=None, render_height=None, consistency_weight=None, 
                 device=None, cache_manager=None):
        """Initialize DINO matcher with configuration parameters"""
        # Load configuration defaults
        from ..utils.config import ConfigManager
        config_manager = ConfigManager()
        dino_config = config_manager.get_dino_config()
        
        # Use config values as defaults if not provided
        self.dino_model = dino_model if dino_model is not None else dino_config['model_name']
        self.dino_image_size = dino_image_size if dino_image_size is not None else dino_config['image_size']
        self.dino_pca_dim = dino_pca_dim if dino_pca_dim is not None else dino_config['pca_dim']
        self.num_views = num_views if num_views is not None else dino_config['num_views']
        self.render_width = render_width if render_width is not None else dino_config['render_width']
        self.render_height = render_height if render_height is not None else dino_config['render_height']
        self.consistency_weight = consistency_weight if consistency_weight is not None else dino_config['consistency_weight']
        self.device = torch.device(device) if device is not None else torch.device(dino_config['device'])
        self.cache_manager = cache_manager
        
        # Initialize processors
        self.feature_processor = DinoFeatureProcessor(
            model_name=self.dino_model, image_size=self.dino_image_size, 
            pca_dim=self.dino_pca_dim, device=str(self.device)
        )
        self.consistency_processor = DinoConsistencyLoss(self.consistency_weight)
        
        # Cache for mesh features
        self.mesh_feature_cache = {}
    
    def _compute_mesh_hash(self, mesh):
        """Compute hash for mesh to enable caching"""
        vertices_hash = hash(mesh.vertices.tobytes())
        faces_hash = hash(mesh.faces.tobytes())
        return f"{vertices_hash}_{faces_hash}"
    
    def _interpolate_missing_features(self, vertex_features_matrix, vertices_with_features, vertices, 
                                    knn_neighbors=None):
        """Interpolate features for vertices without direct feature extraction"""
        from scipy.spatial import KDTree
        
        # Get knn_neighbors from config if not provided
        if knn_neighbors is None:
            from ..utils.config import ConfigManager
            config_manager = ConfigManager()
            dino_config = config_manager.get_dino_config()
            knn_neighbors = dino_config['knn_neighbors']
        
        if len(vertices_with_features) == 0:
            return np.zeros((len(vertices), vertex_features_matrix.shape[1]))
        
        # Build KDTree for vertices with features
        tree = KDTree(vertices[vertices_with_features])
        
        interpolated_features = np.zeros((len(vertices), vertex_features_matrix.shape[1]))
        
        for i, vertex in enumerate(vertices):
            if i in vertices_with_features:
                # Use direct feature
                feature_idx = np.where(vertices_with_features == i)[0][0]
                interpolated_features[i] = vertex_features_matrix[feature_idx]
            else:
                # Interpolate from k nearest neighbors
                distances, neighbor_indices = tree.query(vertex, k=min(knn_neighbors, len(vertices_with_features)))
                
                if np.isscalar(distances):
                    distances = [distances]
                    neighbor_indices = [neighbor_indices]
                
                # Inverse distance weighting
                weights = 1.0 / (np.array(distances) + 1e-8)
                weights /= weights.sum()
                
                interpolated_feature = np.zeros(vertex_features_matrix.shape[1])
                for w, neighbor_idx in zip(weights, neighbor_indices):
                    feature_idx = np.where(vertices_with_features == neighbor_idx)[0][0]
                    interpolated_feature += w * vertex_features_matrix[feature_idx]
                
                interpolated_features[i] = interpolated_feature
        
        return interpolated_features
    
    def load_keypoints(self, mesh, keypoints_path, mesh_name="mesh"):
        """Load keypoints from file and find corresponding vertex indices"""
        if keypoints_path and os.path.exists(keypoints_path):
            keypoints = np.load(keypoints_path)
            
            # Find closest vertices to keypoints
            from scipy.spatial import KDTree
            tree = KDTree(mesh.vertices)
            
            keypoint_vertex_indices = []
            for kp in keypoints:
                _, closest_idx = tree.query(kp)
                keypoint_vertex_indices.append(closest_idx)
            
            print(f"  Loaded {len(keypoints)} keypoints from {keypoints_path}")
            return keypoints, keypoint_vertex_indices
        else:
            print(f"  No keypoints file found for {mesh_name}")
            return None, None
    
    def extract_dino_features(self, mesh, mesh_name="mesh", force_rerender=False):
        """Extract DINO features for a mesh"""
        # Check cache first
        mesh_hash = self._compute_mesh_hash(mesh)
        if mesh_hash in self.mesh_feature_cache and not force_rerender:
            return self.mesh_feature_cache[mesh_hash]
        
        # Use renderer to get multi-view images
        from ..utils.renderer import render_views_for_dino
        rendered_views, normalization_params = render_views_for_dino(
            mesh_file=None,  # Pass mesh object directly
            num_views=self.num_views,
            enhance_images=True,
            cache_manager=self.cache_manager
        )
        
        # Extract features from rendered views
        vertex_features_dict = self._compute_vertex_features(mesh, rendered_views, normalization_params)
        
        # Cache results
        self.mesh_feature_cache[mesh_hash] = vertex_features_dict
        
        return vertex_features_dict

    def _compute_vertex_features(self, mesh, rendered_views, normalization_params):
        """Compute DINO features for each vertex by projecting from multiple views"""
        all_vertex_features = []
        all_vertex_visibility = []
        
        for view_id, view_data in rendered_views.items():
            image_np = view_data['image']
            depth_np = view_data['depth']
            mask_np = view_data['mask']
            camera_params = view_data['camera_params']
            
            # Extract DINO features from the image
            image_batch = self.feature_processor.prepare_images_batch([image_np])
            patch_features = self.feature_processor.extract_dense_features_batch(image_batch)
            
            # Project 3D vertices to 2D and sample features
            vertex_features = self._sample_vertex_features(mesh.vertices, patch_features[0], camera_params)
            visibility = self._check_visibility(mesh.vertices, camera_params, depth_np, mask_np)
            
            all_vertex_features.append(vertex_features)
            all_vertex_visibility.append(visibility)
        
        # Combine features from all views
        combined_features, coverage, consistency_loss = self._combine_multi_view_features(
            all_vertex_features, all_vertex_visibility
        )
        
        return {
            'features': combined_features,
            'coverage': coverage,
            'dino_consistency_loss': consistency_loss
        }

    def _sample_vertex_features(self, vertices, patch_features, camera_params):
        """Sample DINO features at vertex locations by projecting to image space"""
        feature_dim = patch_features.shape[-1]
        vertex_features = np.zeros((len(vertices), feature_dim))
        
        # Get intrinsic and extrinsic matrices from camera params
        intrinsic = camera_params.get('intrinsic', np.eye(3))
        extrinsic = camera_params.get('extrinsic', np.eye(4))
        
        # Calculate patch grid dimensions (assuming square patch grid)
        num_patches = patch_features.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            # If not square, use approximation
            grid_size = int(np.ceil(np.sqrt(num_patches)))
        
        # Reshape patch features to 2D grid
        patch_grid = patch_features.reshape(grid_size, grid_size, feature_dim)
        
        for i, vertex in enumerate(vertices):
            try:
                # Project vertex to 2D image coordinates
                point_2d = project_3d_to_2d(vertex, intrinsic, extrinsic)
                
                # Map to patch grid coordinates (normalized to [0, grid_size-1])
                # Assuming image coordinates are normalized to [0, 1]
                patch_x = max(0, min(grid_size - 1, int(point_2d[0] * grid_size)))
                patch_y = max(0, min(grid_size - 1, int(point_2d[1] * grid_size)))
                
                # Sample feature from patch grid
                vertex_features[i] = patch_grid[patch_y, patch_x]
                
            except (ValueError, IndexError):
                # If projection fails, use zero features
                vertex_features[i] = np.zeros(feature_dim)
        
        return vertex_features

    def _check_visibility(self, vertices, camera_params, depth_map, mask_map):
        """Check which vertices are visible in the current view"""
        intrinsic = camera_params.get('intrinsic', np.eye(3))
        extrinsic = camera_params.get('extrinsic', np.eye(4))
        image_shape = depth_map.shape if depth_map is not None else (256, 256)
        
        return get_visible_vertices(vertices, intrinsic, extrinsic, image_shape, depth_map)

    def _combine_multi_view_features(self, all_vertex_features, all_vertex_visibility):
        """Combine features from multiple views with consistency weighting"""
        num_vertices = len(all_vertex_features[0])
        feature_dim = all_vertex_features[0].shape[1]
        
        combined_features = np.zeros((num_vertices, feature_dim))
        visibility_counts = np.zeros(num_vertices)
        
        for view_features, visibility in zip(all_vertex_features, all_vertex_visibility):
            for v_idx in range(num_vertices):
                if visibility[v_idx]:
                    combined_features[v_idx] += view_features[v_idx]
                    visibility_counts[v_idx] += 1
        
        # Average features across views
        for v_idx in range(num_vertices):
            if visibility_counts[v_idx] > 0:
                combined_features[v_idx] /= visibility_counts[v_idx]
        
        coverage = np.mean(visibility_counts > 0)
        
        # Calculate consistency loss based on feature variance across views
        consistency_loss = 0.0
        if len(all_vertex_features) > 1:
            total_variance = 0.0
            valid_vertices = 0
            
            for v_idx in range(num_vertices):
                if visibility_counts[v_idx] > 1:
                    # Get features from all views for this vertex
                    vertex_features_across_views = []
                    for view_features, visibility in zip(all_vertex_features, all_vertex_visibility):
                        if visibility[v_idx]:
                            vertex_features_across_views.append(view_features[v_idx])
                    
                    if len(vertex_features_across_views) > 1:
                        # Calculate variance across views for this vertex
                        features_matrix = np.array(vertex_features_across_views)
                        vertex_variance = np.mean(np.var(features_matrix, axis=0))
                        total_variance += vertex_variance
                        valid_vertices += 1
            
            if valid_vertices > 0:
                consistency_loss = total_variance / valid_vertices
        
        return combined_features, coverage, consistency_loss

    def extract_mesh_features(self, mesh, mesh_name="mesh", force_rerender=False):
        """Extract mesh features (wrapper for compatibility)"""
        return self.extract_dino_features(mesh, mesh_name, force_rerender)

    def find_correspondences(self, features1, features2, vertices1, vertices2, 
                           bidirectional_consistency=True, use_global_optimization=False,
                           keypoints1=None, keypoints2=None, mesh1=None, mesh2=None):
        """Find correspondences between two sets of features"""
        
        if keypoints1 is not None and keypoints2 is not None and mesh1 is not None and mesh2 is not None:
            # Use keypoint-constrained matching
            print("Using keypoint-constrained matching...")
            
            # Label vertices by keypoints
            vertex_labels1, _ = label_vertices_by_keypoints(mesh1, keypoints1)
            vertex_labels2, _ = label_vertices_by_keypoints(mesh2, keypoints2)
            
            # Find correspondences within same labels
            vertex_indices1 = np.arange(len(vertices1))
            vertex_indices2 = np.arange(len(vertices2))
            
            correspondences = find_vertex_correspondences_by_labels(
                features1, vertex_indices1, vertex_labels1,
                features2, vertex_indices2, vertex_labels2,
                vertices1, vertices2,
                bidirectional_consistency, use_global_optimization
            )
        else:
            # Standard matching without keypoint constraints
            if use_global_optimization:
                correspondences = self._global_optimization_matching(features1, features2, vertices1, vertices2)
            else:
                correspondences = self._nearest_neighbor_matching(features1, features2, vertices1, vertices2, bidirectional_consistency)
        
        return correspondences
    
    def _nearest_neighbor_matching(self, features1, features2, vertices1, vertices2, bidirectional_consistency):
        """Standard nearest neighbor matching"""
        from sklearn.neighbors import NearestNeighbors
        
        # Forward matching
        nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(features2)
        distances, indices = nbrs.kneighbors(features1)
        
        correspondences = []
        for i in range(len(features1)):
            j = indices[i, 0]
            dist = distances[i, 0]
            correspondences.append((i, j, dist))
        
        if bidirectional_consistency:
            # Backward matching for consistency check
            nbrs_back = NearestNeighbors(n_neighbors=1, metric='cosine').fit(features1)
            distances_back, indices_back = nbrs_back.kneighbors(features2)
            
            consistent_correspondences = []
            for i, j, dist in correspondences:
                if indices_back[j, 0] == i:
                    consistent_correspondences.append((i, j, dist))
            
            correspondences = consistent_correspondences
        
        return correspondences
    
    def _global_optimization_matching(self, features1, features2, vertices1, vertices2):
        """Global optimization matching using Hungarian algorithm"""
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        distance_matrix = cdist(features1, features2, metric='cosine')
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        correspondences = []
        for i, j in zip(row_indices, col_indices):
            correspondences.append((i, j, distance_matrix[i, j]))
        
        return correspondences
    
    def match_meshes(self, mesh1, mesh2, mesh1_name="mesh1", mesh2_name="mesh2",
                    bidirectional_consistency=True, use_global_optimization=False,
                    keypoints1_path=None, keypoints2_path=None, force_rerender=False):
        """Match two meshes using DINO features"""
        
        print(f"Matching meshes: {mesh1_name} ↔ {mesh2_name}")
        
        # Extract features for both meshes
        features1_result = self.extract_dino_features(mesh1, mesh1_name, force_rerender)
        features2_result = self.extract_dino_features(mesh2, mesh2_name, force_rerender)
        
        features1 = features1_result['features']
        features2 = features2_result['features']
        
        # Load keypoints if available
        keypoints1, keypoint_indices1 = self.load_keypoints(mesh1, keypoints1_path, mesh1_name)
        keypoints2, keypoint_indices2 = self.load_keypoints(mesh2, keypoints2_path, mesh2_name)
        
        # Find correspondences
        correspondences = self.find_correspondences(
            features1, features2, mesh1.vertices, mesh2.vertices,
            bidirectional_consistency, use_global_optimization,
            keypoints1, keypoints2, mesh1, mesh2
        )
        
        print(f"Found {len(correspondences)} correspondences")
        
        return correspondences, features1_result, features2_result
    
    def _trimesh_to_o3d_normalized(self, mesh):
        """Convert trimesh to Open3D mesh with normalization"""
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh
    
    def save_correspondences(self, correspondences, features1_dict, features2_dict, output_file):
        """Save correspondences to JSON file"""
        
        output_data = {
            'correspondences': [
                {
                    'source_vertex': int(src),
                    'target_vertex': int(tgt),
                    'distance': float(dist)
                } for src, tgt, dist in correspondences
            ],
            'metadata': {
                'num_correspondences': len(correspondences),
                'source_features_shape': list(features1_dict['features'].shape),
                'target_features_shape': list(features2_dict['features'].shape),
                'dino_model': self.dino_model,
                'timestamp': datetime.now().isoformat(),
                'source_coverage': float(features1_dict.get('coverage', 0)),
                'target_coverage': float(features2_dict.get('coverage', 0)),
                'source_consistency_loss': float(features1_dict.get('dino_consistency_loss', 0)),
                'target_consistency_loss': float(features2_dict.get('dino_consistency_loss', 0))
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Correspondences saved to: {output_file}")
