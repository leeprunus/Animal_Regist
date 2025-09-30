"""
DINO Feature Extraction for 3D Models

This module processes 3D models and computes DINO visual features
using multi-view rendering and dense feature extraction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import open3d as o3d
import os
import copy
import json
import math
import hashlib
import glob
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2

# Import the rendering module
from src.utils.renderer import render_views_for_dino

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Import shared function
from ..utils.common import find_all_models


def process_single_model(model_info, feature_extractor, cache_manager):
    """
    Process a single model and compute DINO features using cache.
    
    Args:
        model_info: Dictionary with model information
        feature_extractor: DinoFeatureExtractor instance
        cache_manager: Cache manager instance
        
    Returns:
        True if successful, False otherwise
    """
    model_name = model_info['name']
    model_path = model_info['path']
    model_class = model_info['class']
    
    
    try:
        # Check if already cached
        cached_result = cache_manager.load_dino_features(model_path)
        if cached_result:
            features, metadata = cached_result
            return True
        
        # Load mesh
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        mesh = trimesh.load(model_path)
        
        # Extract DINO features
        features_result = feature_extractor.extract_dino_features(mesh, model_name)
        
        # Prepare output data
        features = features_result['features']
        if torch.is_tensor(features):
            features_array = features.cpu().numpy()
        else:
            features_array = features
        
        metadata = {
            'model_name': model_name,
            'model_class': model_class,
            'model_path': model_path,
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'feature_dim': features_array.shape[1],
            'dino_consistency_loss': float(features_result.get('consistency_loss', 0)),
            'dino_coverage': float(features_result.get('coverage', 0)),
            'extraction_parameters': {
                'dino_model': feature_extractor.feature_extractor_name if hasattr(feature_extractor, 'feature_extractor_name') else 'dinov2_vitb14',
                'dino_pca_dim': feature_extractor.dino_pca_dim,
                'num_views': feature_extractor.num_views,
                'image_size': feature_extractor.feature_extractor.image_size,
                'consistency_weight': feature_extractor.consistency_weight
            }
        }
        
        # Save features and metadata to cache
        cache_path = cache_manager.cache_dino_features(model_path, features_array, metadata)
        
        
        return True
        
    except Exception as e:
        return False


def process_all_models(models_dir, cache_manager, feature_extractor):
    """
    Process all models in the directory and compute DINO features using cache.
    
    Args:
        models_dir: Path to the models directory
        cache_manager: Cache manager instance
        feature_extractor: DinoFeatureExtractor instance
        
    Returns:
        Number of successfully processed models
    """
    # Find all models
    models = find_all_models(models_dir)
    if not models:
        print(f"No models found in {models_dir}")
        return 0
    
    successful_models = 0
    failed_models = []
    
    
    for i, model_info in enumerate(models, 1):
        try:
            success = process_single_model(model_info, feature_extractor, cache_manager)
            
            if success:
                successful_models += 1
            else:
                failed_models.append(model_info['name'])
            
        except Exception as e:
            failed_models.append(model_info['name'])
    
    
    return successful_models


class DenseFeatureExtractor:
    """DINO feature extractor with dense sampling and bilinear interpolation."""
    
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14",
                 image_size=448, device="cuda"):
        self.image_size = image_size
        self.device = device
        
        # Load DINO model (with caching)
        # Load model with suppressed output to avoid redundant cache messages
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name, force_reload=False)
        self.model = self.model.to(self.device).eval()
        
        # Get model properties
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
    
    def prepare_images_batch(self, images_list):
        """Prepare batch of images for DINO processing."""
        tensors = []
        grid_sizes = []
        resize_scales = []
        
        for img in images_list:
            if isinstance(img, np.ndarray):
                # Convert RGBA to RGB if necessary
                if img.shape[2] == 4:
                    img = img[:, :, :3]  # Drop alpha channel
                img = Image.fromarray((img * 255).astype(np.uint8) if img.dtype == np.float32 else img)
            
            # Ensure PIL image is RGB (not RGBA)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Apply transforms
            tensor = self.transform(img)
            
            # Calculate resize scale
            resize_scale = img.width / tensor.shape[2]
            
            # Crop to patch size
            height, width = tensor.shape[1:]
            cropped_width = width - width % self.patch_size
            cropped_height = height - height % self.patch_size
            tensor = tensor[:, :cropped_height, :cropped_width]
            
            grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
            
            tensors.append(tensor)
            grid_sizes.append(grid_size)
            resize_scales.append(resize_scale)
        
        # Stack into batch
        batch_tensor = torch.stack(tensors).to(self.device)
        
        return batch_tensor, grid_sizes, resize_scales
    
    def extract_dense_features_batch(self, image_batch):
        """Extract dense features from batch of images."""
        with torch.inference_mode():
            # Get intermediate layer features (dense patch features)
            features = self.model.get_intermediate_layers(image_batch, n=1)[0]
            # features shape: [batch_size, num_patches, embed_dim]
        
        return features.cpu()
    
    def bilinear_sample_features(self, features, pixel_coords, grid_size, resize_scale):
        """Sample features at pixel coordinates using bilinear interpolation."""
        # Convert features to spatial format: [embed_dim, grid_h, grid_w]
        grid_h, grid_w = grid_size
        features_spatial = features.reshape(grid_h, grid_w, -1).permute(2, 0, 1)
        features_spatial = features_spatial.unsqueeze(0)  # Add batch dimension
        
        # Convert pixel coordinates to grid coordinates
        # pixel_coords: [N, 2] in (u, v) format
        grid_coords = pixel_coords.clone()
        grid_coords[:, 0] = (grid_coords[:, 0] / resize_scale - self.patch_size/2) / self.patch_size  # u -> grid_x
        grid_coords[:, 1] = (grid_coords[:, 1] / resize_scale - self.patch_size/2) / self.patch_size  # v -> grid_y
        
        # Normalize to [-1, 1] for grid_sample
        grid_coords[:, 0] = 2.0 * grid_coords[:, 0] / (grid_w - 1) - 1.0
        grid_coords[:, 1] = 2.0 * grid_coords[:, 1] / (grid_h - 1) - 1.0
        
        # Reshape for grid_sample: [1, N, 1, 2]
        grid_coords = grid_coords.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
        
        # Sample features using bilinear interpolation
        sampled = F.grid_sample(features_spatial, grid_coords, 
                               mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Reshape output: [embed_dim, N] -> [N, embed_dim]
        sampled_features = sampled.squeeze(0).squeeze(2).transpose(0, 1)
        
        return sampled_features


class MultiViewConsistency:
    """Multi-view consistency module for feature alignment."""
    
    def __init__(self, consistency_weight=1.0):
        self.consistency_weight = consistency_weight
    
    def compute_consistency_loss(self, vertex_features_dict):
        """Compute multi-view consistency loss for overlapping vertices."""
        losses = []
        vertex_pairs = 0
        
        view_ids = list(vertex_features_dict.keys())
        
        for i in range(len(view_ids)):
            for j in range(i + 1, len(view_ids)):
                view_i, view_j = view_ids[i], view_ids[j]
                features_i = vertex_features_dict[view_i]
                features_j = vertex_features_dict[view_j]
                
                # Find common vertices
                common_vertices = set(features_i.keys()) & set(features_j.keys())
                
                if len(common_vertices) > 0:
                    # Extract features for common vertices
                    feat_i = torch.stack([features_i[v] for v in common_vertices])
                    feat_j = torch.stack([features_j[v] for v in common_vertices])
                    
                    # L2 consistency loss
                    loss = F.mse_loss(feat_i, feat_j)
                    losses.append(loss)
                    vertex_pairs += len(common_vertices)
        
        if losses:
            total_loss = sum(losses) / len(losses)
            return total_loss, vertex_pairs
        else:
            return torch.tensor(0.0), 0
    
    def weighted_feature_aggregation(self, vertex_features_dict, vertex_counts):
        """Aggregate features across views with visibility-based weighting."""
        all_vertices = set()
        for features in vertex_features_dict.values():
            all_vertices.update(features.keys())
        
        aggregated_features = {}
        
        for vertex_idx in all_vertices:
            vertex_features = []
            weights = []
            
            for view_id, features in vertex_features_dict.items():
                if vertex_idx in features:
                    vertex_features.append(features[vertex_idx])
                    weights.append(vertex_counts.get((view_id, vertex_idx), 1.0))
            
            if vertex_features:
                # Weight by visibility count
                features_tensor = torch.stack(vertex_features)
                weights_tensor = torch.tensor(weights, dtype=features_tensor.dtype, device=features_tensor.device)
                weights_tensor = weights_tensor / weights_tensor.sum()
                
                # Weighted average
                aggregated_features[vertex_idx] = torch.sum(
                    features_tensor * weights_tensor.unsqueeze(1), dim=0
                )
        
        return aggregated_features 


def project_3d_to_2d(point_3d, intrinsic, extrinsic):
    """Project 3D point to 2D image coordinates."""
    point_h = np.append(point_3d, 1.0)
    proj = intrinsic @ (extrinsic @ point_h)[:3]
    if proj[2] <= 0:
        return None, None, False
    return proj[0] / proj[2], proj[1] / proj[2], proj[2]


def get_visible_vertices(vertices, intrinsic, extrinsic, image_shape, depth_buffer):
    """Get vertices visible in rendered image using depth buffer."""
    visible_vertices, pixel_positions = [], []
    height, width = image_shape[:2]
    
    if depth_buffer is None:
        depth_buffer = np.full((height, width), np.inf)
    
    for i, vertex in enumerate(vertices):
        u, v, depth = project_3d_to_2d(vertex, intrinsic, extrinsic)
        
        if u is not None and 0 <= u < width and 0 <= v < height:
            pixel_u, pixel_v = int(round(u)), int(round(v))
            if 0 <= pixel_u < width and 0 <= pixel_v < height:
                if depth <= depth_buffer[pixel_v, pixel_u] + 0.01:  # depth tolerance
                    visible_vertices.append(i)
                    pixel_positions.append((u, v, depth))
    
    return visible_vertices, pixel_positions 


class DinoFeatureExtractor:
    """DINO feature extractor for individual 3D models."""
    
    def __init__(self, dino_model="dinov2_vitb14", dino_image_size=448, dino_pca_dim=32, 
                 num_views=12, render_width=800, render_height=600, consistency_weight=1.0, 
                 device=None, cache_manager=None):
        """Initialize DINO feature extractor."""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        
        # Set deterministic seeds
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Store parameters
        self.dino_pca_dim = dino_pca_dim
        self.num_views = num_views
        self.render_width = render_width
        self.render_height = render_height
        self.consistency_weight = consistency_weight
        self.feature_extractor_name = dino_model
        self.cache_manager = cache_manager
        
        # Initialize DINO feature extractor
        self.feature_extractor = DenseFeatureExtractor(model_name=dino_model, image_size=dino_image_size, device=self.device)
        
        # Initialize multi-view consistency module
        self.consistency_module = MultiViewConsistency(consistency_weight)
        
        # Initialize shared PCA model (will be fit on first model)
        self.pca_model = None
        
    
    def _compute_mesh_hash(self, mesh):
        """Compute mesh hash for caching."""
        vertices_bytes = mesh.vertices.tobytes()
        faces_bytes = mesh.faces.tobytes()
        return hashlib.md5(vertices_bytes + faces_bytes).hexdigest()[:16]
    
    def _interpolate_missing_features(self, vertex_features_matrix, vertices_with_features, vertices, 
                                      k_neighbors=8, max_distance_ratio=0.2):
        """
        Interpolate features for non-visible vertices using distance-weighted interpolation
        from nearby visible vertices.
        """
        from scipy.spatial import KDTree
        
        # Get vertices with and without features
        visible_vertices = vertices[vertices_with_features]
        visible_features = vertex_features_matrix[vertices_with_features]
        non_visible_indices = np.where(~vertices_with_features)[0]
        non_visible_vertices = vertices[non_visible_indices]
        
        if len(visible_vertices) == 0 or len(non_visible_vertices) == 0:
            return vertex_features_matrix
        
        # Calculate bounding box diagonal for distance normalization
        bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
        max_distance = bbox_diagonal * max_distance_ratio
        
        
        # Build KDTree for nearest neighbor search
        kdtree = KDTree(visible_vertices)
        
        # For each non-visible vertex, find nearby visible vertices and interpolate
        
        for i, non_visible_idx in enumerate(non_visible_indices):
            non_visible_pos = non_visible_vertices[i]
            
            # Find k nearest visible vertices within max_distance
            distances, indices = kdtree.query(
                non_visible_pos, 
                k=min(k_neighbors, len(visible_vertices)),
                distance_upper_bound=max_distance
            )
            
            # Filter out invalid distances (beyond max_distance)
            valid_mask = distances < np.inf
            if not np.any(valid_mask):
                # No nearby visible vertices found, use global average
                avg_features = np.mean(visible_features, axis=0)
                vertex_features_matrix[non_visible_idx] = avg_features
                continue
            
            valid_distances = distances[valid_mask]
            valid_indices = indices[valid_mask]
            
            # Compute weights using inverse distance weighting
            epsilon = 1e-8
            weights = 1.0 / (valid_distances + epsilon)
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Interpolate features using weighted average
            nearby_features = visible_features[valid_indices]
            interpolated_feature = np.sum(nearby_features * weights[:, np.newaxis], axis=0)
            vertex_features_matrix[non_visible_idx] = interpolated_feature
        
        
        return vertex_features_matrix
    
    def extract_dino_features(self, mesh, mesh_name="mesh"):
        """Extract DINO visual features using multi-view rendering with bilinear interpolation."""
        
        # Save mesh temporarily if it's a trimesh object
        if hasattr(mesh, 'vertices'):
            temp_mesh_file = f"temp_{mesh_name}.obj"
            mesh.export(temp_mesh_file)
            cleanup_temp_file = True
        else:
            temp_mesh_file = mesh
            cleanup_temp_file = False
        
        try:
            # Use the renderer from render.py
            rendered_views, normalization_params = render_views_for_dino(
                mesh_file=temp_mesh_file,
                num_views=self.num_views,
                enhance_images=True,
                cache_manager=self.cache_manager
            )
            
            
            # Convert to expected format for feature extraction
            rendered_data = []
            for view_idx in range(len(rendered_views)):
                if view_idx not in rendered_views:
                    continue
                    
                view_data = rendered_views[view_idx]
                # The renderer already provides processed images
                view_data['processed_image'] = view_data['image']
                rendered_data.append(view_data)
            
            # Extract vertices for processing
            if hasattr(mesh, 'vertices'):
                vertices = mesh.vertices
            else:
                # Load mesh to get vertices
                temp_mesh = trimesh.load(temp_mesh_file)
                vertices = temp_mesh.vertices
            
            # Apply normalization to vertices to match rendered views
            original_center = np.array(normalization_params["original_center"])
            original_scale = normalization_params["original_scale"]
            normalized_vertices = (vertices - original_center) / original_scale
            
            # Batch feature extraction
            images_batch = [data['image'] for data in rendered_data]
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(4, len(images_batch))
            all_features = []
            
            for i in range(0, len(images_batch), batch_size):
                batch_images = images_batch[i:i+batch_size]
                batch_tensor, grid_sizes, resize_scales = self.feature_extractor.prepare_images_batch(batch_images)
                batch_features = self.feature_extractor.extract_dense_features_batch(batch_tensor)
                
                for j, features in enumerate(batch_features):
                    all_features.append((features, grid_sizes[j], resize_scales[j]))
            
            # Extract vertex features for each view with bilinear interpolation
            vertex_features_per_view = {}
            vertex_counts = {}
            
            for view_idx, ((features, grid_size, resize_scale), render_data) in enumerate(zip(all_features, rendered_data)):
                
                # Get visible vertices
                visible_vertices, pixel_positions = get_visible_vertices(
                    normalized_vertices, render_data['intrinsic'], render_data['extrinsic'],
                    render_data['image'].shape, render_data.get('depth')
                )
                
                if len(visible_vertices) == 0:
                    continue
                
                # Convert to tensor for bilinear sampling
                pixel_coords = torch.tensor([[u, v] for u, v, _ in pixel_positions], dtype=torch.float32)
                
                # Sample features using bilinear interpolation
                sampled_features = self.feature_extractor.bilinear_sample_features(
                    features, pixel_coords, grid_size, resize_scale
                )
                
                # Store features per vertex
                view_vertex_features = {}
                for i, vertex_idx in enumerate(visible_vertices):
                    view_vertex_features[vertex_idx] = sampled_features[i]
                    vertex_counts[(view_idx, vertex_idx)] = 1.0
                
                vertex_features_per_view[view_idx] = view_vertex_features
            
            # Compute multi-view consistency loss
            consistency_loss, vertex_pairs = self.consistency_module.compute_consistency_loss(vertex_features_per_view)
            
            # Aggregate features with consistency weighting
            aggregated_features = self.consistency_module.weighted_feature_aggregation(
                vertex_features_per_view, vertex_counts
            )
            
            # Convert to numpy for PCA
            n_vertices = len(vertices)
            vertex_features_matrix = np.zeros((n_vertices, self.feature_extractor.embed_dim))
            vertices_with_features = np.zeros(n_vertices, dtype=bool)
            
            for vertex_idx, feature in aggregated_features.items():
                vertex_features_matrix[vertex_idx] = feature.numpy()
                vertices_with_features[vertex_idx] = True
            
            
            # Handle vertices without features
            if np.sum(vertices_with_features) == 0:
                raise ValueError("No vertices have DINO features!")
            
            if np.sum(vertices_with_features) < n_vertices:
                vertex_features_matrix = self._interpolate_missing_features(
                    vertex_features_matrix, vertices_with_features, vertices
                )
            
            # PCA dimensionality reduction
            if self.pca_model is None:
                from sklearn.decomposition import PCA
                self.pca_model = PCA(n_components=self.dino_pca_dim)
                vertex_features_pca = self.pca_model.fit_transform(vertex_features_matrix)
            else:
                vertex_features_pca = self.pca_model.transform(vertex_features_matrix)
            
            # Normalize features
            dino_features = torch.tensor(vertex_features_pca, dtype=torch.float32, device=self.device)
            dino_features_normalized = F.normalize(dino_features, p=2, dim=1)
            
            
            return {
                'features': dino_features_normalized,
                'consistency_loss': consistency_loss,
                'coverage': np.sum(vertices_with_features) / n_vertices
            }
            
        finally:
            # Remove temporary mesh file if we created one
            if cleanup_temp_file and os.path.exists(temp_mesh_file):
                os.remove(temp_mesh_file)
