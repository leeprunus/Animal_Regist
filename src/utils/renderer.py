#!/usr/bin/env python3
"""
Multi-View Rendering Module

This module provides multi-view rendering functionality for 3D meshes,
consolidating the rendering logic from both keypoints_pipeline.py and dino_features.py.
Includes caching to avoid repeated rendering and supports both basic circular views
and configurable camera positioning.

Features:
- Multi-view rendering interface
- Automatic caching to avoid repeated rendering
- Support for both keypoints and DINO feature extraction use cases
- Flexible camera positioning (circular, custom angles, etc.)
- Normalization and denormalization support
- Depth buffer and mask generation
- Image preprocessing options
"""

import os
import json
import math
import copy
import hashlib
import shutil
import tempfile
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class MultiViewRenderer:
    """
    Multi-view renderer with caching support for 3D mesh rendering
    """
    
    def __init__(self, cache_manager=None, enable_cache: bool = True, 
                 render_width: int = 256, render_height: int = 256):
        """
        Initialize the multi-view renderer
        
        Args:
            cache_manager: Cache manager instance
            enable_cache: Whether to enable caching of rendered views
            render_width: Width of rendered images
            render_height: Height of rendered images
        """
        self.enable_cache = enable_cache
        self.render_width = render_width
        self.render_height = render_height
        
        # Use cache manager if provided
        if cache_manager is not None:
            self.cache_manager = cache_manager
            self.use_cache = True
        else:
            # Use separate cache directory
            self.use_cache = False
            cache_dir = os.path.join(os.getcwd(), "cached_renders")
            self.cache_dir = Path(cache_dir)
            
            if self.enable_cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_mesh_hash(self, mesh_file: str) -> str:
        """Compute hash of mesh file for caching"""
        with open(mesh_file, 'rb') as f:
            file_content = f.read()
        return hashlib.md5(file_content).hexdigest()[:16]
    
    def _compute_render_params_hash(self, render_params: Dict) -> str:
        """Compute hash of rendering parameters for cache key"""
        # Custom JSON encoder for deterministic serialization
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif str(type(obj)).startswith('<class \'numpy.'):  # other numpy objects
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Create a deterministic string from render parameters
        param_str = json.dumps(render_params, sort_keys=True, default=json_serializer)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def _get_cache_key(self, mesh_file: str, render_params: Dict) -> str:
        """Generate cache key for rendered views"""
        mesh_hash = self._compute_mesh_hash(mesh_file)
        params_hash = self._compute_render_params_hash(render_params)
        return f"{mesh_hash}_{params_hash}"
    
    def _load_cached_renders(self, cache_key: str) -> Optional[Dict]:
        """Load cached rendered views if they exist"""
        if not self.enable_cache:
            return None
        
        # Use cache if available
        if self.use_cache:
            try:
                # We need mesh_file and render_config for cache
                # These will be passed from the render_multiview method
                if hasattr(self, '_current_mesh_file') and hasattr(self, '_current_render_config'):
                    cache_result = self.cache_manager.load_renders(self._current_mesh_file, self._current_render_config)
                    if cache_result is not None:
                        cache_path, render_data = cache_result
                        print(f"Loaded {len(render_data.get('views', {}))} cached views from cache")
                        return render_data
                else:
                    print("Warning: Cannot load from cache - mesh_file or render_config not set")
                return None
            except Exception as e:
                print(f"Error loading renders from cache: {e}")
                return None
        
        # Legacy cache system
        cache_path = self.cache_dir / cache_key
        metadata_file = cache_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Verify all expected files exist
            render_data = {}
            for view_id in metadata['view_ids']:
                view_dir = cache_path / f"view_{view_id}"
                image_file = view_dir / "image.png"
                mask_file = view_dir / "mask.png"
                depth_file = view_dir / "depth.npy"
                params_file = view_dir / "params.json"
                
                if not all([image_file.exists(), params_file.exists()]):
                    print(f"Warning: Incomplete cached data for view {view_id}, will re-render")
                    return None
                
                # Load view data
                image = plt.imread(str(image_file))
                with open(params_file, 'r') as f:
                    params = json.load(f)
                
                view_data = {
                    'image': image,
                    'intrinsic': np.array(params['intrinsic']),
                    'extrinsic': np.array(params['extrinsic']),
                    'camera_center': np.array(params['camera_center']),
                    'camera_front': np.array(params['camera_front']),
                    'azimuth': params['azimuth'],
                    'elevation': params['elevation']
                }
                
                # Load optional data if it exists
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    view_data['mask'] = mask > 127
                
                if depth_file.exists():
                    view_data['depth'] = np.load(str(depth_file))
                
                render_data[view_id] = view_data
            
            print(f"Loaded {len(render_data)} cached views for key: {cache_key}")
            return {
                'views': render_data,
                'normalization_params': metadata['normalization_params'],
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error loading cached renders: {e}")
            return None
    
    def _save_cached_renders(self, cache_key: str, render_data: Dict, 
                           normalization_params: Dict, metadata: Dict):
        """Save rendered views to cache"""
        if not self.enable_cache:
            return
        
        # Use cache if available
        if self.use_cache:
            try:
                # Prepare the data for cache
                data_to_cache = {
                    'views': render_data,
                    'normalization_params': normalization_params,
                    'metadata': {**metadata, 'cache_version': '1.0'}
                }
                
                # We need mesh_file and render_config for cache
                if hasattr(self, '_current_mesh_file') and hasattr(self, '_current_render_config'):
                    self.cache_manager.cache_renders(self._current_mesh_file, self._current_render_config, data_to_cache)
                    print(f"Saved {len(render_data)} views to cache")
                else:
                    print("Warning: Cannot save to cache - mesh_file or render_config not set")
                return
                
            except Exception as e:
                return
        
        # Legacy cache system
        try:
            cache_path = self.cache_dir / cache_key
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Save each view
            view_ids = []
            for view_id, view_data in render_data.items():
                view_dir = cache_path / f"view_{view_id}"
                view_dir.mkdir(parents=True, exist_ok=True)
                
                # Save image
                image_file = view_dir / "image.png"
                plt.imsave(str(image_file), view_data['image'])
                
                # Save mask if available
                if 'mask' in view_data:
                    mask_file = view_dir / "mask.png"
                    mask_uint8 = (view_data['mask'] * 255).astype(np.uint8)
                    cv2.imwrite(str(mask_file), mask_uint8)
                
                # Save depth if available
                if 'depth' in view_data:
                    depth_file = view_dir / "depth.npy"
                    np.save(str(depth_file), view_data['depth'])
                
                # Save camera parameters
                params_file = view_dir / "params.json"
                params = {
                    'intrinsic': view_data['intrinsic'].tolist(),
                    'extrinsic': view_data['extrinsic'].tolist(),
                    'camera_center': view_data['camera_center'].tolist(),
                    'camera_front': view_data['camera_front'].tolist(),
                    'azimuth': float(view_data['azimuth']),
                    'elevation': float(view_data['elevation'])
                }
                with open(params_file, 'w') as f:
                    json.dump(params, f, indent=2)
                
                view_ids.append(view_id)
            
            # Save metadata with custom JSON encoder
            metadata_complete = {
                **metadata,
                'view_ids': view_ids,
                'normalization_params': normalization_params,
                'cache_version': '1.0'
            }
            
            # Custom JSON encoder to handle numpy and datetime objects
            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                elif str(type(obj)).startswith('<class \'numpy.'):  # other numpy objects
                    return str(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            metadata_file = cache_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_complete, f, indent=2, default=json_serializer)
            
            print(f"Saved {len(render_data)} views to cache: {cache_key}")
            
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def normalize_rendered_image(self, image_np: np.ndarray, 
                                clahe_clip_limit: float = 2.0, 
                                clahe_tile_size: int = 8) -> np.ndarray:
        """
        Normalize rendered image for feature extraction
        
        Args:
            image_np: Input image as numpy array
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile grid size
            
        Returns:
            Normalized image (always RGB, 3 channels)
        """
        # Convert RGBA to RGB if necessary
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]  # Drop alpha channel
        
        image_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8) if image_np.dtype != np.uint8 else image_np
        
        # Apply CLAHE to each channel
        normalized_channels = []
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
        for i in range(3):
            normalized_channels.append(clahe.apply(image_uint8[:, :, i]))
        
        normalized_image = np.stack(normalized_channels, axis=2)
        normalized_image = cv2.bilateralFilter(normalized_image, 5, 50, 50)
        
        return normalized_image.astype(np.float32) / 255.0
    
    def generate_view_angles(self, num_views: int, view_type: str = "circular", 
                           custom_angles: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """
        Generate camera view angles
        
        Args:
            num_views: Number of views to generate
            view_type: Type of view arrangement ("circular", "hemisphere", "custom")
            custom_angles: List of (azimuth, elevation) tuples for custom views
            
        Returns:
            List of (azimuth, elevation) angle pairs in degrees
        """
        if view_type == "custom" and custom_angles:
            return custom_angles[:num_views]
        elif view_type == "hemisphere":
            # Generate views on hemisphere for coverage
            angles = []
            elevation_levels = max(1, int(np.sqrt(num_views)))
            azimuth_per_level = max(1, num_views // elevation_levels)
            
            for elev_idx in range(elevation_levels):
                elevation = (elev_idx + 1) * 60 / elevation_levels  # 0 to 60 degrees
                for az_idx in range(azimuth_per_level):
                    azimuth = az_idx * 360 / azimuth_per_level
                    angles.append((azimuth, elevation))
                    
                    if len(angles) >= num_views:
                        break
                if len(angles) >= num_views:
                    break
            
            return angles[:num_views]
        else:  # "circular" - default
            # Generate views in a circle around the model (keypoints_pipeline style)
            angles = []
            for i in range(num_views):
                azimuth = 360.0 * i / num_views
                elevation = 0.0  # Keep at horizon level
                angles.append((azimuth, elevation))
            return angles
    
    def render_single_view(self, mesh: o3d.geometry.TriangleMesh, 
                          center: np.ndarray, radius: float,
                          azimuth: float, elevation: float,
                          zoom_factor: float = 0.7,
                          generate_depth: bool = True,
                          generate_mask: bool = True,
                          enhance_image: bool = False) -> Dict:
        """
        Render a single view of the mesh
        
        Args:
            mesh: Open3D mesh to render
            center: Center point for camera focus
            radius: Distance multiplier for camera positioning
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            zoom_factor: Zoom factor for the camera
            generate_depth: Whether to generate depth buffer
            generate_mask: Whether to generate object mask
            enhance_image: Whether to apply image enhancement
            
        Returns:
            Dictionary with rendered data
        """
        mesh_to_render = copy.deepcopy(mesh)
        
        # Create visualizer with error handling
        vis = o3d.visualization.Visualizer()
        try:
            success = vis.create_window(visible=False, width=self.render_width, height=self.render_height)
            if not success:
                raise RuntimeError("Failed to create Open3D visualization window")
        except Exception as e:
            raise RuntimeError(f"OpenGL/Display context creation failed: {e}. "
                             f"This may occur in headless environments or with incompatible graphics drivers. "
                             f"Consider running in an environment with proper display support.")
        
        opt = vis.get_render_option()
        if opt is None:
            vis.destroy_window()
            raise RuntimeError("Failed to get render options - OpenGL context may not be properly initialized")
            
        # Photo-realistic rendering settings
        opt.light_on = True
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = False
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
        opt.background_color = [1.0, 1.0, 1.0]  # White background for masking
        
        # Set up lighting for realistic appearance (if available in this version)
        if hasattr(opt, 'light_ambient_color'):
            opt.light_ambient_color = [0.3, 0.3, 0.3]  # Soft ambient light
        if hasattr(opt, 'light_diffuse_color'):
            opt.light_diffuse_color = [0.7, 0.7, 0.7]  # Main diffuse lighting
        if hasattr(opt, 'light_specular_color'):
            opt.light_specular_color = [0.2, 0.2, 0.2]  # Subtle specular highlights
        if hasattr(opt, 'light_position'):
            opt.light_position = [1.0, 1.0, 1.0]  # Light from upper right
        
        # Improve mesh appearance
        if not mesh_to_render.has_vertex_colors():
            # Apply neutral gray color if no vertex colors
            mesh_to_render.paint_uniform_color([0.7, 0.7, 0.7])
        
        if not mesh_to_render.has_vertex_normals():
            mesh_to_render.compute_vertex_normals()
        
        vis.add_geometry(mesh_to_render)
        vis.poll_events()
        vis.update_renderer()
        view_ctl = vis.get_view_control()
        
        # Set camera position
        camera_distance = radius * 2.0  # Reduced from 2.5 to zoom in closer
        azimuth_rad, elevation_rad = azimuth * math.pi / 180, elevation * math.pi / 180
        
        x = camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = camera_distance * math.sin(elevation_rad)
        z = camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        
        cam_pos = center + np.array([x, y, z])
        front_vec = (center - cam_pos) / np.linalg.norm(center - cam_pos)
        
        view_ctl.set_lookat(center)
        view_ctl.set_front(front_vec)
        view_ctl.set_up(np.array([0, -1, 0]))  # Changed from [0, 1, 0] to rotate camera upside down
        view_ctl.set_zoom(zoom_factor * 1.3)  # Increased zoom factor to zoom in closer
        
        vis.poll_events()
        vis.update_renderer()
        
        # Get camera parameters
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()
        
        # Render image
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        
        # Convert RGBA to RGB if necessary (Open3D returns RGBA by default)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]  # Drop alpha channel
        
        # Enhance image if requested
        if enhance_image:
            image_np = self.normalize_rendered_image(image_np)
        
        result = {
            'image': image_np,
            'intrinsic': cam_params.intrinsic.intrinsic_matrix,
            'extrinsic': cam_params.extrinsic,
            'camera_center': cam_pos,
            'camera_front': front_vec,
            'azimuth': azimuth,
            'elevation': elevation
        }
        
        # Generate depth buffer if requested
        if generate_depth:
            depth = vis.capture_depth_float_buffer(do_render=True)
            result['depth'] = np.asarray(depth)
        
        # Generate mask if requested
        if generate_mask:
            if generate_depth and 'depth' in result:
                # Use depth buffer for mask generation
                depth_np = result['depth']
                mask = (depth_np < np.inf) & (depth_np > 0)
            else:
                # Fallback: use color-based mask (object vs white background)
                # Convert to grayscale and threshold against white background
                gray = np.mean(image_np, axis=2)
                mask = gray < 0.95  # Anything not pure white is object
            
            # Process mask with morphological operations
            mask_uint8 = (mask * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            
            # Fill holes in the mask
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Fill the largest contour (assuming it's the main object)
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask_uint8, [largest_contour], 255)
            
            result['mask'] = mask_uint8 > 127
            result['mask_uint8'] = mask_uint8  # Also provide as uint8 for saving
        
        vis.destroy_window()
        return result
    
    def normalize_mesh(self, mesh: Union[str, o3d.geometry.TriangleMesh]) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """
        Load and normalize mesh for consistent rendering
        
        Args:
            mesh: Path to mesh file or Open3D mesh object
            
        Returns:
            Tuple of (normalized_mesh, normalization_params)
        """
        if isinstance(mesh, str):
            # Load mesh from file
            o3d_mesh = o3d.io.read_triangle_mesh(mesh)
            if o3d_mesh.is_empty():
                raise ValueError(f"Could not load mesh from {mesh}")
        else:
            o3d_mesh = copy.deepcopy(mesh)
        
        # Store original parameters
        original_center = o3d_mesh.get_center()
        original_scale = np.linalg.norm(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound())
        
        # Normalize mesh
        o3d_mesh.translate(-original_center)
        o3d_mesh.scale(1.0/original_scale, center=np.array([0, 0, 0]))
        o3d_mesh.compute_vertex_normals()
        
        normalization_params = {
            "original_center": original_center.tolist(),
            "original_scale": float(original_scale)
        }
        
        return o3d_mesh, normalization_params
    
    def denormalize_points(self, points: np.ndarray, normalization_params: Dict) -> np.ndarray:
        """
        Convert points from normalized coordinate system back to original
        
        Args:
            points: Points in normalized coordinates
            normalization_params: Normalization parameters from normalize_mesh
            
        Returns:
            Points in original coordinate system
        """
        original_center = np.array(normalization_params["original_center"])
        original_scale = normalization_params["original_scale"]
        return points * original_scale + original_center
    
    def render_multiview(self, mesh_file: str, num_views: int = 12,
                        view_type: str = "circular",
                        custom_angles: List[Tuple[float, float]] = None,
                        zoom_factor: float = 0.7,
                        generate_depth: bool = True,
                        generate_mask: bool = True,
                        enhance_images: bool = False,
                        save_to_disk: bool = False,
                        output_dir: str = None) -> Dict:
        """
        Generate multi-view renderings of a 3D mesh with caching support
        
        Args:
            mesh_file: Path to input mesh file
            num_views: Number of views to generate
            view_type: Type of view arrangement ("circular", "hemisphere", "custom")
            custom_angles: List of (azimuth, elevation) tuples for custom views
            zoom_factor: Camera zoom factor
            generate_depth: Whether to generate depth buffers
            generate_mask: Whether to generate object masks
            enhance_images: Whether to apply image enhancement
            save_to_disk: Whether to save renders to disk (in addition to cache)
            output_dir: Directory to save renders to disk (if save_to_disk=True)
            
        Returns:
            Dictionary containing:
            - 'views': Dict of rendered views {view_id: view_data}
            - 'normalization_params': Mesh normalization parameters
            - 'metadata': Rendering metadata
        """
        
        # Create render parameters for cache key
        render_params = {
            'num_views': num_views,
            'view_type': view_type,
            'custom_angles': custom_angles,
            'zoom_factor': zoom_factor,
            'generate_depth': generate_depth,
            'generate_mask': generate_mask,
            'enhance_images': enhance_images,
            'render_width': self.render_width,
            'render_height': self.render_height
        }
        
        # Set temporary attributes for cache
        if self.use_cache:
            self._current_mesh_file = mesh_file
            self._current_render_config = render_params
        
        # Check cache
        cache_key = self._get_cache_key(mesh_file, render_params)
        cached_data = self._load_cached_renders(cache_key)
        if cached_data is not None:
            # Save to disk if requested, even when loading from cache
            if save_to_disk and output_dir:
                self._save_renders_to_disk(cached_data['views'], cached_data['normalization_params'], output_dir)
            
            return cached_data
        
        
        # Load and normalize mesh
        normalized_mesh, normalization_params = self.normalize_mesh(mesh_file)
        
        # Set up rendering parameters
        center = np.array([0, 0, 0])  # Mesh is already centered
        radius = 2.0
        
        # Generate view angles
        view_angles = self.generate_view_angles(num_views, view_type, custom_angles)
        
        # Render each view
        rendered_views = {}
        for i, (azimuth, elevation) in enumerate(view_angles):
            
            view_data = self.render_single_view(
                normalized_mesh, center, radius, azimuth, elevation,
                zoom_factor, generate_depth, generate_mask, enhance_images
            )
            
            rendered_views[i] = view_data
        
        # Create metadata
        from datetime import datetime
        metadata = {
            'mesh_file': os.path.abspath(mesh_file),
            'num_views': num_views,
            'view_type': view_type,
            'render_params': render_params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to cache
        self._save_cached_renders(cache_key, rendered_views, normalization_params, metadata)
        
        # Save to disk if requested
        if save_to_disk and output_dir:
            self._save_renders_to_disk(rendered_views, normalization_params, output_dir)
        
        result = {
            'views': rendered_views,
            'normalization_params': normalization_params,
            'metadata': metadata
        }
        
        return result
    
    def _save_renders_to_disk(self, rendered_views: Dict, normalization_params: Dict, output_dir: str):
        """Save rendered views to disk directory structure for keypoints pipeline compatibility"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save normalization parameters
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif str(type(obj)).startswith('<class \'numpy.'):  # other numpy objects
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_path / "normalization.json", 'w') as f:
            json.dump(normalization_params, f, indent=2, default=json_serializer)
        
        # Save each view in keypoints pipeline format
        for view_id, view_data in rendered_views.items():
            # Save image
            image_file = output_path / f"camera_view_{view_id}.png"
            plt.imsave(str(image_file), view_data['image'])
            
            # Save mask if available
            if 'mask_uint8' in view_data:
                mask_file = output_path / f"camera_view_{view_id}_mask.png"
                cv2.imwrite(str(mask_file), view_data['mask_uint8'])
            
            # Save depth if available
            if 'depth' in view_data:
                depth_file = output_path / f"camera_view_{view_id}_depth.png"
                depth_normalized = (view_data['depth'] / np.max(view_data['depth']) * 255).astype(np.uint8)
                cv2.imwrite(str(depth_file), depth_normalized)
            
            # Save camera parameters (keypoints_pipeline format)
            params = {
                "K": view_data['intrinsic'].tolist(),
                "extrinsic": view_data['extrinsic'].tolist(),
                "view_angle": view_data['azimuth'] * math.pi / 180,  # Convert to radians
                "normalized": True
            }
            params_file = output_path / f"camera_view_{view_id}_camparams.json"
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2)
        
        print(f"Saved {len(rendered_views)} views to disk: {output_path}")
        
        # Also create a symlink/reference to cache for DINO features compatibility
        cache_link = output_path / "cache_reference.txt"
        with open(cache_link, 'w') as f:
            f.write(f"Cached renders available at: {self.cache_dir}\n")
            f.write(f"Use this path for DINO feature extraction\n")
    
    def clear_cache(self, mesh_file: str = None):
        """
        Clear render cache
        
        Args:
            mesh_file: If provided, only clear cache for this specific mesh
        """
        if not self.enable_cache or not self.cache_dir.exists():
            return
        
        if mesh_file:
            # Clear cache for specific mesh
            mesh_hash = self._compute_mesh_hash(mesh_file)
            for cache_item in self.cache_dir.glob(f"{mesh_hash}_*"):
                if cache_item.is_dir():
                    shutil.rmtree(cache_item)
            print(f"Cleared cache for mesh: {mesh_file}")
        else:
            # Clear entire cache
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cleared entire render cache")


# Convenience functions for backward compatibility



def render_views_for_dino(mesh_file: str, num_views: int = 12, 
                         enhance_images: bool = True,
                         cache_manager=None) -> Tuple[Dict, Dict]:
    """
    Convenience function for DINO feature extraction multi-view rendering
    
    Args:
        mesh_file: Path to input mesh file
        num_views: Number of views to generate
        enhance_images: Whether to apply image enhancement
        cache_manager: Cache manager
        
    Returns:
        Tuple of (rendered_views_dict, normalization_params)
    """
    # Use cache manager if provided
    renderer = MultiViewRenderer(cache_manager=cache_manager)
    
    result = renderer.render_multiview(
        mesh_file=mesh_file,
        num_views=num_views,
        view_type="circular",
        zoom_factor=0.7,
        generate_depth=True,
        generate_mask=True,
        enhance_images=enhance_images,
        save_to_disk=False
    )
    
    return result['views'], result['normalization_params']





