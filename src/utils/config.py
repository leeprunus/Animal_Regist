"""
Configuration Management

Configuration system for the AniCorres pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class ConfigManager:
    """Centralized configuration manager for the AniCorres pipeline."""
    
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self._config = None
        self._alignment_config = None
        
        # Load main config
        self.load_config()
        
        # Auto-detect device if needed
        self._resolve_device_settings()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _resolve_device_settings(self) -> None:
        """Resolve automatic device detection."""
        device_config = self._config.get('device', {})
        compute_device = device_config.get('compute_device', 'auto')
        
        if compute_device == 'auto':
            if torch.cuda.is_available() and device_config.get('use_gpu', True):
                self._config['device']['compute_device'] = 'cuda'
            else:
                self._config['device']['compute_device'] = 'cpu'
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'dino.model.model_name')
            default: Default value if key is not found
            
        Returns:
            Configuration value
            
        Example:
            config.get('dino.model.model_name')  # returns 'dinov2_vitb14'
            config.get('keypoints.hks.num_eigenvalues', 80)  # returns 80 or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_device(self) -> str:
        """Get the configured compute device."""
        return self.get('device.compute_device', 'cpu')
    
    def get_dino_config(self) -> Dict[str, Any]:
        """Get DINO feature extraction configuration."""
        return {
            'repo_name': self.get('dino.model.repo_name', 'facebookresearch/dinov2'),
            'model_name': self.get('dino.model.model_name', 'dinov2_vitb14'),
            'image_size': self.get('dino.model.image_size', 448),
            'force_reload': self.get('dino.model.force_reload', False),
            'pca_dim': self.get('dino.features.pca_dim', 32),
            'consistency_weight': self.get('dino.features.consistency_weight', 1.0),
            'num_views': self.get('dino.rendering.num_views', 16),
            'render_width': self.get('dino.rendering.render_width', 1024),
            'render_height': self.get('dino.rendering.render_height', 1024),
            'zoom_factor': self.get('dino.rendering.zoom_factor', 0.7),
            'view_type': self.get('dino.rendering.view_type', 'circular'),
            'knn_neighbors': self.get('dino.matching.knn_neighbors', 5),
            'bidirectional_consistency': self.get('dino.matching.bidirectional_consistency', True),
            'distance_metric': self.get('dino.matching.distance_metric', 'cosine'),
            'device': self.get_device()
        }
    
    def get_keypoints_config(self) -> Dict[str, Any]:
        """Get keypoint detection configuration."""
        return {
            'num_views': self.get('keypoints.rendering.num_views', 16),
            'device': self.get_device() if self.get('keypoints.rendering.device') == 'auto' 
                     else self.get('keypoints.rendering.device', 'cpu'),
            'num_eigenvalues': self.get('keypoints.hks.num_eigenvalues', 80),
            'time_samples': self.get('keypoints.hks.time_samples', 40),
            'curvature_threshold': self.get('keypoints.hks.curvature_threshold', 0.8),
            'hks_threshold': self.get('keypoints.hks.hks_threshold', 0.0),
            'min_distance_ratio': self.get('keypoints.hks.min_distance_ratio', 0.1),
            'max_vertices': self.get('keypoints.hks.max_vertices', 50000),
            'use_hks': self.get('keypoints.hks.use_hks', True),
            'add_spine_points': self.get('keypoints.spine.add_spine_points', True),
            'num_spine_points': self.get('keypoints.spine.num_spine_points', 5),
            'pose2d': self.get('keypoints.pose2d', 'animal')
        }
    
    def get_mesh_config(self) -> Dict[str, Any]:
        """Get mesh processing configuration."""
        return {
            'max_vertices': self.get('mesh.processing.max_vertices', 100000),
            'simplify_threshold': self.get('mesh.processing.simplify_threshold', 0.01),
            'smooth_iterations': self.get('mesh.processing.smooth_iterations', 2),
            'scale_method': self.get('mesh.normalization.scale_method', 'bbox'),
            'center_method': self.get('mesh.normalization.center_method', 'centroid')
        }
    
    def get_deformation_config(self) -> Dict[str, Any]:
        """Get mesh deformation configuration."""
        return {
            'autorig_enabled': self.get('deformation.autorig.enabled', True),
            'blender_executable': self.get('deformation.autorig.blender_executable', 'blender'),
            'timeout_seconds': self.get('deformation.autorig.timeout_seconds', 300),
            'lbs_min_keypoints': self.get('deformation.lbs.min_keypoints_required', 3),
            'lbs_method': self.get('deformation.lbs.deformation_method', 'translation')
        }
    
    def get_alignment_config(self) -> Dict[str, Any]:
        """Get alignment configuration from config."""
        return {
            'default_paths': self.get('alignment.default_paths', {}),
            'weights': self.get('alignment.weights', {'arap': 1, 'correspondence': 1, 'smoothness': 0.5}),
            'parameters': self.get('alignment.parameters', {
                'correspondence_end_iteration': -1,
                'correspondence_update_frequency': 1,
                'correspondence_update_start_iteration': 1
            }),
            'filter_thresholds': self.get('alignment.filter_thresholds', {'consistency_threshold': 0.01}),
            'optimization': self.get('alignment.optimization', {
                'max_iterations': 5,
                'convergence_tolerance': 1.0e-06,
                'batch_size': 5000,
                'global_max_iterations': 20
            }),
            'output': self.get('alignment.output', {
                'save_frequency': 1,
                'final_mesh': 'aligned_model.obj'
            })
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            'directory': self.get('cache.directory', './cache'),
            'enable_caching': self.get('cache.enable_caching', True),
            'keypoints_dir': self.get('cache.subdirs.keypoints', 'keypoints'),
            'normalization_dir': self.get('cache.subdirs.normalization', 'normalization'),
            'dino_features_dir': self.get('cache.subdirs.dino_features', 'dino_features'),
            'geodesic_distances_dir': self.get('cache.subdirs.geodesic_distances', 'geodesic_distances')
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return {
            'results_dir': self.get('output.results_dir', './result'),
            'create_timestamped_dirs': self.get('output.create_timestamped_dirs', True),
            'mesh_format': self.get('output.mesh_format', 'obj'),
            'keypoints_format': self.get('output.keypoints_format', 'npy'),
            'correspondences_format': self.get('output.correspondences_format', 'json'),
            'save_intermediate_results': self.get('output.save_intermediate_results', True),
            'save_frequency': self.get('output.save_frequency', 1)
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return {
            'mge_enable': self.get('evaluation.mge.enable', True),
            'mge_max_distance': self.get('evaluation.mge.max_geodesic_distance', 1.0),
            'max_correspondences': self.get('evaluation.mge.max_correspondences', 5000),
            'consistency_threshold': self.get('evaluation.filtering.consistency_threshold', 0.01),
            'distance_threshold': self.get('evaluation.filtering.distance_threshold', 0.1)
        }
    
    
    def _deep_merge(self, base_dict: Dict, overlay_dict: Dict) -> None:
        """Deep merge overlay_dict into base_dict."""
        for key, value in overlay_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update a configuration value at runtime.
        
        Args:
            key_path: Dot-separated path to the configuration key
            value: New value to set
        """
        keys = key_path.split('.')
        config_section = self._config
        
        # Navigate to the parent section
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the final value
        config_section[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config (default: original config path)
        """
        save_path = output_path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


# Global configuration instance
_global_config: Optional[ConfigManager] = None


def get_config(config_path: str = "config/pipeline.yaml") -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(config_path)
    return _global_config


def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)."""
    global _global_config
    _global_config = None
