#!/usr/bin/env python3
"""
Animal_Regist Cache Manager

This module provides a caching system for all Animal_Regist components.
All cached data is stored in a single cache directory and uses content-based 
hashing to ensure consistency regardless of input file paths.
"""

import os
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import tempfile
import shutil

class CacheManager:
    """Cache manager for all Animal_Regist data"""
    
    def __init__(self, cache_root: str = "./cache"):
        """
        Initialize cache manager
        
        Args:
            cache_root: Root directory for all cache data
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of cache
        self.keypoints_cache = self.cache_root / "keypoints"
        self.dino_cache = self.cache_root / "dino_features" 
        self.renders_cache = self.cache_root / "renders"
        self.geodesic_cache = self.cache_root / "geodesic"
        self.normalization_cache = self.cache_root / "normalization"
        
        # Create all cache subdirectories
        for cache_dir in [self.keypoints_cache, self.dino_cache, self.renders_cache, 
                         self.geodesic_cache, self.normalization_cache]:
            cache_dir.mkdir(exist_ok=True)
    
    def compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Compute content hash for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash of file content
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # Use first 16 chars for brevity
    
    def compute_content_hash(self, content: Union[str, bytes, Dict, List]) -> str:
        """
        Compute hash for arbitrary content
        
        Args:
            content: Content to hash (string, bytes, dict, list)
            
        Returns:
            SHA-256 hash of content
        """
        if isinstance(content, str):
            data = content.encode('utf-8')
        elif isinstance(content, bytes):
            data = content
        elif isinstance(content, (dict, list)):
            data = json.dumps(content, sort_keys=True).encode('utf-8')
        else:
            data = str(content).encode('utf-8')
            
        return hashlib.sha256(data).hexdigest()[:16]
    
    def get_keypoints_cache_path(self, mesh_file: Union[str, Path]) -> Path:
        """Get cache path for keypoints"""
        file_hash = self.compute_file_hash(mesh_file)
        return self.keypoints_cache / f"{file_hash}.npy"
    
    def get_keypoints_info_cache_path(self, mesh_file: Union[str, Path]) -> Path:
        """Get cache path for keypoints metadata"""
        file_hash = self.compute_file_hash(mesh_file)
        return self.keypoints_cache / f"{file_hash}_info.json"
    
    def cache_keypoints(self, mesh_file: Union[str, Path], keypoints: np.ndarray, info: Dict[str, Any]) -> Path:
        """
        Cache keypoints and metadata
        
        Args:
            mesh_file: Original mesh file path
            keypoints: Keypoint coordinates array
            info: Metadata dictionary
            
        Returns:
            Path to cached keypoints file
        """
        cache_path = self.get_keypoints_cache_path(mesh_file)
        info_path = self.get_keypoints_info_cache_path(mesh_file)
        
        # Save keypoints and info
        np.save(cache_path, keypoints)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        return cache_path
    
    def load_keypoints(self, mesh_file: Union[str, Path]) -> Optional[tuple]:
        """
        Load cached keypoints
        
        Args:
            mesh_file: Original mesh file path
            
        Returns:
            Tuple of (keypoints, info) or None if not cached
        """
        cache_path = self.get_keypoints_cache_path(mesh_file)
        info_path = self.get_keypoints_info_cache_path(mesh_file)
        
        if cache_path.exists() and info_path.exists():
            keypoints = np.load(cache_path)
            with open(info_path, 'r') as f:
                info = json.load(f)
            return keypoints, info
        return None
    
    def save_keypoints(self, mesh_file: Union[str, Path], keypoints, info):
        """
        Save keypoints to cache
        
        Args:
            mesh_file: Original mesh file path
            keypoints: Keypoints array
            info: Keypoint metadata dict
            
        Returns:
            Tuple of paths (keypoints_path, info_path)
        """
        cache_path = self.get_keypoints_cache_path(mesh_file)
        info_path = self.get_keypoints_info_cache_path(mesh_file)
        
        np.save(cache_path, keypoints)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        return cache_path, info_path
    
    def get_dino_cache_path(self, mesh_file: Union[str, Path]) -> Path:
        """Get cache path for DINO features"""
        file_hash = self.compute_file_hash(mesh_file)
        return self.dino_cache / f"{file_hash}_dino_features.npy"
    
    def get_dino_metadata_cache_path(self, mesh_file: Union[str, Path]) -> Path:
        """Get cache path for DINO metadata"""
        file_hash = self.compute_file_hash(mesh_file)
        return self.dino_cache / f"{file_hash}_dino_metadata.json"
    
    def cache_dino_features(self, mesh_file: Union[str, Path], features: np.ndarray, metadata: Dict[str, Any]) -> Path:
        """
        Cache DINO features and metadata
        
        Args:
            mesh_file: Original mesh file path
            features: DINO feature array
            metadata: Metadata dictionary
            
        Returns:
            Path to cached features file
        """
        cache_path = self.get_dino_cache_path(mesh_file)
        metadata_path = self.get_dino_metadata_cache_path(mesh_file)
        
        # Save features and metadata
        np.save(cache_path, features)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return cache_path
    
    def load_dino_features(self, mesh_file: Union[str, Path]) -> Optional[tuple]:
        """
        Load cached DINO features
        
        Args:
            mesh_file: Original mesh file path
            
        Returns:
            Tuple of (features, metadata) or None if not cached
        """
        cache_path = self.get_dino_cache_path(mesh_file)
        metadata_path = self.get_dino_metadata_cache_path(mesh_file)
        
        if cache_path.exists() and metadata_path.exists():
            features = np.load(cache_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return features, metadata
        return None
    
    def save_dino_features(self, mesh_file: Union[str, Path], features, metadata):
        """
        Save DINO features to cache
        
        Args:
            mesh_file: Original mesh file path
            features: DINO features array
            metadata: DINO metadata dict
            
        Returns:
            Tuple of paths (features_path, metadata_path)
        """
        cache_path = self.get_dino_cache_path(mesh_file)
        metadata_path = self.get_dino_metadata_cache_path(mesh_file)
        
        np.save(cache_path, features)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return cache_path, metadata_path
    
    def get_renders_cache_key(self, mesh_file: Union[str, Path], render_config: Dict[str, Any]) -> str:
        """Get cache key for rendered views"""
        file_hash = self.compute_file_hash(mesh_file)
        config_hash = self.compute_content_hash(render_config)
        return f"{file_hash}_{config_hash}"
    
    def get_renders_cache_path(self, cache_key: str) -> Path:
        """Get cache directory for rendered views"""
        return self.renders_cache / cache_key
    
    def cache_renders(self, mesh_file: Union[str, Path], render_config: Dict[str, Any], 
                     render_data: Dict[str, Any]) -> Path:
        """
        Cache rendered views
        
        Args:
            mesh_file: Original mesh file path
            render_config: Rendering configuration
            render_data: Dictionary containing render results
            
        Returns:
            Path to cached render directory
        """
        cache_key = self.get_renders_cache_key(mesh_file, render_config)
        cache_path = self.get_renders_cache_path(cache_key)
        cache_path.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_path = cache_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'render_config': render_config,
                'render_data': render_data
            }, f, indent=2)
            
        print(f"Cached renders: {cache_path}")
        return cache_path
    
    def load_renders(self, mesh_file: Union[str, Path], render_config: Dict[str, Any]) -> Optional[tuple]:
        """
        Load cached renders
        
        Args:
            mesh_file: Original mesh file path  
            render_config: Rendering configuration
            
        Returns:
            Tuple of (cache_path, render_data) or None if not cached
        """
        cache_key = self.get_renders_cache_key(mesh_file, render_config)
        cache_path = self.get_renders_cache_path(cache_key)
        metadata_path = cache_path / "metadata.json"
        
        if cache_path.exists() and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return cache_path, metadata['render_data']
        return None
    
    def get_geodesic_cache_path(self, mesh_file: Union[str, Path]) -> Path:
        """Get cache path for geodesic distances"""
        file_hash = self.compute_file_hash(mesh_file)
        return self.geodesic_cache / f"geodesic_{file_hash}.pkl"
    
    def cache_geodesic(self, mesh_file: Union[str, Path], geodesic_data: Any) -> Path:
        """
        Cache geodesic distance matrix
        
        Args:
            mesh_file: Original mesh file path
            geodesic_data: Geodesic distance data to cache
            
        Returns:
            Path to cached geodesic file
        """
        import pickle
        cache_path = self.get_geodesic_cache_path(mesh_file)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(geodesic_data, f)
            
        return cache_path
    
    def load_geodesic(self, mesh_file: Union[str, Path]) -> Optional[Any]:
        """
        Load cached geodesic distances
        
        Args:
            mesh_file: Original mesh file path
            
        Returns:
            Cached geodesic data or None if not cached
        """
        import pickle
        cache_path = self.get_geodesic_cache_path(mesh_file)
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_normalization_cache_path(self, mesh_file: Union[str, Path]) -> Path:
        """Get cache path for normalized mesh"""
        file_hash = self.compute_file_hash(mesh_file)
        return self.normalization_cache / f"{file_hash}_normalized.obj"
    
    def cache_normalized_mesh(self, mesh_file: Union[str, Path], normalized_mesh_content: str) -> Path:
        """
        Cache normalized mesh
        
        Args:
            mesh_file: Original mesh file path
            normalized_mesh_content: Normalized mesh OBJ content
            
        Returns:
            Path to cached normalized mesh
        """
        cache_path = self.get_normalization_cache_path(mesh_file)
        
        with open(cache_path, 'w') as f:
            f.write(normalized_mesh_content)
            
        print(f"Cached normalized mesh: {cache_path}")
        return cache_path
    
    def load_normalized_mesh(self, mesh_file: Union[str, Path]) -> Optional[Path]:
        """
        Load cached normalized mesh
        
        Args:
            mesh_file: Original mesh file path
            
        Returns:
            Path to cached normalized mesh or None if not cached
        """
        cache_path = self.get_normalization_cache_path(mesh_file)
        
        if cache_path.exists():
            return cache_path
        return None
    
    def save_normalized_mesh(self, mesh_file: Union[str, Path], normalized_mesh_o3d) -> Path:
        """
        Save normalized mesh to cache
        
        Args:
            mesh_file: Path to original mesh file
            normalized_mesh_o3d: Open3D mesh object
            
        Returns:
            Path where normalized mesh was saved
        """
        import open3d as o3d
        cache_path = self.get_normalization_cache_path(mesh_file)
        
        # Save the normalized mesh
        o3d.io.write_triangle_mesh(str(cache_path), normalized_mesh_o3d)
        
        return cache_path
    
    def clear_cache(self):
        """Clear all cached data"""
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(exist_ok=True)
        print(f"Cleared cache directory: {self.cache_root}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        def count_files(directory):
            if directory.exists():
                return len(list(directory.rglob('*')))
            return 0
            
        def get_size(directory):
            if directory.exists():
                return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            return 0
        
        stats = {
            'cache_root': str(self.cache_root),
            'keypoints': {
                'files': count_files(self.keypoints_cache),
                'size_mb': get_size(self.keypoints_cache) / (1024*1024)
            },
            'dino_features': {
                'files': count_files(self.dino_cache),
                'size_mb': get_size(self.dino_cache) / (1024*1024)
            },
            'renders': {
                'files': count_files(self.renders_cache),
                'size_mb': get_size(self.renders_cache) / (1024*1024)
            },
            'geodesic': {
                'files': count_files(self.geodesic_cache),
                'size_mb': get_size(self.geodesic_cache) / (1024*1024)
            },
            'normalization': {
                'files': count_files(self.normalization_cache),
                'size_mb': get_size(self.normalization_cache) / (1024*1024)
            }
        }
        
        total_size = sum(category['size_mb'] for category in stats.values() if isinstance(category, dict))
        stats['total_size_mb'] = total_size
        
        return stats


# Global cache manager instance
_cache_manager = None

def get_cache_manager(cache_root: str = "./cache") -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_root)
    return _cache_manager


