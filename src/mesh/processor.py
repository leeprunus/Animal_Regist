#!/usr/bin/env python3
"""
Single Mesh Processor

This script processes a single mesh file from anywhere and caches all results
using the cache system. It can be used as a component in the larger pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np
import open3d as o3d

# Ensure current directory is in path for imports
sys.path.append(str(Path(__file__).parent))

from src.utils.cache import get_cache_manager

def process_mesh_keypoints(mesh_file, cache_manager):
    """Process keypoints for a single mesh using cached results"""
    print(f"Processing keypoints for: {mesh_file}")
    
    # Check if already cached
    cached_result = cache_manager.load_keypoints(mesh_file)
    if cached_result:
        keypoints, info = cached_result
        print(f"Loaded keypoints from cache: {keypoints.shape}")
        return keypoints, info
    
    print("Running keypoint detection...")
    
    # Import here to avoid import errors when modules not available
    try:
        from keypoints_pipeline import KeypointsPipeline
    except ImportError:
        print("Error: Keypoints pipeline not available")
        return None, None
    
    # Use keypoints pipeline
    pipeline = KeypointsPipeline()
    try:
        keypoints, info = pipeline.process_mesh(mesh_file)
        
        # Cache the results
        cache_manager.cache_keypoints(mesh_file, keypoints, info)
        return keypoints, info
        
    except Exception as e:
        print(f"Error: Keypoint detection failed: {e}")
        return None, None

def process_mesh_dino_features(mesh_file, cache_manager):
    """Process DINO features for a single mesh using cached results"""
    print(f"Processing DINO features for: {mesh_file}")
    
    # Check if already cached
    cached_result = cache_manager.load_dino_features(mesh_file)
    if cached_result:
        features, metadata = cached_result
        print(f"Loaded DINO features from cache: {features.shape}")
        return features, metadata
    
    print("Running DINO feature extraction...")
    
    # Import here to avoid import errors when modules not available
    try:
        from dino_features import extract_dino_features_single
    except ImportError:
        print("Error: DINO feature extraction not available")
        return None, None
    
    try:
        features, metadata = extract_dino_features_single(mesh_file)
        
        # Cache the results
        cache_manager.cache_dino_features(mesh_file, features, metadata)
        return features, metadata
        
    except Exception as e:
        print(f"Error: DINO feature extraction failed: {e}")
        return None, None

def ensure_mesh_processed(mesh_file, cache_manager, force_reprocess=False, quiet=False):
    """
    Ensure a mesh is fully processed (keypoints + DINO features)
    
    Args:
        mesh_file: Path to mesh file
        cache_manager: Cache manager instance  
        force_reprocess: Force reprocessing even if cached
        
    Returns:
        dict with paths to cached results
    """
    mesh_file = Path(mesh_file).resolve()
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    
    if not quiet:
        print(f"Processing mesh: {mesh_file}")
    
    results = {
        'mesh_file': str(mesh_file),
        'keypoints_file': None,
        'dino_features_file': None,
        'normalized_mesh_file': None
    }
    
    # Check for cached keypoints
    keypoints_cached = cache_manager.load_keypoints(mesh_file)
    if keypoints_cached and not force_reprocess:
        results['keypoints_file'] = str(cache_manager.get_keypoints_cache_path(mesh_file))
        if not quiet:
            print(f"Keypoints already cached")
    else:
        if not quiet:
            print(f"Loading keypoints from preprocessing...")
        # Check if keypoints exist in models directory from preprocessing
        mesh_name = mesh_file.stem
        models_keypoints = mesh_file.parent / f"{mesh_name}.npy"
        models_info = mesh_file.parent / f"{mesh_name}_info.json"
        
        if models_keypoints.exists() and models_info.exists():
            # Load and cache the keypoints
            keypoints = np.load(models_keypoints)
            with open(models_info, 'r') as f:
                info = json.load(f)
            cache_manager.save_keypoints(mesh_file, keypoints, info)
            results['keypoints_file'] = str(cache_manager.get_keypoints_cache_path(mesh_file))
            if not quiet:
                print(f"Keypoints loaded from preprocessing and cached")
        elif force_reprocess:
            # Try to reprocess if forced
            keypoints, info = process_mesh_keypoints(mesh_file, cache_manager, quiet=quiet)
            if keypoints is not None:
                results['keypoints_file'] = str(cache_manager.get_keypoints_cache_path(mesh_file))
    
    # Check for cached DINO features
    dino_cached = cache_manager.load_dino_features(mesh_file)
    if dino_cached and not force_reprocess:
        results['dino_features_file'] = str(cache_manager.get_dino_cache_path(mesh_file))
        if not quiet:
            print(f"DINO features already cached")
    else:
        if not quiet:
            print(f"Loading DINO features from preprocessing...")
        # Check multiple possible locations for DINO features
        mesh_name = mesh_file.stem
        models_dino = mesh_file.parent / f"{mesh_name}_dino_features.npy"
        models_dino_meta = mesh_file.parent / f"{mesh_name}_dino_metadata.json"
        
        # Also check unknown subdirectory (preprocessing creates this)
        unknown_dir = mesh_file.parent / "unknown"
        unknown_dino = unknown_dir / f"{mesh_name}_dino_features.npy" 
        unknown_dino_meta = unknown_dir / f"{mesh_name}_dino_metadata.json"
        
        dino_file = None
        dino_meta_file = None
        
        if models_dino.exists() and models_dino_meta.exists():
            dino_file, dino_meta_file = models_dino, models_dino_meta
        elif unknown_dino.exists() and unknown_dino_meta.exists():
            dino_file, dino_meta_file = unknown_dino, unknown_dino_meta
            
        if dino_file and dino_meta_file:
            # Load and cache the DINO features
            features = np.load(dino_file)
            with open(dino_meta_file, 'r') as f:
                metadata = json.load(f)
            cache_manager.save_dino_features(mesh_file, features, metadata)
            results['dino_features_file'] = str(cache_manager.get_dino_cache_path(mesh_file))
            if not quiet:
                print(f"DINO features loaded from preprocessing and cached")
        elif force_reprocess:
            # Try to reprocess if forced
            features, metadata = process_mesh_dino_features(mesh_file, cache_manager, quiet=quiet)
            if features is not None:
                results['dino_features_file'] = str(cache_manager.get_dino_cache_path(mesh_file))
    
    # Check for normalized mesh (may be created by normalization step)
    normalized_path = cache_manager.load_normalized_mesh(mesh_file)
    if normalized_path:
        results['normalized_mesh_file'] = str(normalized_path)
        if not quiet:
            print(f"Normalized mesh cached")
    else:
        # Cache the original mesh as normalized (preprocessing may have already normalized it)
        normalized_mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_file))
        normalized_path = cache_manager.save_normalized_mesh(mesh_file, normalized_mesh_o3d)
        results['normalized_mesh_file'] = str(normalized_path)
        if not quiet:
            print(f"Original mesh cached as normalized")
    
    return results


