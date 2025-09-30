"""
Keypoint detection functions for 3D models.

This module contains functions for finding model files and processing them 
to generate keypoints using the keypoints pipeline.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from src.keypoints.pipeline import KeypointsPipeline


def find_model_files(models_dir, extensions=('.obj', '.ply')):
    """
    Find all model files in the given directory and its subdirectories
    
    Args:
        models_dir: Directory containing model files
        extensions: Tuple of file extensions to look for
        
    Returns:
        List of model file paths
    """
    model_files = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise ValueError(f"Models directory {models_dir} does not exist")
    
    for ext in extensions:
        # Use rglob to search recursively through all subdirectories
        model_files.extend(list(models_path.rglob(f"*{ext}")))
    
    return sorted(model_files)


def process_single_model(model_file, cache_manager, pipeline_params):
    """
    Process a single model file and cache results
    
    Args:
        model_file: Path to input model
        cache_manager: Cache manager instance for storing results
        pipeline_params: Dictionary of pipeline parameters
        
    Returns:
        Tuple of (success, processing_time, error_message, num_keypoints)
    """
    start_time = time.time()
    
    try:
        # Check if already cached
        cached_result = cache_manager.load_keypoints(model_file)
        if cached_result and not pipeline_params.get('force_recompute', False):
            keypoints, info = cached_result
            processing_time = time.time() - start_time
            num_keypoints = len(keypoints) if keypoints is not None else 0
            return True, processing_time, "Cached", num_keypoints
        
        # Initialize pipeline
        pipeline = KeypointsPipeline(
            num_views=pipeline_params.get('num_views', 8),
            device=pipeline_params.get('device', None),
        )
        
        # Run pipeline (use temporary file since pipeline requires output_file)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            temp_output_file = tmp_file.name
        
        keypoints_3d = pipeline.run(
            input_mesh=str(model_file),
            output_file=temp_output_file,
            pose2d=pipeline_params.get('pose2d', 'animal'),
            num_eigenvalues=pipeline_params.get('num_eigenvalues', 80),
            time_samples=pipeline_params.get('time_samples', 40),
            curvature_threshold=pipeline_params.get('curvature_threshold', 0.8),
            hks_threshold=pipeline_params.get('hks_threshold', 0.0),
            min_distance_ratio=pipeline_params.get('min_distance_ratio', 0.1),
            max_vertices=pipeline_params.get('max_vertices', 20000),
            use_hks=pipeline_params.get('use_hks', True),
            add_spine_points=pipeline_params.get('add_spine_points', True),
            num_spine_points=pipeline_params.get('num_spine_points', 5)
        )
        
        # Remove temporary file and create metadata for caching
        try:
            if os.path.exists(temp_output_file):
                os.unlink(temp_output_file)
        except:
            pass
            
        if keypoints_3d is not None:
            info = {
                'num_keypoints': len(keypoints_3d),
                'processing_params': pipeline_params,
                'model_file': str(model_file),
                'processed_timestamp': time.time()
            }
            
            # Cache the results
            cache_manager.cache_keypoints(model_file, keypoints_3d, info)
        
        processing_time = time.time() - start_time
        num_keypoints = len(keypoints_3d) if keypoints_3d is not None else 0
        return True, processing_time, None, num_keypoints
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        return False, processing_time, error_msg, 0
