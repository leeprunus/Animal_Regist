"""
Keypoint Detection Pipeline Function
"""

from tqdm import tqdm
from src.utils import get_cache_manager
from src.utils.config import get_config
from src.keypoints import find_model_files, process_single_model


def run_keypoint_detection(models_dir, specific_files=None):
    """Run keypoint detection pipeline on models in directory"""
    
    # Initialize cache manager
    cache_manager = get_cache_manager("./cache")
    
    # Get configuration
    config = get_config()
    keypoints_config = config.get_keypoints_config()
    
    # Find model files
    if specific_files:
        # Process only specific files
        from pathlib import Path
        model_files = []
        for file_path in specific_files:
            if Path(file_path).exists():
                model_files.append(Path(file_path))
        if not model_files:
            raise RuntimeError(f"No valid model files found from specified files: {specific_files}")
    else:
        # Process all files in directory
        model_files = find_model_files(models_dir)
        if not model_files:
            raise RuntimeError(f"No model files found in {models_dir}")
    
    # Set up pipeline parameters from config
    pipeline_params = {
        'num_views': keypoints_config['num_views'],
        'device': keypoints_config['device'],
        'max_vertices': keypoints_config['max_vertices'],
        'num_eigenvalues': keypoints_config['num_eigenvalues'],
        'time_samples': keypoints_config['time_samples'],
        'curvature_threshold': keypoints_config['curvature_threshold'],
        'hks_threshold': keypoints_config['hks_threshold'],
        'min_distance_ratio': keypoints_config['min_distance_ratio'],
        'num_spine_points': keypoints_config['num_spine_points'],
        'add_spine_points': keypoints_config['add_spine_points'],
        'pose2d': keypoints_config['pose2d']
    }
    
    # Process models
    for model_file in model_files:
        # Check if already cached
        cached_result = cache_manager.load_keypoints(model_file)
        if cached_result:
            continue
        
        # Process the model
        success, proc_time, error_msg, num_keypoints = process_single_model(
            model_file, cache_manager, pipeline_params
        )
        
        if not success:
            raise RuntimeError(f"Failed to process {model_file.name}: {error_msg}")
    