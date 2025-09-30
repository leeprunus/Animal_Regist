"""
DINO Feature Extraction Pipeline Function
"""

from src.utils import get_cache_manager
from src.utils.config import get_config
from src.features import DinoFeatureExtractor, process_all_models


def run_dino_extraction(models_dir, specific_files=None):
    """Run DINO visual feature extraction on models in directory"""
    
    # Initialize cache manager
    cache_manager = get_cache_manager("./cache")
    
    # Get configuration
    config = get_config()
    dino_config = config.get_dino_config()
    
    # Initialize DINO feature extractor with config parameters
    feature_extractor = DinoFeatureExtractor(
        dino_model=dino_config['model_name'],
        dino_image_size=dino_config['image_size'],
        dino_pca_dim=dino_config['pca_dim'],
        num_views=dino_config['num_views'],
        render_width=dino_config['render_width'],
        render_height=dino_config['render_height'],
        consistency_weight=dino_config['consistency_weight'],
        device=dino_config['device'],
        cache_manager=cache_manager
    )
    
    if specific_files:
        # Process only specific files
        from pathlib import Path
        from src.features.dino_extractor import process_single_model
        successful_models = 0
        for file_path in specific_files:
            if Path(file_path).exists() and file_path.endswith('.obj'):
                model_info = {
                    'name': Path(file_path).stem,
                    'path': file_path,
                    'class': 'animal'  # Default class
                }
                try:
                    success = process_single_model(model_info, feature_extractor, cache_manager)
                    if success:
                        successful_models += 1
                except Exception as e:
                    print(f"Warning: Failed to process DINO features for {file_path}: {e}")
    else:
        # Process all models in directory
        successful_models = process_all_models(
            models_dir=models_dir,
            cache_manager=cache_manager,
            feature_extractor=feature_extractor
        )
    
    if successful_models <= 0:
        raise RuntimeError("DINO feature extraction failed - no models processed")
    