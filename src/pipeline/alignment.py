"""
ARAP Alignment Pipeline Function
"""

import os
from src.alignment import process_single_pair_alignment


def run_alignment(pair_dir, source_model, target_model, config, meshes_dir=None):
    """Run DINO feature matching and ARAP alignment"""
    
    # Use provided meshes directory or fall back to cache
    models_dir = meshes_dir if meshes_dir else './cache'
    
    # Process single pair directly - alignment functions will find files via find_model_files()
    successful_pairs = process_single_pair_alignment(
        models_dir=models_dir,
        pair_dir=pair_dir,
        source_model=source_model,
        target_model=target_model,
        config=config
    )
    
    if successful_pairs <= 0:
        raise RuntimeError("DINO + ARAP alignment failed - no pairs processed")
