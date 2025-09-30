"""
Pipeline Utility Functions

Work directory setup, correspondence extraction, and summary functions.
"""

import os
import glob
import shutil
from datetime import datetime
from .cache import get_cache_manager


def extract_correspondences(pair_dir, source_name, target_name, output_json):
    """Extract final correspondence JSON from processing results"""
    
    # Search patterns in priority order: eval > final > iteration > dino
    search_patterns = [
        ("*eval_correspondences.json", "dense correspondence result"),
        ("correspondences_final.json", "ARAP correspondence result"),
        ("correspondences_iteration_*.json", "iterative correspondence result"),
        (f"*{source_name}*to*{target_name}*correspondences.json", "DINO correspondence result"),
        ("*correspondences*.json", "correspondence result")
    ]
    
    for pattern, description in search_patterns:
        files = glob.glob(os.path.join(pair_dir, "**", pattern), recursive=True)
        if files:
            # For iteration files, use the last one (highest iteration)
            if "iteration" in pattern:
                files.sort()
                corres_file = files[-1]
            else:
                corres_file = files[0]
            
            shutil.copy2(corres_file, output_json)
            return
    
    raise RuntimeError(f"No correspondence files found in {pair_dir}")


def print_summary(source_mesh, target_mesh, output_json, work_dir):
    """Print pipeline completion summary"""
    
    print("\nPipeline completed successfully")
    print("=" * 50)
    print(f"Source: {os.path.basename(source_mesh)}")
    print(f"Target: {os.path.basename(target_mesh)}")
    print(f"Output: {output_json}")
    
    if os.path.exists(output_json):
        # Get file size
        file_size = os.path.getsize(output_json)
        file_size_mb = file_size / (1024 * 1024)
        
        # Get cache size
        cache_manager = get_cache_manager('./cache')
        cache_stats = cache_manager.get_cache_stats()
        cache_size_mb = cache_stats['total_size_mb']
        
        print(f"Output size: {file_size_mb:.1f}MB")
        print(f"Cache size: {cache_size_mb:.0f}MB")
        print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
    else:
        print("Error: Output file was not created")
    
    print("=" * 50)
