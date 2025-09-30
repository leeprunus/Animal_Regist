"""
Mesh Normalization Pipeline Function
"""

from src.utils import get_cache_manager
from src.mesh import process_all_meshes


def run_normalization(models_dir, specific_files=None):
    """Run mesh normalization on models in directory"""
    
    # Initialize cache manager
    cache_manager = get_cache_manager("./cache")
    
    if specific_files:
        # Process only specific files using single mesh processing
        from src.mesh.processor import ensure_mesh_processed
        from pathlib import Path
        successful_meshes = 0
        for file_path in specific_files:
            if Path(file_path).exists() and file_path.endswith('.obj'):
                try:
                    ensure_mesh_processed(file_path, cache_manager, quiet=True)
                    successful_meshes += 1
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")
    else:
        # Process all meshes in directory
        successful_meshes = process_all_meshes(
            models_dir=models_dir,
            extensions=('.obj',),
            backup_originals=False,
            cache_manager=cache_manager
        )
    
    if successful_meshes <= 0:
        raise RuntimeError("Mesh normalization failed - no meshes processed")
    