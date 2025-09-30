"""
AutoRig Integration

Mesh deformation using Blender AutoRig with LBS alternative.
"""

import os
import csv
import subprocess
from src.autorig import AUTORIG_SCRIPT
from .lbs import lbs_mesh_deformation
from src.utils.cache import CacheManager


def run_autorig(pair_dir, source_model, target_model, source_mesh_path=None, target_mesh_path=None):
    """
    Run mesh deformation with Blender AutoRig or LBS alternative.
    
    Uses Blender AutoRig when available, otherwise uses Linear Blend Skinning.
    Works directly with cached files.
    
    Args:
        pair_dir: Output directory for the pair
        source_model: Source model name 
        target_model: Target model name
        source_mesh_path: Actual path to source mesh file (optional)
        target_mesh_path: Actual path to target mesh file (optional)
    """
    
    # Try Blender AutoRig first
    try:
        # Check if Blender is available
        result = subprocess.run(['which', 'blender'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError("Blender not found in PATH")
        
        print("    Attempting Blender AutoRig deformation...")
        
        # Determine the correct models directory
        if source_mesh_path and os.path.exists(source_mesh_path):
            # Extract the directory from the mesh path (should be result folder's meshes dir)
            models_dir = os.path.dirname(source_mesh_path)
        else:
            # Fallback to cache directory
            models_dir = './cache'
        
        print(f"    Using models directory: {models_dir}")
        
        # Run Blender AutoRig with direct model parameters
        env = os.environ.copy()
        env['SIMPLE_STRUCTURE'] = '1'  # Use basic directory structure
        cmd = [
            'blender', '--background', '--python', AUTORIG_SCRIPT, '--',
            '--models_dir', models_dir,
            '--pair_dir', pair_dir, 
            '--source_model', source_model,
            '--target_model', target_model
        ]
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Blender AutoRig failed (exit code {result.returncode}), using LBS alternative")
            raise RuntimeError("Blender AutoRig processing failed")
        
        print("    Blender AutoRig completed successfully")
        return
        
    except (FileNotFoundError, RuntimeError):
        pass  # Use LBS alternative
    
    # Use LBS approach
    print("    Blender AutoRig not available, using LBS alternative...")
    cache_manager = CacheManager()
    
    try:
        # Use provided mesh paths or construct from model names
        if source_mesh_path is None:
            original_source_mesh = f"./models/{source_model}.obj"
        else:
            original_source_mesh = source_mesh_path
            
        if target_mesh_path is None:
            original_target_mesh = f"./models/{target_model}.obj"
        else:
            original_target_mesh = target_mesh_path
        
        # Get cached mesh and keypoints files
        source_mesh_cache_path = cache_manager.get_normalization_cache_path(original_source_mesh)
        source_kp, _ = cache_manager.load_keypoints(original_source_mesh)
        target_kp, _ = cache_manager.load_keypoints(original_target_mesh)
        
        if source_kp is None or target_kp is None:
            raise ValueError(f"Keypoints not found in cache for models {source_model} -> {target_model}")
            
        if not os.path.exists(source_mesh_cache_path):
            raise ValueError(f"Cached normalized mesh not found: {source_mesh_cache_path}")
        
        # Create pair directory (directly under pair_dir, not pair_1 subfolder)
        coarse_folder = os.path.join(pair_dir, 'coarse')
        os.makedirs(coarse_folder, exist_ok=True)
        
        # Output path
        output_mesh = os.path.join(coarse_folder, f"{source_model}_deformed_to_{target_model}.obj")
        
        # Run LBS deformation using cached mesh file
        print(f"    Running LBS deformation: {source_mesh_cache_path} -> {output_mesh}")
        lbs_mesh_deformation(source_mesh_cache_path, source_kp, target_kp, output_mesh)
        if os.path.exists(output_mesh):
            print(f"    Generated LBS deformed mesh: {os.path.basename(output_mesh)}")
        else:
            print(f"    ERROR: LBS deformed mesh not created at: {output_mesh}")
            raise ValueError(f"LBS deformation did not create output file: {output_mesh}")
        
    except Exception as e:
        print(f"LBS deformation failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError("All deformation attempts failed")