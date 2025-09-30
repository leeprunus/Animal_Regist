#!/usr/bin/env python3

import os
# Set environment variables to suppress warnings before any imports
os.environ['PYTHONWARNINGS'] = 'ignore'
# os.environ['TORCH_LOGS'] = 'ERROR'  # Disabled - causes issues with PyTorch 2.4+
"""
Main correspondence pipeline for 3D animal mesh registration.
Processes two input meshes to establish dense correspondences using
keypoint detection, visual features, and deformation-based alignment.

Features:
- Direct mesh-to-mesh processing
- Folder structure: result/[source]_[target]_[timestamp]/ with pair/ subfolder
- Preprocessing: keypoint detection, normalization, DINO features
- Cache system for computational efficiency
- Python implementation
- Automatic output naming: correspondences.json in each result directory
- Essential files generated for correspondence analysis

Usage:
    python correspondence.py <source.obj> <target.obj>
    python correspondence.py --no-simplify <source.obj> <target.obj>
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings('ignore')
logging.getLogger('mmengine').setLevel(logging.ERROR)
logging.getLogger('mmpose').setLevel(logging.ERROR)
logging.getLogger('mmdet').setLevel(logging.ERROR)

# Add current directory to path for imports
sys.path.append('.')

# Import the processing modules from the modular structure
from src.utils import get_cache_manager, extract_correspondences, print_summary
from src.utils.config import get_config as get_pipeline_config
from src.utils.mesh_converter import validate_and_convert_mesh_pair, get_supported_formats
from src.preprocessing import run_keypoint_detection, run_normalization, run_dino_extraction
from src.pipeline import run_alignment
from src.deformation import run_autorig


def main():
    parser = argparse.ArgumentParser(
        description='Animal_Regist - Two-Mesh Correspondence Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source', dest='source_mesh',
                       help='Path to source mesh file (.obj, .ply, .stl, .glb, .gltf, etc.)')
    parser.add_argument('--target', dest='target_mesh',
                       help='Path to target mesh file (.obj, .ply, .stl, .glb, .gltf, etc.)')
    parser.add_argument('--config', default='config/pipeline.yaml',
                       help='Path to pipeline configuration file (.yaml)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable mesh simplification (use full resolution meshes)')
    
    # Keep backward compatibility with positional arguments
    parser.add_argument('source_mesh_pos', nargs='?',
                       help='Path to source mesh file (.obj, .ply, .stl, .glb, .gltf, etc.) - positional argument')
    parser.add_argument('target_mesh_pos', nargs='?',
                       help='Path to target mesh file (.obj, .ply, .stl, .glb, .gltf, etc.) - positional argument')
    
    args = parser.parse_args()
    
    # Initialize configuration system
    config = get_pipeline_config(args.config)
    
    # Handle mesh path resolution: positional -> optional -> config file
    if args.source_mesh_pos and args.target_mesh_pos:
        # Use positional arguments (backward compatibility)
        args.source_mesh = args.source_mesh_pos
        args.target_mesh = args.target_mesh_pos
    elif not args.source_mesh or not args.target_mesh:
        # Read from config file if not provided via arguments
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            return 1
            
        alignment_config = config.get_alignment_config()
        default_paths = alignment_config.get('default_paths', {})
            
        if not args.source_mesh:
            if 'source_mesh' in default_paths:
                args.source_mesh = default_paths['source_mesh']
            else:
                print(f"Error: Source mesh not specified and not found in config file")
                return 1
                
        if not args.target_mesh:
            if 'target_mesh' in default_paths:
                args.target_mesh = default_paths['target_mesh']
            else:
                print(f"Error: Target mesh not specified and not found in config file")
                return 1
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    # Validate mesh files exist
    if not os.path.exists(args.source_mesh):
        print(f"Error: Source mesh file not found: {args.source_mesh}")
        return 1
    
    if not os.path.exists(args.target_mesh):
        print(f"Error: Target mesh file not found: {args.target_mesh}")
        return 1
    
    # Check and convert file formats using modular utility
    supported_formats = get_supported_formats()
    
    source_ext = Path(args.source_mesh).suffix.lower()
    target_ext = Path(args.target_mesh).suffix.lower()
    
    if source_ext not in supported_formats:
        print(f"Error: Unsupported source mesh format: {source_ext}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return 1
    
    if target_ext not in supported_formats:
        print(f"Error: Unsupported target mesh format: {target_ext}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return 1
    
    # Setup paths
    source_name = Path(args.source_mesh).stem
    target_name = Path(args.target_mesh).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get output configuration
    output_config = config.get_output_config()
    cache_config = config.get_cache_config()
    
    result_dir = Path(output_config['results_dir'])
    if output_config['create_timestamped_dirs']:
        work_dir = result_dir / f"{source_name}_to_{target_name}_{timestamp}"
    else:
        work_dir = result_dir / f"{source_name}_to_{target_name}"
    
    # Set standard output JSON path in result directory
    output_json = work_dir / f"correspondences.{output_config['correspondences_format']}"
    
    # Create directories
    result_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    
    # Convert meshes to OBJ format if needed - save directly to models folder
    models_dir = Path('./models')
    try:
        original_source_mesh = args.source_mesh
        original_target_mesh = args.target_mesh
        
        # Target OBJ paths in models folder
        source_obj_path = models_dir / f"{source_name}.obj"
        target_obj_path = models_dir / f"{target_name}.obj"
        
        import shutil
        from src.utils.mesh_converter import needs_conversion
        
        from src.utils.mesh_converter import convert_mesh_to_obj
        
        # Handle source mesh
        if needs_conversion(original_source_mesh):
            # Convert directly to target location
            max_verts = None if args.no_simplify else None  # Let convert function read from config
            if convert_mesh_to_obj(original_source_mesh, str(source_obj_path), max_verts, not args.no_simplify):
                args.source_mesh = str(source_obj_path)
                print(f"Converted and saved: {Path(original_source_mesh).name} -> {source_obj_path.name}")
            else:
                raise RuntimeError(f"Failed to convert source mesh: {original_source_mesh}")
        else:
            # Already OBJ - use original file directly (simplification will happen during copy to result)
            args.source_mesh = original_source_mesh
        
        # Handle target mesh
        if needs_conversion(original_target_mesh):
            # Convert directly to target location
            max_verts = None if args.no_simplify else None  # Let convert function read from config
            if convert_mesh_to_obj(original_target_mesh, str(target_obj_path), max_verts, not args.no_simplify):
                args.target_mesh = str(target_obj_path)
                print(f"Converted and saved: {Path(original_target_mesh).name} -> {target_obj_path.name}")
            else:
                raise RuntimeError(f"Failed to convert target mesh: {original_target_mesh}")
        else:
            # Already OBJ - use original file directly (simplification will happen during copy to result)
            args.target_mesh = original_target_mesh
        
        simplify_status = "disabled" if args.no_simplify else "enabled"
        print(f"Mesh processing completed (simplification: {simplify_status}).")
        
        # Save the processed meshes to result directory for pipeline use
        import shutil
        try:
            # Create meshes subdirectory in result folder
            meshes_dir = work_dir / 'meshes'
            meshes_dir.mkdir(exist_ok=True)
            
            # Process and save source mesh to result directory
            actual_source_path = Path(args.source_mesh)
            result_source_path = meshes_dir / f"{source_name}.obj"
            
            if not args.no_simplify and actual_source_path.suffix.lower() == '.obj':
                # Apply simplification if needed when copying to result
                from src.utils.mesh_converter import create_simplified_obj
                try:
                    # Create a temporary simplified version
                    simplified_path = create_simplified_obj(str(actual_source_path))
                    shutil.copy2(simplified_path, result_source_path)
                    # Clean up temporary simplified file if it's different from original
                    if simplified_path != str(actual_source_path) and os.path.exists(simplified_path):
                        os.remove(simplified_path)
                except Exception as e:
                    print(f"Simplification failed, using original: {e}")
                    shutil.copy2(actual_source_path, result_source_path)
            else:
                # Copy original file without simplification
                shutil.copy2(actual_source_path, result_source_path)
            
            # Process and save target mesh to result directory
            actual_target_path = Path(args.target_mesh)
            result_target_path = meshes_dir / f"{target_name}.obj"
            
            if not args.no_simplify and actual_target_path.suffix.lower() == '.obj':
                # Apply simplification if needed when copying to result
                try:
                    # Create a temporary simplified version
                    simplified_path = create_simplified_obj(str(actual_target_path))
                    shutil.copy2(simplified_path, result_target_path)
                    # Clean up temporary simplified file if it's different from original
                    if simplified_path != str(actual_target_path) and os.path.exists(simplified_path):
                        os.remove(simplified_path)
                except Exception as e:
                    print(f"Simplification failed, using original: {e}")
                    shutil.copy2(actual_target_path, result_target_path)
            else:
                # Copy original file without simplification
                shutil.copy2(actual_target_path, result_target_path)
            
            print(f"Saved processed meshes to result folder:")
            print(f"  Source: {result_source_path.relative_to(work_dir)}")
            print(f"  Target: {result_target_path.relative_to(work_dir)}")
            
            # Update mesh paths to point to result folder for pipeline use
            args.source_mesh = str(result_source_path)
            args.target_mesh = str(result_target_path)
            
        except Exception as e:
            print(f"Warning: Failed to save processed meshes: {e}")
            return 1
        
    except Exception as e:
        print(f"Error during mesh processing: {e}")
        return 1
    
    try:
        # Initialize cache system
        cache_manager = get_cache_manager(cache_config['directory'])
        
        # Step 1: Preprocessing - process the result folder meshes
        print("Processing keypoints...")
        run_keypoint_detection(str(meshes_dir), specific_files=[args.source_mesh, args.target_mesh])
        print("Processing mesh normalization...")
        run_normalization(str(meshes_dir), specific_files=[args.source_mesh, args.target_mesh])
        print("Processing DINO features...")
        run_dino_extraction(str(meshes_dir), specific_files=[args.source_mesh, args.target_mesh])
        
        # Create directory structure with 'pair' subfolder
        pair_dir = work_dir / 'pair'
        pair_dir.mkdir(exist_ok=True)
        
        
        # Create output directories under pair/
        (pair_dir / 'coarse').mkdir(exist_ok=True)
        (pair_dir / 'dino_result').mkdir(exist_ok=True)
        (pair_dir / 'dense').mkdir(exist_ok=True)
        (pair_dir / 'dense_result').mkdir(exist_ok=True)
        
        try:
            # Step 2: Deformation
            print("Processing mesh deformation...")
            run_autorig(str(pair_dir), source_name, target_name, args.source_mesh, args.target_mesh)
        except Exception as e:
            print(f"Mesh deformation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 3: Alignment
        print("Processing ARAP alignment...")
        run_alignment(str(pair_dir), source_name, target_name, config, str(meshes_dir))
        
        # Step 4: Extraction
        print("Extracting results...")
        extract_correspondences(str(pair_dir), source_name, target_name, str(output_json))
        
        # Extract and display final MGE
        try:
            import json
            import glob
            
            mge_value = None
                
            # First check the main output file
            with open(output_json, 'r') as f:
                result_data = json.load(f)
                
            if 'metadata' in result_data and 'mge' in result_data['metadata']:
                mge_value = result_data['metadata']['mge']
            elif 'mge' in result_data:
                mge_value = result_data['mge']
                
            # If not found in main file, check the eval correspondences file
            if mge_value is None:
                eval_files = glob.glob(str(pair_dir) + "/**/1_to_2_eval_correspondences.json", recursive=True)
                if eval_files:
                    with open(eval_files[0], 'r') as f:
                        eval_data = json.load(f)
                    if 'metadata' in eval_data and 'mge' in eval_data['metadata']:
                        mge_value = eval_data['metadata']['mge']
                    elif 'mge' in eval_data:
                        mge_value = eval_data['mge']
                
            if mge_value is not None:
                print(f"MGE: {mge_value:.6f}")
            else:
                print("MGE value not found in results")
                    
        except Exception as e:
            print(f"\nCould not extract MGE: {e}")
        
        # Print completion summary
        print_summary(args.source_mesh, args.target_mesh, str(output_json), str(work_dir))
        
        return 0
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
