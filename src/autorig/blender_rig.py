#!/usr/bin/env python3
"""
AutoRig - Automatic Rigging and Deformation System

This script processes model pairs from CSV files:
1. Auto-rigs source mesh to target keypoints
2. Saves deformed result to coarse folder

Works with the updated pipeline that processes all meshes from models folder.
"""

import bpy
import numpy as np
import os
import argparse
import glob
import csv
import json
import shutil
from pathlib import Path
from mathutils import Vector, Matrix
import sys

def ensure_dir_exists(directory):
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# ====================================================
# 1. Helper function definitions
# ====================================================

# Keypoint names (17 keypoints, including left eye and right eye)
keypoint_names = {
    0: "L_Eye",
    1: "R_Eye",
    2: "Nose",
    3: "Neck",
    4: "Root_of_Tail",
    5: "L_Shoulder",
    6: "L_Elbow",
    7: "L_FrontPaw",
    8: "R_Shoulder",
    9: "R_Elbow",
    10: "R_FrontPaw",
    11: "L_Hip",
    12: "L_Knee",
    13: "L_BackPaw",
    14: "R_Hip",
    15: "R_Knee",
    16: "R_BackPaw",
    17: "Tail_Tip",
    18: "Tail_Spine"
}

# Bone connection relationships: parent→child
bone_connections = [
    [3, 4],          # Spine: Neck -> Root_of_Tail
    # [4, 17],         # Tail: Root_of_Tail -> Tail_Tip
    [4, 18],         # Tail: Root_of_Tail -> Tail_Spine
    [18, 17],        # Tail: Tail_Spine -> Tail_Tip
    [3, 0], [0, 2],  # Left eye to nose: Neck -> L_Eye -> Nose
    [3, 1], [1, 2],  # Right eye to nose: Neck -> R_Eye -> Nose
    [3, 5], [5, 6], [6, 7],  # Left front leg
    [3, 8], [8, 9], [9, 10],  # Right front leg
    [4, 11], [11, 12], [12, 13],  # Left back leg
    [4, 14], [14, 15], [15, 16]   # Right back leg
]

def load_and_process_keypoints(file_path):
    """Load keypoint data, maintaining original structure (17 keypoints)"""
    kp = np.load(file_path)
    return kp

def convert_coordinates(keypoints):
    """
    Convert keypoints from prediction coordinate system to Blender coordinate system.
    For example: swap Y and Z axes, and negate Y axis.
    """
    conv = np.copy(keypoints)
    for i in range(len(conv)):
        y = conv[i, 1]
        z = conv[i, 2]
        conv[i, 1] = -z  # Y = -Z
        conv[i, 2] = y   # Z = Y
    return conv

def convert_back_coordinates(keypoints):
    """
    Convert keypoints from Blender coordinate system back to original coordinate system.
    This is the inverse operation of convert_coordinates.
    """
    conv = np.copy(keypoints)
    for i in range(len(conv)):
        y = conv[i, 1]
        z = conv[i, 2]
        conv[i, 1] = z    # Y = Z
        conv[i, 2] = -y   # Z = -Y
    return conv

def create_control_empties(keypoints):
    """Create empty objects as controllers based on keypoint positions, and set keypoint_index property"""
    converted = convert_coordinates(keypoints)
    empties = {}
    for idx, loc in enumerate(converted):
        name = keypoint_names.get(idx, f"Keypoint_{idx}")
        empty = bpy.data.objects.new(f"Control_{name}", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.05
        empty.location = Vector(loc)
        bpy.context.collection.objects.link(empty)
        empty["keypoint_index"] = idx
        empties[idx] = empty
    return empties

def create_dynamic_armature(empties):
    """Create dynamic armature and add constraints to each bone to follow controllers"""
    armature = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", armature)
    bpy.context.collection.objects.link(arm_obj)
    
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}
    
    # Add main bone connections - don't add intermediate bones to maintain original topology
    for start_idx, end_idx in bone_connections:
        bone_name = f"{keypoint_names[start_idx]}_to_{keypoint_names[end_idx]}"
        bone = armature.edit_bones.new(bone_name)
        bone.head = empties[start_idx].location
        bone.tail = empties[end_idx].location
        bones[bone_name] = bone
        
        # Remove intermediate bone creation code to maintain original topology structure
        # Original code would create intermediate bones, causing vertex count increase
    
    # Establish parent-child relationships
    for start_idx, end_idx in bone_connections:
        bone_name = f"{keypoint_names[start_idx]}_to_{keypoint_names[end_idx]}"
        for p_start, p_end in bone_connections:
            if p_end == start_idx:
                parent_name = f"{keypoint_names[p_start]}_to_{keypoint_names[p_end]}"
                if parent_name in bones and bone_name in bones and parent_name != bone_name:
                    bones[bone_name].parent = bones[parent_name]
                    break
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add constraints in Pose mode (COPY_LOCATION and STRETCH_TO)
    bpy.ops.object.mode_set(mode='POSE')
    for start_idx, end_idx in bone_connections:
        bone_name = f"{keypoint_names[start_idx]}_to_{keypoint_names[end_idx]}"
        pose_bone = arm_obj.pose.bones.get(bone_name)
        if pose_bone:
            con_head = pose_bone.constraints.new('COPY_LOCATION')
            con_head.target = empties[start_idx]
            con_stretch = pose_bone.constraints.new('STRETCH_TO')
            con_stretch.target = empties[end_idx]
            con_stretch.volume = 'NO_VOLUME'
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def import_model(obj_file_path):
    """Import OBJ model without loading MTL material files"""
    # Convert PosixPath to string for Blender compatibility
    obj_file_path = str(obj_file_path)
    if not os.path.exists(obj_file_path):
        print(f"Error: Cannot find model file {obj_file_path}")
        # Try to find alternative files
        for file in os.listdir('.'):
            if file.endswith('.obj'):
                print(f"Found possible model file: {file}")
                obj_file_path = file
                break
        else:
            raise FileNotFoundError(f"Cannot find any .obj model files")
    
    print(f"Importing model: {obj_file_path}")
    
    # Clear naming conflicts
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj)
    
    # Blender 4.0 compatible import
    try:
        # First try Blender 4.0 import method
        bpy.ops.wm.obj_import(
            filepath=obj_file_path,
            # Blender 4.0 no longer supports use_materials parameter
            # We try to use other available parameters
            forward_axis='NEGATIVE_Z',
            up_axis='Y'
        )
    except (AttributeError, TypeError) as e1:
        print(f"Failed using wm.obj_import: {e1}, trying alternative method...")
        try:
            # Try old Blender import method
            bpy.ops.import_scene.obj(
                filepath=obj_file_path,
                axis_forward='-Z',
                axis_up='Y'
            )
        except (AttributeError, TypeError) as e2:
            print(f"Failed using import_scene.obj: {e2}")
            
            # Finally try importing without parameters
            try:
                bpy.ops.import_scene.obj(filepath=obj_file_path)
            except AttributeError:
                try:
                    bpy.ops.wm.obj_import(filepath=obj_file_path)
                except AttributeError:
                    print("Error: Cannot find suitable operator to import OBJ files.")
                    print("Please ensure Blender's OBJ plugin is enabled.")
                    raise
    
    # Get imported model object
    model = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            model = obj
            break
    
    if not model:
        print("Warning: No imported mesh object found, searching in all objects")
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                model = obj
                break
    
    if not model:
        raise ValueError("Cannot find valid mesh object after import")
    
    return model

def export_model_obj(model_obj, filepath):
    """Export model as OBJ file without creating MTL material files, maintaining original topology"""
    # Convert PosixPath to string for Blender compatibility
    filepath = str(filepath)
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    
    # Check vertex count before export
    print(f"Vertex count before export: {len(model_obj.data.vertices)}")
    
    # Blender 4.0 compatible export, set to not modify geometry
    try:
        # First try Blender 4.0 export method
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            # Blender 4.0 may use different parameters
            forward_axis='NEGATIVE_Z',
            up_axis='Y',
            # Key: ensure no geometry modifications
            export_smooth_groups=False,
            export_normals=False,
            export_uv=False,
            export_materials=False
        )
    except (AttributeError, TypeError) as e1:
        print(f"Failed using wm.obj_export: {e1}, trying alternative method...")
        try:
            # Try old Blender export method
            bpy.ops.export_scene.obj(
                filepath=filepath,
                use_selection=True,
                axis_forward='-Z',
                axis_up='Y',
                # Key: ensure no geometry modifications
                use_smooth_groups=False,
                use_normals=False,
                use_uvs=False,
                use_materials=False
            )
        except (AttributeError, TypeError) as e2:
            print(f"Failed using export_scene.obj: {e2}")
            
            # Finally try simplest export
            try:
                bpy.ops.export_scene.obj(
                    filepath=filepath,
                    use_selection=True
                )
            except AttributeError:
                try:
                    bpy.ops.wm.obj_export(
                        filepath=filepath,
                        export_selected_objects=True
                    )
                except AttributeError:
                    print("Error: Cannot find suitable operator to export OBJ files.")
                    print("Please ensure Blender's OBJ plugin is enabled.")
                    raise
    
    # Check if MTL file was generated, delete it if so
    mtl_filepath = os.path.splitext(filepath)[0] + ".mtl"
    if os.path.exists(mtl_filepath):
        try:
            os.remove(mtl_filepath)
            print(f"Deleted unnecessary MTL file: {mtl_filepath}")
        except OSError as e:
            print(f"Warning: Cannot delete MTL file {mtl_filepath}: {e}")
    
    print(f"Model exported to {filepath}")

def extract_keypoints_from_empties(empties):
    """Extract keypoint positions from controller empty objects"""
    kp_count = len(empties)
    keypoints = np.zeros((kp_count, 3))
    
    for idx, empty in empties.items():
        keypoints[idx] = np.array([empty.location.x, empty.location.y, empty.location.z])
    
    return keypoints

def bind_model_to_armature(model_obj, armature_obj):
    """Bind model to armature, maintaining original topology structure"""
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    
    # Use automatic weight binding, ensuring topology is not modified
    try:
        # Remove any existing Armature modifiers before adding modifier
        for mod in model_obj.modifiers:
            if mod.type == 'ARMATURE':
                bpy.ops.object.modifier_remove(modifier=mod.name)
        
        # Add modifier, set to maintain topology
        modifier = model_obj.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = armature_obj
        modifier.use_vertex_groups = True
        modifier.use_bone_envelopes = False  # Don't use envelopes, only vertex groups
        
        # Generate vertex groups using automatic weights
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        
        # Optimize modifier settings to maintain topology
        modifier.use_deform_preserve_volume = False  # Turn off volume preservation to avoid mesh subdivision
        modifier.use_multi_modifier = False  # Ensure not using multi-modifier
        
        print("Model successfully bound to armature, maintaining original topology")
    except Exception as e:
        print(f"Error binding model to armature: {e}")
        import traceback
        traceback.print_exc()
        
        # If binding fails, fall back to basic binding
        print("Falling back to basic armature binding...")
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

def apply_armature_modifier(model_obj):
    """Apply Armature modifier on model to bake bone deformation into mesh"""
    bpy.context.view_layer.objects.active = model_obj
    for mod in model_obj.modifiers:
        if mod.type == 'ARMATURE':
            bpy.ops.object.modifier_apply(modifier=mod.name)
            break

def process_single_target(source_model_path, source_kp_path, target_kp_path, output_path):
    """
    Process single target model: deform template

    Args:
        source_model_path: Template model path
        source_kp_path: Template keypoints path
        target_kp_path: Target keypoints path
        output_path: Output path for deformed model
    """
    # Clear current scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    print(f"\nStarting processing: {os.path.basename(target_kp_path)}")
    
    # Load template and target keypoints
    source_kp = load_and_process_keypoints(source_kp_path)
    target_kp = load_and_process_keypoints(target_kp_path)
    
    # Create template controllers and armature
    control_empties = create_control_empties(source_kp)
    armature_obj = create_dynamic_armature(control_empties)
    
    # Import template model and bind armature
    source_mesh = import_model(source_model_path)
    if source_mesh is None:
        raise Exception("Cannot import template model, please check file path.")
    print(f"Vertex count after import: {len(source_mesh.data.vertices)}")
    
    bind_model_to_armature(source_mesh, armature_obj)
    print(f"Vertex count after binding: {len(source_mesh.data.vertices)}")
    
    # Move controllers to separate collection
    if "Controls" not in bpy.data.collections:
        controls_collection = bpy.data.collections.new("Controls")
        bpy.context.scene.collection.children.link(controls_collection)
    else:
        controls_collection = bpy.data.collections["Controls"]
    for empty in control_empties.values():
        if empty.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(empty)
        controls_collection.objects.link(empty)
    
    # Convert coordinate system
    target_kp_blender = convert_coordinates(target_kp)
    
    # Directly use target keypoint positions
    print("Directly using target keypoint positions instead of Procrustes aligned positions")
    
    # Update controller positions to target keypoint positions
    for idx, empty in control_empties.items():
        if idx < len(target_kp_blender):
            empty.location = Vector(target_kp_blender[idx])
        else:
            print(f"Warning: Controller index {idx} out of range")
    
    # Update scene
    bpy.context.view_layer.update()
    print(f"Vertex count after updating controllers: {len(source_mesh.data.vertices)}")
    
    # Apply armature deformation
    apply_armature_modifier(source_mesh)
    print(f"Vertex count after applying modifier: {len(source_mesh.data.vertices)}")
    
    # Export deformed template
    export_model_obj(source_mesh, output_path)
    print(f"Deformed template saved: {output_path}")
    
    # Save deformed keypoints
    try:
        # Extract keypoint positions from deformed model
        deformed_keypoints = []
        for idx in range(len(source_kp)):
            kp_name = keypoint_names.get(idx, f"Keypoint_{idx}")
            empty_name = f"Control_{kp_name}"
            
            # Find controller
            empty = None
            for obj in bpy.data.objects:
                if obj.name.startswith(empty_name):
                    empty = obj
                    break
            
            if empty:
                pos = [empty.location.x, empty.location.y, empty.location.z]
                # Convert back to original coordinate system
                pos_orig = convert_back_coordinates(np.array([pos]))[0]
                deformed_keypoints.append(pos_orig)
            else:
                print(f"Warning: Controller not found {empty_name}")
                if idx < len(target_kp):
                    # If controller not found, directly use target keypoint
                    deformed_keypoints.append(target_kp[idx])
                else:
                    # As final fallback, use original template keypoint
                    deformed_keypoints.append(source_kp[idx])
        
        # Save deformed keypoints
        deformed_keypoints = np.array(deformed_keypoints)
        deformed_keypoints_path = os.path.splitext(output_path)[0] + ".npy"
        np.save(deformed_keypoints_path, deformed_keypoints)
        print(f"Deformed template keypoints saved: {deformed_keypoints_path}")
        
        return True
    except Exception as e:
        print(f"Error saving deformed keypoints: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_model_files(models_dir, model_name):
    """
    Find model files (obj and npy) preferring cache system over models directory.
    
    Args:
        models_dir: Path to the models directory (or cache directory)
        model_name: Name of the model (without extension)
        
    Returns:
        Dictionary with file paths
    """
    model_files = {'obj': None, 'npy': None}
    
    # Use cache system - no fallbacks
    # Add current script directory to Python path for Blender environment
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from src/autorig/ to reach project root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import cache manager directly to avoid importing the entire src tree
    sys.path.insert(0, os.path.join(project_root, 'src', 'utils'))
    from cache import get_cache_manager
    cache_manager = get_cache_manager('./cache')
    
    # Find model file to compute cache hash
    # First try the provided models_dir (should be result folder's meshes directory)
    models_path = Path(models_dir)
    mesh_file = models_path / f"{model_name}.obj"
    
    if mesh_file.exists():
        # Use the mesh file from the provided directory
        original_obj = str(mesh_file)
        print(f"  Found mesh in provided directory: {original_obj}")
    else:
        # Fallback to ./models directory
        original_models_dir = './models'
        original_path = Path(original_models_dir)
        original_obj_found = False
        
        if original_path.exists():
            for original_obj_path in original_path.rglob(f"{model_name}.obj"):
                original_obj = str(original_obj_path)
                original_obj_found = True
                print(f"  Found mesh in models directory: {original_obj}")
                break
        
        if not original_obj_found:
            raise ValueError(f"Model file not found: {model_name}.obj (searched: {models_dir}, ./models)")
    
    # Get cached files using the found mesh file
    normalized_mesh = cache_manager.get_normalization_cache_path(original_obj)
    keypoints_file = cache_manager.get_keypoints_cache_path(original_obj)
    
    if not os.path.exists(normalized_mesh):
        raise ValueError(f"Cached normalized mesh not found: {normalized_mesh}")
    if not os.path.exists(keypoints_file):
        raise ValueError(f"Cached keypoints not found: {keypoints_file}")
        
    model_files['obj'] = normalized_mesh
    model_files['npy'] = keypoints_file
    print(f"  Using cached files for {model_name}:")
    print(f"      Mesh: {normalized_mesh}")
    print(f"      Keypoints: {keypoints_file}")
    return model_files

def create_pair_directory_structure(pair_dir, pair_id, source_model, target_model):
    """
    Create pair directory structure.
    
    Args:
        pair_dir: Base pair directory
        pair_id: Unique identifier for this pair
        source_model: Name of the source model
        target_model: Name of the target model
        
    Returns:
        Dictionary with paths to directories
    """
    # Check if we should use simple structure (no pair_0001_ prefix)
    use_simple_structure = os.environ.get('SIMPLE_STRUCTURE', '0') == '1'
    
    if use_simple_structure:
        # Simple structure: use pair_dir directly as the pair folder
        pair_folder = pair_dir
        coarse_dir = os.path.join(pair_folder, "coarse")
        os.makedirs(coarse_dir, exist_ok=True)
    else:
        # Original structure: create pair-specific directory
        pair_folder = os.path.join(pair_dir, f"pair_{pair_id:04d}_{source_model}_to_{target_model}")
        os.makedirs(pair_folder, exist_ok=True)
        
        # Create coarse subdirectory
        coarse_dir = os.path.join(pair_folder, "coarse")
        os.makedirs(coarse_dir, exist_ok=True)
    
    return {
        'pair_folder': pair_folder,
        'coarse_dir': coarse_dir
    }

def process_batch_pairs(models_dir, pair_dir, csv_file):
    """
    Process all pairs from CSV file in batch mode.
    Workflow:
    1. Auto-rig source mesh to target keypoints
    2. Save deformed result to coarse folder
    
    Args:
        models_dir: Path to the models directory containing all normalized meshes
        pair_dir: Path to the pair directory where results will be stored
        csv_file: Path to the CSV file containing model pairs
        
    Returns:
        Number of successfully processed pairs
    """
    # Load pairs from CSV
    pairs = load_pairs_from_csv(csv_file)
    if pairs is None:
        raise ValueError(f"Failed to load pairs from CSV file: {csv_file}")
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING {len(pairs)} MODEL PAIRS WITH AUTORIG")
    print(f"Workflow: Auto-rig source mesh to target keypoints")
    print(f"{'='*80}")
    
    for i, pair in enumerate(pairs, 1):
        pair_id = pair['pair_id']
        source_model = pair['source_model']
        target_model = pair['target_model']
        
        print(f"\n[{i}/{len(pairs)}] Processing Pair {pair_id}: {source_model} -> {target_model}")
        
        # Create pair directory structure
        pair_dirs = create_pair_directory_structure(pair_dir, pair_id, source_model, target_model)
        print(f"  Pair folder: {pair_dirs['pair_folder']}")
        
        # Check if already processed
        expected_output = os.path.join(pair_dirs['coarse_dir'], f"{source_model}_deformed_to_{target_model}.obj")
        autorig_done_marker = os.path.join(pair_dirs['pair_folder'], ".autorig_done")
        
        if os.path.exists(autorig_done_marker) and os.path.exists(expected_output):
            print(f"  ✓ AutoRig already processed, skipping...")
            continue
        
        # Find source and target model files in models directory
        source_files = find_model_files(models_dir, source_model)
        target_files = find_model_files(models_dir, target_model)
        
        # Check if all required files exist
        missing_files = []
        if not source_files['obj'] or not os.path.exists(source_files['obj']):
            missing_files.append(f"source mesh: {source_model}.obj")
        if not source_files['npy'] or not os.path.exists(source_files['npy']):
            missing_files.append(f"source keypoints: {source_model}.npy")
        if not target_files['obj'] or not os.path.exists(target_files['obj']):
            missing_files.append(f"target mesh: {target_model}.obj")
        if not target_files['npy'] or not os.path.exists(target_files['npy']):
            missing_files.append(f"target keypoints: {target_model}.npy")
        
        if missing_files:
            raise ValueError(f"Missing files: {', '.join(missing_files)}")
        
        print(f"  Found source files: {source_files['obj']}")
        print(f"  Found target files: {target_files['obj']}")
        
        # Auto-rig source mesh to target keypoints
        print(f"  Auto-rigging source mesh to target keypoints...")
        print(f"    Source mesh: {source_files['obj']}")
        print(f"    Source keypoints: {source_files['npy']}")
        print(f"    Target keypoints: {target_files['npy']}")
        print(f"    Output: {expected_output}")
    
        # Process the pair with AutoRig
        process_single_target(
            source_files['obj'],
            source_files['npy'],
            target_files['npy'],
            expected_output
        )
        
        # Create marker file to indicate AutoRig is done (only if not in minimal output mode)
        if not os.environ.get('MINIMAL_OUTPUT'):
            with open(autorig_done_marker, 'w') as f:
                f.write(f"AutoRig completed\n")
                f.write(f"source: {source_model}\n")
                f.write(f"target: {target_model}\n")
                f.write(f"deformed_mesh: {os.path.basename(expected_output)}\n")
                f.write(f"workflow: direct_autorig\n")
            
        print(f"  ✓ Successfully processed pair {pair_id}")
    
    print(f"\n{'='*80}")
    print(f"BATCH AUTORIG PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total pairs: {len(pairs)}")
    print(f"All pairs processed successfully")
    
    return len(pairs)

def process_single_pair_direct(models_dir, pair_dir, source_model, target_model):
    """
    Process a single pair directly without CSV file.
    
    Args:
        models_dir: Path to the models directory containing all normalized meshes
        pair_dir: Path to the pair directory where results will be stored
        source_model: Source model name (without .obj extension)
        target_model: Target model name (without .obj extension)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing pair: {source_model} -> {target_model}")
        
        # Find source and target model files in models directory
        source_files = find_model_files(models_dir, source_model)
        target_files = find_model_files(models_dir, target_model)
        
        # Check if all required files exist
        missing_files = []
        for model_name, files in [("source", source_files), ("target", target_files)]:
            if not files['obj'] or not os.path.exists(files['obj']):
                missing_files.append(f"{model_name} OBJ file")
            if not files['npy'] or not os.path.exists(files['npy']):
                missing_files.append(f"{model_name} keypoints file")
        
        if missing_files:
            print(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        # Create pair directory structure
        coarse_dir = os.path.join(pair_dir, "coarse")
        os.makedirs(coarse_dir, exist_ok=True)
        
        # Output path for deformed mesh
        output_path = os.path.join(coarse_dir, f"{source_model}_deformed_to_{target_model}.obj")
        
        # Process the pair using AutoRig
        success = process_single_target(
            source_files['obj'], source_files['npy'],
            target_files['npy'], output_path
        )
        
        if success:
            print(f"  Successfully processed: {source_model} -> {target_model}")
        else:
            print(f"  Failed to process: {source_model} -> {target_model}")
        
        return success
        
    except Exception as e:
        print(f"Error processing pair {source_model} -> {target_model}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ====================================================
# Main Execution
# ====================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    # Filter out Blender arguments - only process arguments after '--'
    script_args = []
    found_separator = False
    for arg in sys.argv:
        if found_separator:
            script_args.append(arg)
        elif arg == '--':
            found_separator = True
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AutoRig - Automatic Rigging and Deformation System')
    parser.add_argument('--models_dir', required=True, help='Directory containing model files')
    parser.add_argument('--pair_dir', required=True, help='Directory for pair output')
    parser.add_argument('--csv_file', help='CSV file with model pairs (for batch processing)')
    parser.add_argument('--source_model', help='Source model name (for single pair processing)')
    parser.add_argument('--target_model', help='Target model name (for single pair processing)')
    
    # Parse the filtered arguments
    args = parser.parse_args(script_args)
    
    print(f"\n{'='*80}")
    print(f"AUTORIG - AUTOMATIC RIGGING AND DEFORMATION SYSTEM")
    print(f"{'='*80}")
    print(f"Models directory: {args.models_dir}")
    print(f"Pair directory: {args.pair_dir}")
    
    try:
        if args.csv_file:
            # Batch processing mode
            print(f"CSV file: {args.csv_file}")
            print(f"{'='*80}")
            num_pairs = process_batch_pairs(args.models_dir, args.pair_dir, args.csv_file)
            print(f"\nAutoRig processing completed successfully - {num_pairs} pairs processed")
            
        elif args.source_model and args.target_model:
            # Single pair processing mode
            print(f"Source model: {args.source_model}")
            print(f"Target model: {args.target_model}")
            print(f"{'='*80}")
            
            # Process single pair directly
            success = process_single_pair_direct(args.models_dir, args.pair_dir, args.source_model, args.target_model)
            if success:
                print(f"\nAutoRig processing completed successfully - 1 pair processed")
            else:
                raise RuntimeError("Single pair processing failed")
                
        else:
            raise ValueError("Either --csv_file or both --source_model and --target_model must be provided")
        
    except Exception as e:
        print(f"\nAutoRig processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
