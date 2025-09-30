#!/usr/bin/env python3
"""
Mesh Format Converter

This module handles conversion of various 3D mesh formats to OBJ format
for processing in the Animal_Regist pipeline.
"""

import os
import shutil
import trimesh
import tempfile
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# Supported input formats (trimesh can load these)
SUPPORTED_FORMATS = {
    '.obj': 'Wavefront OBJ',
    '.ply': 'Polygon File Format',
    '.stl': 'STereoLithography',
    '.glb': 'glTF Binary',
    '.gltf': 'GL Transmission Format', 
    '.dae': 'COLLADA',
    '.3ds': '3D Studio Max',
    '.x3d': 'X3D',
    '.off': 'Object File Format',
    '.mesh': 'MESH format'
}


def get_supported_formats() -> List[str]:
    """Return list of supported mesh file extensions."""
    return list(SUPPORTED_FORMATS.keys())


def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported for conversion."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS


def needs_conversion(file_path: str) -> bool:
    """Check if file needs conversion to OBJ format."""
    ext = Path(file_path).suffix.lower()
    return ext != '.obj' and ext in SUPPORTED_FORMATS


def simplify_mesh(mesh, target_vertex_count=50000):
    """
    Simplify mesh using Open3D's quadric decimation.
    
    Args:
        mesh: Trimesh object to simplify
        target_vertex_count: Maximum number of vertices
        
    Returns:
        Simplified trimesh object
    """
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > target_vertex_count:
        print(f"Mesh has {len(vertices)} vertices, simplifying to {target_vertex_count}...")
        
        try:
            # Convert trimesh to Open3D mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            
            # Calculate target triangle count
            ratio = target_vertex_count / len(vertices)
            num_triangles = len(mesh.faces)
            target_triangles = int(num_triangles * ratio)
            
            # Apply quadric decimation
            simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            
            # Convert back to trimesh
            simplified_vertices = np.asarray(simplified_o3d_mesh.vertices)
            simplified_faces = np.asarray(simplified_o3d_mesh.triangles)
            
            simplified_mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)
            print(f"Simplified from {len(vertices)} to {len(simplified_vertices)} vertices")
            return simplified_mesh
            
        except Exception as e:
            print(f"Quadric decimation failed: {e}")
            print("Using original mesh without simplification")
            return mesh
    else:
        return mesh


def convert_mesh_to_obj(input_file: str, output_file: str, max_vertices: Optional[int] = None, enable_simplify: bool = True) -> bool:
    """
    Convert a 3D mesh file to OBJ format with optional simplification.
    
    Args:
        input_file: Path to input mesh file
        output_file: Path to output OBJ file
        max_vertices: Maximum number of vertices (if None, reads from config)
        enable_simplify: Whether to enable mesh simplification
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        print(f"Converting {Path(input_file).name} to OBJ format...")
        
        # Get max_vertices from config if not provided
        if max_vertices is None:
            try:
                from .config import ConfigManager
                config = ConfigManager()
                max_vertices = config.get('mesh.processing.max_vertices', 100000)
            except Exception as e:
                print(f"Warning: Could not read max_vertices from config: {e}")
                max_vertices = 100000  # Default fallback
        
        # Load mesh using trimesh
        mesh = trimesh.load(input_file)
        
        # Handle scene objects (e.g., from glTF files)
        if hasattr(mesh, 'geometry'):
            # If it's a Scene, extract the first geometry
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = geometries[0]
            else:
                raise ValueError("No geometry found in the mesh file")
        
        # Ensure we have a valid mesh
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise ValueError("Invalid mesh: missing vertices or faces")
        
        # Basic mesh validation
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
        
        # Simplify mesh if it exceeds max_vertices and simplification is enabled
        if enable_simplify:
            mesh = simplify_mesh(mesh, max_vertices)
        else:
            print(f"Mesh simplification disabled, using {len(mesh.vertices)} vertices")
        
        # Create output directory if it doesn't exist
        os.makedirs(Path(output_file).parent, exist_ok=True)
        
        # Export as OBJ
        mesh.export(output_file)
        
        if not os.path.exists(output_file):
            raise RuntimeError("OBJ export failed - file not created")
        
        print(f"Successfully converted to: {Path(output_file).name}")
        return True
        
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False


def create_simplified_obj(input_file: str, max_vertices: Optional[int] = None) -> str:
    """
    Create a simplified version of an OBJ file if it exceeds the vertex limit.
    Original file is never modified.
    
    Args:
        input_file: Path to input OBJ file
        max_vertices: Maximum number of vertices (if None, reads from config)
        
    Returns:
        Path to file to use (either original or simplified version)
    """
    try:
        # Get max_vertices from config if not provided
        if max_vertices is None:
            try:
                from .config import ConfigManager
                config = ConfigManager()
                max_vertices = config.get('mesh.processing.max_vertices', 100000)
            except Exception as e:
                print(f"Warning: Could not read max_vertices from config: {e}")
                max_vertices = 100000  # Default fallback
        
        # Load mesh
        mesh = trimesh.load(input_file)
        
        # Basic validation
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise ValueError("Invalid mesh: missing vertices or faces")
        
        original_vertices = len(mesh.vertices)
        
        # Check if simplification is needed
        if original_vertices <= max_vertices:
            return input_file  # Return original file path - no simplification needed
        
        # Create simplified version with different name
        input_path = Path(input_file)
        simplified_filename = f"{input_path.stem}_simplified.obj"
        simplified_path = input_path.parent / simplified_filename
        
        # Check if simplified version already exists and is up-to-date
        if simplified_path.exists():
            # Check if simplified version is newer than original
            original_mtime = os.path.getmtime(input_file)
            simplified_mtime = os.path.getmtime(simplified_path)
            if simplified_mtime >= original_mtime:
                print(f"Using existing simplified mesh: {simplified_filename}")
                return str(simplified_path)
        
        # Simplify the mesh
        print(f"Mesh has {original_vertices} vertices, simplifying to {max_vertices}...")
        mesh = simplify_mesh(mesh, max_vertices)
        
        # Export simplified mesh
        mesh.export(str(simplified_path))
        
        if not simplified_path.exists():
            raise RuntimeError("Simplified OBJ export failed - file not created")
        
        print(f"Simplified mesh created: {simplified_filename}")
        return str(simplified_path)
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        print(f"Using original file instead: {Path(input_file).name}")
        return input_file  # Return original file path as fallback


def validate_and_convert_mesh_pair(source_file: str, target_file: str, 
                                 output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Validate and convert a pair of mesh files to OBJ format if needed.
    
    Args:
        source_file: Path to source mesh file
        target_file: Path to target mesh file  
        output_dir: Directory for converted files (if None, uses temp directory)
        
    Returns:
        Tuple of (converted_source_path, converted_target_path)
        
    Raises:
        ValueError: If files don't exist or formats are unsupported
        RuntimeError: If conversion fails
    """
    # Validate input files exist
    if not os.path.exists(source_file):
        raise ValueError(f"Source mesh file not found: {source_file}")
    
    if not os.path.exists(target_file):
        raise ValueError(f"Target mesh file not found: {target_file}")
    
    # Check formats are supported
    if not is_supported_format(source_file):
        source_ext = Path(source_file).suffix.lower()
        raise ValueError(f"Unsupported source format: {source_ext}. Supported: {', '.join(get_supported_formats())}")
    
    if not is_supported_format(target_file):
        target_ext = Path(target_file).suffix.lower()
        raise ValueError(f"Unsupported target format: {target_ext}. Supported: {', '.join(get_supported_formats())}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mesh_conversion_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process source file
    source_name = Path(source_file).stem
    if needs_conversion(source_file):
        converted_source = os.path.join(output_dir, f"{source_name}.obj")
        if not convert_mesh_to_obj(source_file, converted_source):
            raise RuntimeError(f"Failed to convert source mesh: {source_file}")
    else:
        # Already OBJ format, copy to output directory
        converted_source = os.path.join(output_dir, f"{source_name}.obj")
        shutil.copy2(source_file, converted_source)
    
    # Process target file
    target_name = Path(target_file).stem
    if needs_conversion(target_file):
        converted_target = os.path.join(output_dir, f"{target_name}.obj")
        if not convert_mesh_to_obj(target_file, converted_target):
            raise RuntimeError(f"Failed to convert target mesh: {target_file}")
    else:
        # Already OBJ format, copy to output directory
        converted_target = os.path.join(output_dir, f"{target_name}.obj")
        shutil.copy2(target_file, converted_target)
    
    return converted_source, converted_target


def get_mesh_info(file_path: str) -> Dict[str, any]:
    """
    Get basic information about a mesh file.
    
    Args:
        file_path: Path to mesh file
        
    Returns:
        Dictionary with mesh information
    """
    try:
        mesh = trimesh.load(file_path)
        
        # Handle scene objects
        if hasattr(mesh, 'geometry'):
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = geometries[0]
        
        info = {
            'file_path': file_path,
            'format': Path(file_path).suffix.lower(),
            'vertices': len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            'faces': len(mesh.faces) if hasattr(mesh, 'faces') else 0,
            'is_watertight': mesh.is_watertight if hasattr(mesh, 'is_watertight') else False,
            'bounds': mesh.bounds.tolist() if hasattr(mesh, 'bounds') else None,
            'volume': float(mesh.volume) if hasattr(mesh, 'volume') else None,
            'area': float(mesh.area) if hasattr(mesh, 'area') else None
        }
        
        return info
        
    except Exception as e:
        return {
            'file_path': file_path,
            'format': Path(file_path).suffix.lower(),
            'error': str(e),
            'vertices': 0,
            'faces': 0
        }


def cleanup_conversion_files(conversion_dir: str) -> None:
    """
    Clean up temporary conversion files.
    
    Args:
        conversion_dir: Directory containing conversion files
    """
    try:
        if os.path.exists(conversion_dir):
            shutil.rmtree(conversion_dir)
    except Exception as e:
        print(f"Warning: Failed to cleanup conversion directory {conversion_dir}: {e}")


if __name__ == "__main__":
    # Simple CLI for testing conversion
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python mesh_converter.py <input_file> <output_file>")
        print(f"Supported formats: {', '.join(get_supported_formats())}")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if convert_mesh_to_obj(input_file, output_file):
        print("Conversion successful!")
        
        # Display mesh info
        info = get_mesh_info(output_file)
        print(f"Vertices: {info['vertices']}")
        print(f"Faces: {info['faces']}")
        print(f"Watertight: {info['is_watertight']}")
    else:
        print("Conversion failed!")
        sys.exit(1)
