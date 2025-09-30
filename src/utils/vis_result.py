#!/usr/bin/env python3
"""
Result Directory Visualization Script

Simple script to visualize correspondences from a result directory.
Just point it to any result directory and it will automatically find and load
the meshes and correspondences.

Features:
- Source mesh has a smooth, uniform color gradient based on spatial position
- Target mesh copies colors from corresponding source vertices
- Matching colors clearly indicate corresponding regions between meshes
- Clean visualization without connection lines

Usage:
    python vis_result.py result/...
    python vis_result.py result/... --front-back
"""

import numpy as np
import open3d as o3d
import json
import sys
import os
import colorsys
import argparse
from pathlib import Path


def load_mesh(obj_path):
    """Load a 3D mesh from an OBJ file."""
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
        if len(mesh.vertices) == 0:
            print(f"Error: No vertices found in {obj_path}")
            return None
        
        mesh.compute_vertex_normals()
        print(f"Loaded {Path(obj_path).name}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        return mesh
    
    except Exception as e:
        print(f"Error loading mesh {obj_path}: {e}")
        return None


def load_correspondences(json_path):
    """Load correspondences from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract correspondences - should be a dict mapping source to target indices
        if 'correspondences' in data:
            correspondences = data['correspondences']
        else:
            correspondences = data
        
        # Convert to list of (source, target) pairs
        pairs = []
        for source_str, target_idx in correspondences.items():
            try:
                source_idx = int(source_str)
                target_idx = int(target_idx)
                pairs.append((source_idx, target_idx))
            except (ValueError, TypeError):
                continue
        
        print(f"Loaded {len(pairs)} correspondences")
        return pairs
    
    except Exception as e:
        print(f"Error loading correspondences {json_path}: {e}")
        return None


def smooth_vertex_colors(mesh, colors, iterations=3):
    """Apply Laplacian smoothing to vertex colors for more uniform appearance."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Build adjacency list for vertices
    adjacency = [[] for _ in range(len(vertices))]
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
    
    # Remove duplicates and convert to sets for faster lookup
    adjacency = [list(set(neighbors)) for neighbors in adjacency]
    
    # Smooth colors using Laplacian smoothing
    smoothed_colors = colors.copy()
    
    for iteration in range(iterations):
        new_colors = smoothed_colors.copy()
        
        for v_idx in range(len(vertices)):
            if len(adjacency[v_idx]) > 0:
                # Average color with neighbors
                neighbor_colors = smoothed_colors[adjacency[v_idx]]
                avg_neighbor_color = np.mean(neighbor_colors, axis=0)
                
                # Blend current color with neighbor average
                smoothing_factor = 0.3  # How much to blend with neighbors
                new_colors[v_idx] = (1 - smoothing_factor) * smoothed_colors[v_idx] + smoothing_factor * avg_neighbor_color
        
        smoothed_colors = new_colors
    
    return smoothed_colors


def generate_smooth_source_colors(vertices):
    """Generate smooth, uniform colors for the source mesh based on spatial position."""
    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    ranges = max_coords - min_coords
    
    colors = np.zeros((len(vertices), 3))
    
    for i, vertex in enumerate(vertices):
        # Normalize coordinates to [0, 1] range
        normalized = (vertex - min_coords) / (ranges + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Create smooth color mapping
        # Use X coordinate for hue (color), Y for saturation, Z for value
        hue = normalized[0] * 0.8  # Use 80% of hue spectrum (avoids red-to-red wrap)
        saturation = 0.6 + normalized[1] * 0.4  # Saturation from 0.6 to 1.0
        value = 0.7 + normalized[2] * 0.3  # Value from 0.7 to 1.0
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = list(rgb)
    
    return colors


def apply_correspondence_colors(source_mesh, target_mesh, correspondences):
    """Apply uniform coloring to source mesh, then copy colors to target via correspondences."""
    source_vertices = np.asarray(source_mesh.vertices)
    target_vertices = np.asarray(target_mesh.vertices)
    
    # Step 1: Generate smooth, uniform colors for the entire source mesh
    print("Generating smooth colors for source mesh...")
    source_colors = generate_smooth_source_colors(source_vertices)
    
    # Step 2: Initialize target mesh with neutral gray
    target_colors = np.full((len(target_vertices), 3), [0.8, 0.8, 0.8])  # Light gray base
    
    # Step 3: Copy colors from source to target based on ALL correspondences
    print(f"Copying colors for all {len(correspondences)} correspondences...")
    
    # Copy colors directly from source to target
    colored_count = 0
    for source_idx, target_idx in correspondences:
        # Check bounds
        if (source_idx < len(source_vertices) and target_idx < len(target_vertices) and
            source_idx >= 0 and target_idx >= 0):
            
            # Copy the source color to the target vertex
            target_colors[target_idx] = source_colors[source_idx]
            colored_count += 1
    
    # Step 4: Smooth colors on both meshes for natural appearance
    print("Applying color smoothing to source mesh...")
    source_colors = smooth_vertex_colors(source_mesh, source_colors, iterations=3)
    
    if colored_count > 0:
        print("Applying color smoothing to target mesh...")
        target_colors = smooth_vertex_colors(target_mesh, target_colors, iterations=3)
    
    # Apply colors to meshes
    source_mesh.vertex_colors = o3d.utility.Vector3dVector(source_colors)
    target_mesh.vertex_colors = o3d.utility.Vector3dVector(target_colors)
    
    print(f"Applied uniform source coloring and copied {colored_count} correspondence colors to target")
    return colored_count



def position_meshes(source_mesh, target_mesh, separation_factor=1.5, axis='x', center_meshes=True):
    """Position meshes for better visualization."""
    if center_meshes:
        # Center both meshes at origin
        source_bbox = source_mesh.get_axis_aligned_bounding_box()
        target_bbox = target_mesh.get_axis_aligned_bounding_box()
        
        source_center = source_bbox.get_center()
        target_center = target_bbox.get_center()
        
        source_mesh.translate(-source_center)
        target_mesh.translate(-target_center)
    
    # Calculate separation distance based on mesh sizes
    source_bbox = source_mesh.get_axis_aligned_bounding_box()
    target_bbox = target_mesh.get_axis_aligned_bounding_box()
    
    source_size = source_bbox.get_extent()
    target_size = target_bbox.get_extent()
    
    # Use the maximum dimension of both meshes to calculate separation
    max_dimension = max(max(source_size), max(target_size))
    translation_distance = max_dimension * separation_factor
    
    # Create translation vector based on chosen axis
    translation = [0, 0, 0]
    axis_names = {'x': 0, 'y': 1, 'z': 2}
    
    if axis.lower() in axis_names:
        axis_idx = axis_names[axis.lower()]
        
        # Position source mesh on negative side, target on positive side
        translation[axis_idx] = translation_distance / 2
        source_mesh.translate([-translation[0], -translation[1], -translation[2]])
        target_mesh.translate(translation)
        
        print(f"Positioned meshes {translation_distance:.2f} units apart along {axis.upper()}-axis")
    else:
        print(f"Invalid axis '{axis}', using X-axis")
        source_mesh.translate([-translation_distance/2, 0, 0])
        target_mesh.translate([translation_distance/2, 0, 0])


def visualize_correspondences(source_mesh, target_mesh, correspondences, separation=1.5, axis='x', center=True):
    """Main visualization function."""
    print("Setting up visualization...")
    
    # Apply color-coding to vertices based on correspondences
    colored_count = apply_correspondence_colors(source_mesh, target_mesh, correspondences)
    
    # Position meshes for better viewing
    position_meshes(source_mesh, target_mesh, separation, axis, center)
    
    # Just use the meshes - no connection lines needed
    geometries = [source_mesh, target_mesh]
    
    print("Opening visualization window...")
    print("Controls: Mouse to rotate, scroll to zoom, Shift+click to pan")
    print("Color guide:")
    print("  • Source mesh: Smooth color gradient based on spatial position")
    print("  • Target mesh: Regions painted with colors from corresponding source vertices")
    print("  • Matching colors: Same colors indicate corresponding regions")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Animal Registration - Color-Coded Correspondences",
        width=1400,
        height=900
    )


def find_result_files(result_dir):
    """Find mesh files and correspondences in a result directory."""
    result_path = Path(result_dir)
    
    if not result_path.exists():
        raise ValueError(f"Result directory does not exist: {result_dir}")
    
    # Find correspondences.json
    correspondences_file = result_path / "correspondences.json"
    if not correspondences_file.exists():
        raise ValueError(f"correspondences.json not found in {result_dir}")
    
    # Find meshes directory
    meshes_dir = result_path / "meshes"
    if not meshes_dir.exists():
        raise ValueError(f"meshes directory not found in {result_dir}")
    
    # Find mesh files
    mesh_files = list(meshes_dir.glob("*.obj"))
    if len(mesh_files) < 2:
        raise ValueError(f"Need at least 2 mesh files, found {len(mesh_files)} in {meshes_dir}")
    
    # Sort mesh files by name to ensure consistent ordering
    mesh_files.sort(key=lambda x: x.name)
    
    source_mesh = str(mesh_files[0])
    target_mesh = str(mesh_files[1])
    
    result = {
        'source_mesh': source_mesh,
        'target_mesh': target_mesh,
        'correspondences': str(correspondences_file),
        'mesh_count': len(mesh_files)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Visualize correspondences from a result directory with color-coded vertices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Color-Coding System:
    • Source mesh: Smooth color gradient across entire surface
    • Target mesh: Gray background with colored regions from source
    • Matching colors: Same colors indicate corresponding regions

Examples:
    python vis_result.py result/1_to_2_20250929_190722
    python vis_result.py result/horse_000040_to_horse_000283_20250929_184634 --front-back
    python vis_result.py result/my_source_to_my_target_20250929_185405 --side-by-side
        """
    )
    
    parser.add_argument('result_dir', help='Path to result directory')
    parser.add_argument('--separation', type=float, default=1.5,
                       help='Distance factor between meshes (default: 1.5)')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='x',
                       help='Axis to separate meshes along (default: x)')
    parser.add_argument('--no-center', action='store_true',
                       help='Do not center meshes at origin before positioning')
    parser.add_argument('--side-by-side', action='store_true', 
                       help='Place meshes side-by-side (equivalent to --axis x)')
    parser.add_argument('--front-back', action='store_true',
                       help='Place meshes front-to-back (equivalent to --axis z)')
    parser.add_argument('--top-bottom', action='store_true', 
                       help='Place meshes top-to-bottom (equivalent to --axis y)')
    
    args = parser.parse_args()
    
    # Handle convenience positioning flags
    axis = args.axis
    if args.side_by_side:
        axis = 'x'
    elif args.front_back:
        axis = 'z'
    elif args.top_bottom:
        axis = 'y'
    
    center_meshes = not args.no_center
    
    print("=" * 60)
    print("Animal Registration - Result Visualization")
    print("=" * 60)
    print(f"Result directory: {args.result_dir}")
    print(f"Positioning: {args.separation:.1f}x separation along {axis.upper()}-axis")
    if center_meshes:
        print("Meshes will be centered at origin")
    
    try:
        # Find all files in result directory
        print("Searching result directory...")
        result_files = find_result_files(args.result_dir)
        
        print(f"Found {result_files['mesh_count']} mesh files")
        print(f"Source mesh: {Path(result_files['source_mesh']).name}")
        print(f"Target mesh: {Path(result_files['target_mesh']).name}")
        print(f"Correspondences: {Path(result_files['correspondences']).name}")
        
        # Load meshes
        print("Loading meshes...")
        source_mesh = load_mesh(result_files['source_mesh'])
        if source_mesh is None:
            return 1
        
        target_mesh = load_mesh(result_files['target_mesh'])
        if target_mesh is None:
            return 1
        
        # Load correspondences
        print("Loading correspondences...")
        correspondences = load_correspondences(result_files['correspondences'])
        if correspondences is None or len(correspondences) == 0:
            print("No valid correspondences found!")
            return 1
        
        # Visualize
        visualize_correspondences(source_mesh, target_mesh, correspondences, 
                                args.separation, axis, center_meshes)
        print("Visualization complete!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
