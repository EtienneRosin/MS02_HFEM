import meshio
import gmsh
import pygmsh
import numpy as np
import os
from typing import Optional, Union, Tuple

from hfem.mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh

def rectangular_mesh(
    save_name: str, 
    h: float = 0.1, 
    L_x: float = 2.0, 
    L_y: float = 1.0, 
    output_directory: Optional[str] = None,
    boundary_labels: Optional[dict] = None
) -> str:
    """
    Create a rectangular mesh using PyGMSH with flexible configuration.

    Parameters
    ----------
    save_name : str
        Base filename for the mesh file
    h : float, optional
        Mesh element size (default: 0.1)
    L_x : float, optional
        Domain width (default: 2.0)
    L_y : float, optional
        Domain height (default: 1.0)
    output_directory : Optional[str], optional
        Directory to save the mesh file
    boundary_labels : Optional[dict], optional
        Custom boundary labels

    Returns
    -------
    str
        Full path to the generated mesh file
    """
    # Logging and information
    print(f"Creating a rectangular mesh of size {L_x:.1e} x {L_y:.1e} with element size h = {h:.1e}.")

    # Define default boundary labels if not provided
    # default_boundary_labels = {
    #     'bottom': r'\partial\Omega_{\text{bottom}}',
    #     'right': r'\partial\Omega_{\text{right}}',
    #     'top': r'\partial\Omega_{\text{top}}',
    #     'left': r'\partial\Omega_{\text{left}}',
    #     'domain': r'\Omega'
    # }
    
    # default_boundary_labels = {
    #     'bottom': r'\partial\Omega',
    #     'right': r'\partial\Omega',
    #     'top': r'\partial\Omega',
    #     'left': r'\partial\Omega',
    #     'domain': r'\Omega'
    # }
    default_boundary_labels = {
        'boundary': r'\partial\Omega',
        'domain': r'\Omega'
    }
    
    boundary_labels = boundary_labels or default_boundary_labels

    # Prepare output directory
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        full_path = os.path.join(output_directory, save_name)
    else:
        full_path = save_name

    # Ensure .msh extension
    if not full_path.endswith('.msh'):
        full_path += '.msh'

    points_definition = [
        (0, 0, 0),    # Bottom-left
        (L_x, 0, 0),  # Bottom-right
        (L_x, L_y, 0),# Top-right
        (0, L_y, 0)   # Top-left
    ]

    # Initialize geometry using PyGMSH context manager
    with pygmsh.geo.Geometry() as geometry:
        model = geometry.__enter__()
        
        # Add points with specified mesh size
        points = [model.add_point(point, mesh_size=h) for point in points_definition]
        
        # Create boundary lines with specific labels
        # lines = {
        #     'bottom': model.add_line(points[0], points[1]),
        #     'right':  model.add_line(points[1], points[2]),
        #     'top':    model.add_line(points[2], points[3]),
        #     'left':   model.add_line(points[3], points[0])
        # }
        lines = [
            model.add_line(points[0], points[1]),
            model.add_line(points[1], points[2]),
            model.add_line(points[2], points[3]),
            model.add_line(points[3], points[0])
        ]
        
        # Create line loop and surface
        # line_loop = model.add_curve_loop([lines[side] for side in ['bottom', 'right', 'top', 'left']])
        line_loop = model.add_curve_loop([line for line in lines])
        surface = model.add_plane_surface(line_loop)
        
        # Synchronize model
        model.synchronize()
        
        # Add physical groups
        model.add_physical([surface], boundary_labels['domain'])
        
        # Add physical boundary groups
        # for side, line in lines.items():
        #     model.add_physical([line], boundary_labels[side])
        # for line in lines:
        #     model.add_physical([line], boundary_labels['boundary'])
        model.add_physical([*lines], boundary_labels['boundary'])
        
        # Generate mesh
        geometry.generate_mesh(dim=2)
        
        # Write mesh file
        gmsh.write(full_path)
        
        # Clear GMSH resources
        gmsh.clear()

    print(f"Rectangular mesh created and saved to {full_path}")
    return full_path

if __name__ == '__main__':
    # Examples of mesh generation
    # Single mesh generation
    mesh_path = rectangular_mesh('rectangle.msh')
    mesh_rectangle = CustomTwoDimensionMesh(filename=mesh_path)
    mesh_rectangle.display()
    mesh_rectangle.export_enhanced_info()
    