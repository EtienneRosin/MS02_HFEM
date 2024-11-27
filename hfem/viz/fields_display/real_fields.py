"""
Advanced 2D Mesh and Field Visualization Module

This module provides sophisticated visualization functions for triangular 2D meshes,
supporting contour and surface representations with high customization.

Main Functions:
- display_field_as_contourf: 2D filled contour visualization
- display_field_as_trisurface: 3D triangular surface visualization

Usage Example:
    mesh = CustomTwoDimensionMesh(filename='mesh.msh')
    field = compute_field(mesh)
    display_field_as_contourf(mesh, field)
    display_field_as_trisurface(mesh, field)

Author: Etienne Rosin
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from mesh_manager import CustomTwoDimensionMesh
from hfem.viz.conditional_style_context import conditional_style_context
from hfem.viz.custom_components import CustomFigure

@conditional_style_context()
def display_field_as_contourf(
    mesh: CustomTwoDimensionMesh,
    field: np.ndarray,
    cmap = 'cmr.lavender',
    ax_props: dict = {'xlabel': r'$x$', 'ylabel': r'$y$', 'aspect': 'equal'},
    cbar: bool = False,
    cbar_props: dict = {},
    save_name: str = None,
    **kwargs
    ) -> None:
    """
    Visualize a field on a 2D triangular mesh using filled contours.

    This function creates a 2D visualization of a field defined on a triangular mesh,
    using interpolated filled contours and optional customization.

    Parameters
    ----------
    mesh : CustomTwoDimensionMesh
        The source triangular 2D mesh containing node coordinates and connectivity
    
    field : np.ndarray
        Field values corresponding to mesh nodes. Should match mesh node count.
    
    cmap : str, optional
        Colormap for field representation. 
        Defaults to 'cmr.lavender' from cmasher library.
    
    ax_props : dict, optional
        Axis properties like labels, aspect ratio.
        Default includes LaTeX-style x and y labels with equal aspect.
    
    cbar : bool, optional
        Flag to display colorbar. Defaults to False.
    
    cbar_props : dict, optional
        Custom colorbar properties for fine-tuning.
    
    save_name : str, optional
        File path to save the generated figure. 
        If provided, figure is saved before display.

    Raises
    ------
    ValueError
        If field size doesn't match mesh node coordinates.
    
    TypeError
        If input types are incorrect.

    Notes
    -----
    - Uses matplotlib's tricontourf for mesh-based interpolation
    - Supports custom figure class from hfem.viz
    - Applies conditional styling via decorator
    """
    # Input validation (recommended addition)
    if not isinstance(mesh, CustomTwoDimensionMesh):
        raise TypeError("mesh must be a CustomTwoDimensionMesh instance")
    
    if field.shape[0] != mesh.node_coords.shape[0]:
        raise ValueError("Field size must match mesh node count")

    # Create figure with custom figure class
    fig = plt.figure(FigureClass=CustomFigure)
    
    # Create subplot with custom properties
    ax = fig.add_subplot(**ax_props)
    
    # Create filled contour plot on triangular mesh
    contour = ax.tricontourf(
        *mesh.node_coords.T,  # Unpacked node coordinates
        mesh.tri_nodes,       # Triangulation connectivity
        field,                # Field values
        cmap=cmap,            # Colormap
        **kwargs
    )
    
    # Optional colorbar
    if cbar or cbar_props:
        cbar = fig.custom_colorbar(contour, ax=ax, **cbar_props)
    
    # Adjust plot aspect
    ax.set_box_aspect(None)
    
    # Save figure if filename provided
    if save_name:
        plt.savefig(save_name)
    
    # Display the plot
    plt.show()

@conditional_style_context()
def display_field_as_trisurface(
    mesh: CustomTwoDimensionMesh,
    field: np.ndarray,
    cmap = 'cmr.lavender',
    ax_props: dict = {'xlabel': r'$x$', 'ylabel': r'$y$', 'aspect': 'equalxy'},
    cbar: bool = False,
    cbar_props: dict = {},
    save_name: str = None,
    view_init: tuple = (40, -30),
    **kwargs
    ) -> None:
    """
    Visualize a field on a 2D triangular mesh using a 3D triangular surface.

    Creates a 3D surface representation of a field defined on a triangular mesh,
    with customizable view and styling options.

    Parameters
    ----------
    mesh : CustomTwoDimensionMesh
        Source triangular 2D mesh with node coordinates
    
    field : np.ndarray
        Field values corresponding to mesh nodes
    
    cmap : str, optional
        Colormap for surface coloration. 
        Defaults to 'cmr.lavender' from cmasher.
    
    ax_props : dict, optional
        3D axis properties like labels and aspect ratio
    
    cbar : bool, optional
        Flag to display colorbar. Defaults to False.
    
    cbar_props : dict, optional
        Custom colorbar configuration
    
    save_name : str, optional
        Path to save generated figure
    
    view_init : tuple, optional
        Initial 3D view angle (elevation, azimuth).
        Defaults to (40, -30) for a standard perspective.

    Raises
    ------
    ValueError
        If field dimensions are incompatible with mesh
    
    Notes
    -----
    - Uses matplotlib's plot_trisurf for 3D mesh interpolation
    - Supports custom figure and styling context
    - Allows precise 3D view angle initialization
    """
    # Figure creation with custom class
    fig = plt.figure(FigureClass=CustomFigure)
    
    # 3D subplot with custom properties
    ax = fig.add_subplot(projection='3d', **ax_props)
    
    # Create 3D triangular surface plot
    trisurf = ax.plot_trisurf(
        *mesh.node_coords.T,  # Node coordinates
        mesh.tri_nodes,       # Triangulation
        field,                # Field values
        cmap=cmap,            # Colormap
        **kwargs
    )
    
    # Optional colorbar
    if cbar or cbar_props:
        cbar = fig.custom_colorbar(trisurf, ax=ax, **cbar_props)
    
    # Adjust plot aspects
    ax.set_box_aspect(None)
    
    # Set initial 3D view
    ax.view_init(*view_init)
    
    # Save figure if requested
    if save_name:
        plt.savefig(save_name)
    
    # Display plot
    plt.show()

# Demonstration block
if __name__ == '__main__':
    # Load a triangular mesh
    fname = 'meshes/coarse_mesh.msh'
    mesh = CustomTwoDimensionMesh(filename=fname)
    
    # Create a test field function
    a = 2
    def v(x, y):
        """
        Generate a sinusoidal test field.
        
        Parameters
        ----------
        x, y : array-like
            Coordinate inputs
        
        Returns
        -------
        np.ndarray
            Computed field values
        """
        return np.sin(a * np.pi * x) * np.sin(a * np.pi * y) + 2
    
    # Compute field on mesh nodes
    field = v(*mesh.node_coords.T)
    
    # Demonstrate 3D surface visualization
    display_field_as_trisurface(
        mesh=mesh, 
        field=field,
        view_init=(45, -45),  # Optional custom view
        cbar_props = {'label': r'$u_h - u$'},
        save_name="hihi.pdf"
    )