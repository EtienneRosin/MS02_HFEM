"""
CustomTwoDimensionMesh: An advanced 2D mesh handling class with enhanced capabilities.

This module provides a comprehensive mesh processing utility for 2D geometries,
offering advanced functionality for mesh manipulation, visualization, 
and analysis.
"""

import meshio
import numpy as np
import json
import jsbeautifier
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from typing import Union, List, Optional, Tuple, Dict, Any, Literal
import logging
import cmasher as cmr
from dataclasses import dataclass, asdict
import warnings

# Custom Exception Hierarchy
class MeshError(Exception):
    """Base exception for mesh-related errors"""
    pass

class MeshValidationError(MeshError):
    """Raised when mesh validation fails"""
    pass

class MeshDimensionError(MeshValidationError):
    """Raised when mesh dimension is incorrect"""
    pass

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if np.issubdtype(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class MeshMetadata:
    """Comprehensive metadata for mesh"""
    filename: str
    creation_date: str
    source: Optional[str] = None
    preprocessing_steps: List[str] = None

def custom_serializer(obj: Any) -> Any:
    """Enhanced JSON serializer for handling non-serializable objects."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, meshio.CellBlock):
        return {
            "type": obj.type,
            "data": obj.data.tolist(),
            "num_cells": len(obj.data),
            "dim": obj.dim,
            "tags": obj.tags
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class CustomTwoDimensionMesh(meshio.Mesh):
    """
    Advanced 2D mesh processing class with comprehensive capabilities.

    Extends meshio.Mesh with enhanced functionality for mesh handling.
    """

    def __init__(
        self, 
        filename: str, 
        reordering: bool = True, 
        log_level: int = logging.INFO,
        metadata: Optional[MeshMetadata] = None
    ):
        """
        Initialize the CustomTwoDimensionMesh with advanced configuration.

        Parameters
        ----------
        filename : str
            Path to the mesh file
        reordering : bool, optional
            Whether to reorder mesh elements
        log_level : int, optional
            Logging level
        metadata : MeshMetadata, optional
            Additional metadata about the mesh
        """
        # Enhanced logging configuration
        # logging.basicConfig(
        #     level=log_level, 
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S'
        # )
        self.logger = logging.getLogger(__name__)
        
        # Metadata handling
        self.metadata = metadata or MeshMetadata(
            filename=filename, 
            creation_date=str(np.datetime64('today')),
            preprocessing_steps=[]
        )

        try:
            # Load mesh with enhanced error handling
            mesh = self._load_mesh(filename)
            super().__init__(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=mesh.cell_data,
                field_data=mesh.field_data,
                point_sets=mesh.point_sets,
                cell_sets=mesh.cell_sets,
                gmsh_periodic=mesh.gmsh_periodic,
                info=mesh.info
            )
            
            # Comprehensive mesh validation and initialization
            self._validate_mesh()
            self._initialize_attributes()
            self._assign_references_to_nodes()
            
            # Optional element reordering with enhanced logging
            if reordering:
                self._reorder_elements()
                self.metadata.preprocessing_steps.append("Element Reordering")
                
            if self.tri_nodes.size > 0:
                edges = np.vstack([
                    self.tri_nodes[:, [0, 1]],
                    self.tri_nodes[:, [1, 2]],
                    self.tri_nodes[:, [2, 0]]
                ])
                edge_vectors = self.node_coords[edges[:, 1]] - self.node_coords[edges[:, 0]]
                self.h = np.median(np.sqrt(np.sum(edge_vectors**2, axis=1)))
            else:
                self.h = None

        except Exception as e:
            self.logger.error(f"Mesh initialization failed: {e}")
            raise MeshValidationError(f"Mesh initialization error: {e}")

    def _load_mesh(self, filename: str) -> meshio.Mesh:
        """
        Enhanced mesh loading with additional checks.

        Parameters
        ----------
        filename : str
            Path to mesh file

        Returns
        -------
        meshio.Mesh
            Loaded mesh object
        """
        try:
            mesh = meshio.read(filename)
            # self.logger.info(f"Successfully loaded mesh from {filename}")
            return mesh
        except FileNotFoundError:
            self.logger.error(f"Mesh file not found: {filename}")
            raise
        except Exception as e:
            self.logger.error(f"Mesh loading error: {e}")
            raise MeshValidationError(f"Cannot load mesh: {e}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CustomTwoDimensionMesh':
        """
        Alternate constructor for flexible mesh initialization.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for mesh initialization

        Returns
        -------
        CustomTwoDimensionMesh
            Initialized mesh object
        """
        required_keys = ['filename']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        metadata = MeshMetadata(**config.get('metadata', {}))
        
        return cls(
            filename=config['filename'],
            reordering=config.get('reordering', True),
            log_level=config.get('log_level', logging.INFO),
            metadata=metadata
        )

    def export_enhanced_info(
        self, 
        filename: str = "mesh_info.json", 
        include_full_data: bool = False
    ) -> None:
        """
        Enhanced mesh information export with comprehensive details.

        Parameters
        ----------
        filename : str, optional
            Output JSON filename
        include_full_data : bool, optional
            Whether to include full mesh data
        """
        try:
            export_data = {
                'metadata': asdict(self.metadata),
                'summary': {
                    'num_nodes': self.num_nodes,
                    'num_triangles': self.num_triangles,
                    'num_edges': self.num_edges,
                    'references': list(map(int, self.refs)),
                    'labels': self.labels
                }
            }

            if include_full_data:
                export_data['full_mesh_data'] = self.__dict__

            options = jsbeautifier.default_options()
            options.indent_size = 4
            
            formatted_json = jsbeautifier.beautify(
                json.dumps(export_data, cls=NumpyEncoder), 
                options
            )
            
            with open(filename, "w") as json_file:
                json_file.write(formatted_json)
            
            self.logger.info(f"Enhanced mesh info saved to {filename}")
        
        except Exception as e:
            self.logger.error(f"Failed to save enhanced mesh info: {e}")
            raise
    
    def _validate_mesh(self) -> None:
        """
        Comprehensive mesh validation.

        Checks:
        - Mesh dimension is 2
        - Mesh is not empty
        - Consistency between cell and physical references
        """
        try:
            # Dimension validation
            elements_max_dimension = np.max([data[1] for data in self.field_data.values()])
            if elements_max_dimension != 2:
                raise ValueError(f"Mesh dimension must be 2, got {elements_max_dimension}")

            # Empty mesh check
            if len(self.points) == 0:
                raise ValueError("Mesh is empty")

            # Reference consistency check
            triangle_cells = self.cells_dict.get('triangle', [])
            triangle_refs = self.cell_data_dict.get('gmsh:physical', {}).get('triangle', [])
            
            if len(triangle_cells) > 0 and len(triangle_refs) == 0:
                self.logger.warning("Triangle cells exist but no physical references found")

        except Exception as e:
            self.logger.error(f"Mesh validation failed: {e}")
            raise

    def _initialize_attributes(self) -> None:
        r"""Initialize the custom mesh attributes."""
        self.node_coords = self.points[:, :-1]
        self.num_nodes = len(self.node_coords)
        self.node_refs = np.zeros(self.num_nodes, dtype=int)

        self.tri_nodes = self.cells_dict.get('triangle', np.array([]))
        self.num_triangles = len(self.tri_nodes)
        self.tri_refs = self.cell_data_dict.get('gmsh:physical', {}).get('triangle', np.array([])).astype(int)

        self.edge_nodes = self.cells_dict.get('line', np.array([]))
        self.num_edges = len(self.edge_nodes)
        self.edge_refs = self.cell_data_dict.get('gmsh:physical', {}).get('line', np.array([])).astype(int)

        self.refs = np.sort(np.array([data[0] for data in self.field_data.values()]))
        self.labels = {f"${field}$": data[0] for field, data in self.field_data.items()}

    def _reorder_elements(self) -> None:
        r"""Reorder the mesh elements (nodes, edges, and triangles) so that the nodes are ordered by their reference.

        Notes
        -----
        Reordering steps:
            1. Sort the node indices by their reference and reorder them.
            2. Create a correspondence table between old and new indices.
            3. Reorder the edges and triangles.
        """
        
        if len(self.node_coords) != len(self.node_refs):
            raise ValueError("Mismatch between node coordinates and node references length.")
        
        # Step 1: Sort the node indices based on self.node_refs
        sort_indices = np.argsort(self.node_refs)  # Indices sorted in ascending order of node_refs
        
        # Reorder the node coordinates and node references
        self.node_coords = self.node_coords[sort_indices]
        self.node_refs = self.node_refs[sort_indices]

        # Step 2: Create a correspondence table from old indices to new indices
        inverse_indices = np.zeros_like(sort_indices)
        inverse_indices[sort_indices] = np.arange(len(sort_indices))

        # Step 3: Update the indices in self.tri_nodes and self.edge_nodes
        self.tri_nodes = inverse_indices[self.tri_nodes]
        self.edge_nodes = inverse_indices[self.edge_nodes]
        

    def _assign_references_to_nodes(self) -> None:
        r"""Assign a physical reference to the nodes."""
        for nodes, tag in zip(self.tri_nodes, self.tri_refs):
            self.node_refs[nodes] = tag
        for nodes, tag in zip(self.edge_nodes, self.edge_refs):
            self.node_refs[nodes] = tag

    def _create_colors(self, color_map: str = 'cmr.lavender') -> np.ndarray:
        r"""Create a color list from the physical references of the mesh.

        Parameters
        ----------
        color_map: str, default 'cmr.lavender'
            color map to use
            
        Returns
        -------
        colors: np.ndarray
            list of color based on the physical references.
        
        Raises
        ------
        ValueError
            if no physical reference is found.
        """
        if len(self.refs) == 0:
            raise ValueError("No references found to generate colors.")
        norm = plt.Normalize(self.refs.min(), self.refs.max())
        colors = plt.get_cmap(color_map)(norm(self.refs))
        return colors
    
    def _plot_elements(
        self, 
        ax: plt.Axes, 
        element_type: str, 
        element_nodes: np.ndarray, 
        colors: np.ndarray, 
        alpha: float, 
        lw: float) -> None:
        r"""Plot mesh elements (edges or triangles) on a axes.

        Parameters
        ----------
        ax: plt.Axes
            Axes to plot on.
        element_type: str
            The type of element ('edges' or 'triangles') to plot.
        element_nodes: np.ndarray
            Node indices of the elements (edges or triangles).
        colors: np.ndarray
            Color for each element.
        alpha: float
            Transparency level for the elements (used for triangles).
        lw: float
            Line width for the edges or triangle borders.
        
        Raises
        ------
        ValueError
            if the element is not 'edges' or 'triangles'.
        """
        vertices = np.stack((self.node_coords[element_nodes, 0], self.node_coords[element_nodes, 1]), axis=-1)
        match element_type:
            case 'edges':
                collection = LineCollection(vertices, lw=lw)
                collection.set_colors(colors)
                
            case 'triangles':
                collection = PolyCollection(vertices, edgecolor='k', alpha=alpha, lw=lw)
                collection.set_facecolor(colors)
            case _ :
                raise ValueError("Element should be edge_nodes or tri_nodes.")
            
        ax.add_collection(collection)

    # def display_nodes(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     colors = self._create_colors(color_map=color_map)
    #     scatter = ax.scatter(*self.node_coords.T, c = colors[self.node_refs - 1], s=8)
    #     if show:
    #         self._add_legend(ax, colors)
    #         ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
    #         plt.show()

    # def display_triangles(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     colors = self._create_colors(color_map=color_map)
    #     tri_colors = colors[self.tri_refs - 1]
    #     self._plot_elements(ax = ax, element_type = 'triangles', element_nodes = self.tri_nodes, colors=tri_colors, alpha=0.5, lw=0.5)
    #     if show:
    #         self._add_legend(ax, colors)
    #         ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
    #         plt.show()

    # def display_edges(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     colors = self._create_colors(color_map=color_map)
    #     edge_colors = colors[self.edge_refs - 1]
    #     self._plot_elements(ax=ax, element_type = 'edges', element_nodes=self.edge_nodes, colors=edge_colors, alpha=1, lw=1)

    #     if show:
    #         self._add_legend(ax, colors)
    #         ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
    #         plt.show()

    def _add_legend(self, ax: plt.Axes, colors: np.ndarray):
        custom_lines = [Line2D([0], [0], color=colors[i - 1], lw=2) for i in self.labels.values()]
        ax.legend(custom_lines, 
            list(self.labels.keys()),bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=len(self.labels.keys()),)

        ax.autoscale_view()

    # def display(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     colors = self._create_colors(color_map=color_map)
        
    #     self.display_triangles(ax=ax, color_map=color_map, show=False)
    #     self.display_edges(ax=ax, color_map=color_map, show=False)
    #     if show:
    #         self._add_legend(ax, colors)
    #         ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
    #         plt.show()
    
    def display_nodes(self, 
                  ax: Optional[plt.Axes] = None, 
                  color_map: str = 'cmr.lavender', 
                  show: bool = True,
                  marker_size: int = 8,
                  marker_style: str = 'o',
                  title: Optional[str] = "Mesh Nodes",
                  grid: bool = False):
        """
        Enhanced node display method with more customization options.

        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on
        color_map : str, optional
            Colormap for node coloration
        show : bool, optional
            Whether to show the plot immediately
        marker_size : int, optional
            Size of scatter plot markers
        marker_style : str, optional
            Matplotlib marker style
        title : Optional[str], optional
            Plot title
        grid : bool, optional
            Whether to display grid
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = self._create_colors(color_map=color_map)
        scatter = ax.scatter(
            *self.node_coords.T, 
            c=colors[self.node_refs - 1], 
            s=marker_size, 
            marker=marker_style
        )
        
        ax.set_title(title)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.5)
        
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.tight_layout()
            plt.show()
        
        return ax

    def display_triangles(self, 
                        ax: Optional[plt.Axes] = None, 
                        color_map: str = 'cmr.lavender', 
                        show: bool = True,
                        alpha: float = 0.5,
                        edge_color: str = 'k',
                        edge_linewidth: float = 0.5,
                        title: Optional[str] = "Mesh Triangles",
                        grid: bool = False):
        """
        Enhanced triangle display method with more visualization options.

        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on
        color_map : str, optional
            Colormap for triangle coloration
        show : bool, optional
            Whether to show the plot immediately
        alpha : float, optional
            Transparency of triangle fill
        edge_color : str, optional
            Color of triangle edges
        edge_linewidth : float, optional
            Linewidth of triangle edges
        title : Optional[str], optional
            Plot title
        grid : bool, optional
            Whether to display grid
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = self._create_colors(color_map=color_map)
        tri_colors = colors[self.tri_refs - 1]
        
        self._plot_elements(
            ax=ax, 
            element_type='triangles', 
            element_nodes=self.tri_nodes, 
            colors=tri_colors, 
            alpha=alpha, 
            lw=edge_linewidth
        )
        
        # Set edge color if specified
        for collection in ax.collections:
            collection.set_edgecolor(edge_color)
        
        ax.set_title(title)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.5)
        
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.tight_layout()
            plt.show()
        
        return ax

    def display_edges(self, 
                    ax: Optional[plt.Axes] = None, 
                    color_map: str = 'cmr.lavender', 
                    show: bool = True,
                    linewidth: float = 1.0,
                    linestyle: str = '-',
                    title: Optional[str] = "Mesh Edges",
                    grid: bool = False):
        """
        Enhanced edge display method with more visualization options.

        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on
        color_map : str, optional
            Colormap for edge coloration
        show : bool, optional
            Whether to show the plot immediately
        linewidth : float, optional
            Width of edge lines
        linestyle : str, optional
            Style of edge lines
        title : Optional[str], optional
            Plot title
        grid : bool, optional
            Whether to display grid
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = self._create_colors(color_map=color_map)
        edge_colors = colors[self.edge_refs - 1]
        
        self._plot_elements(
            ax=ax, 
            element_type='edges', 
            element_nodes=self.edge_nodes, 
            colors=edge_colors, 
            alpha=1, 
            lw=linewidth
        )
        
        # Update line styles for each collection
        for collection in ax.collections:
            collection.set_linestyle(linestyle)
        
        ax.set_title(title)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.5)
        
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.tight_layout()
            plt.show()
        
        return ax

    def display(self, 
                ax: Optional[plt.Axes] = None, 
                color_map: str = 'cmr.lavender', 
                show: bool = True,
                triangles_alpha: float = 0.5,
                edge_color: str = 'k',
                edge_linewidth: float = 0.5,
                title: Optional[str] = None,
                grid: bool = False):
        """
        Comprehensive mesh display method with advanced configuration.

        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on
        color_map : str, optional
            Colormap for element coloration
        show : bool, optional
            Whether to show the plot immediately
        triangles_alpha : float, optional
            Transparency of triangle fill
        edge_color : str, optional
            Color of triangle edges
        edge_linewidth : float, optional
            Linewidth of triangle edges
        title : Optional[str], optional
            Plot title
        grid : bool, optional
            Whether to display grid
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        colors = self._create_colors(color_map=color_map)
        
        # Display triangles and edges
        self.display_triangles(
            ax=ax, 
            color_map=color_map, 
            show=False, 
            alpha=triangles_alpha,
            edge_color=edge_color, 
            edge_linewidth=edge_linewidth
        )
        self.display_edges(ax=ax, color_map=color_map, show=False)
        
        ax.set_title(title)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.5)
        
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.tight_layout()
            plt.show()
        
        return ax

    def get_corner_and_boundary_pairs(
        self, 
        boundary_labels: Union[str, List[str]] = '$\\partial\\Omega$', 
        tolerance: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Enhanced method to get corner and boundary node pairs.

        Parameters
        ----------
        boundary_labels : str or list of str, optional
            Labels indicating the boundaries
        tolerance : float, optional
            Tolerance for coordinate comparison

        Returns
        -------
        corner_indices : np.ndarray
            Indices of corner nodes
        pairs_same_x : np.ndarray
            Pairs of boundary nodes with same x coordinate
        pairs_same_y : np.ndarray
            Pairs of boundary nodes with same y coordinate
        inner_indices : np.ndarray
            Indices of internal nodes
        """
        if isinstance(boundary_labels, str):
            boundary_labels = [boundary_labels]

        # Collect boundary indices
        # print(f"{boundary_labels = }")
        boundary_indices = np.hstack([np.where(self.node_refs == self.labels[label])[0] for label in boundary_labels])
        border_nodes = self.node_coords[boundary_indices]

        # Determine internal node indices
        inner_indices = np.setdiff1d(np.arange(self.node_coords.shape[0]), boundary_indices)

        x_min = border_nodes[:, 0].min()
        x_max = border_nodes[:, 0].max()
        y_min = border_nodes[:, 1].min()
        y_max = border_nodes[:, 1].max()

        left_indices = np.where(self.node_coords[:, 0] == x_min)[0]
        right_indices = np.where(self.node_coords[:, 0] == x_max)[0]
        bottom_indices = np.where(self.node_coords[:, 1] == y_min)[0]
        top_indices = np.where(self.node_coords[:, 1] == y_max)[0]

        bottom_left = np.where(np.logical_and(self.node_coords[:, 0] == x_min, self.node_coords[:, 1] == y_min))[0]
        bottom_right = np.where(np.logical_and(self.node_coords[:, 0] == x_max, self.node_coords[:, 1] == y_min))[0]
        top_left = np.where(np.logical_and(self.node_coords[:, 0] == x_min, self.node_coords[:, 1] == y_max))[0]
        top_right = np.where(np.logical_and(self.node_coords[:, 0] == x_max, self.node_coords[:, 1] == y_max))[0]

        corner_indices = np.concatenate((bottom_left, bottom_right, top_left, top_right))

        non_corner_indices = np.setdiff1d(boundary_indices, corner_indices)
        non_corner_left_indices = np.setdiff1d(left_indices, corner_indices)
        non_corner_right_indices = np.setdiff1d(right_indices, corner_indices)
        non_corner_bottom_indices = np.setdiff1d(bottom_indices, corner_indices)
        non_corner_top_indices = np.setdiff1d(top_indices, corner_indices)

        pairs_same_x = [(i, j) for i in non_corner_bottom_indices for j in non_corner_top_indices if np.abs(self.node_coords[i, 0] - self.node_coords[j, 0]) <= tolerance]
        pairs_same_y = [(i, j) for i in non_corner_right_indices for j in non_corner_left_indices if np.abs(self.node_coords[i, 1] - self.node_coords[j, 1]) <= tolerance]
        
        return corner_indices, np.array(pairs_same_x), np.array(pairs_same_y), inner_indices


    def display_corner_and_boundary_pairs(self, boundary_labels: str|list[str] = '$\\partial\\Omega$', save_name: str = None) -> None:
        if isinstance(boundary_labels, str):
            boundary_labels = [boundary_labels]
        corner_indices, pairs_same_x, pairs_same_y, inner_indices = self.get_corner_and_boundary_pairs(boundary_labels)
        boundary_indices = np.hstack([np.where(self.node_refs == self.labels[label])[0] for label in boundary_labels])
        with plt.style.context('science' if save_name else 'default'):
            fig, ax = plt.subplots()
        
            for (i, j) in pairs_same_x:
                ax.plot(*zip(*self.node_coords[[i, j]]), color="blue", alpha = 0.5)
            
            for (i, j) in pairs_same_y:
                ax.plot(*zip(*self.node_coords[[i, j]]), color="green", alpha = 0.5)

            # ax.legend(loc='upper right')
            ax.scatter(*self.node_coords[boundary_indices].T, label="border nodes", zorder = 2, s = 8)
            ax.scatter(*self.node_coords[inner_indices].T, label="inner nodes", s = 8)
            
            ax.scatter(*self.node_coords[corner_indices].T, label="corner nodes", zorder = 2, s = 8)
            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
            ax.set(xlabel = "$x$", ylabel = "$y$", aspect = "equal")
            if save_name:
                fig.savefig(f"{save_name}.pdf")
        plt.show()

# Example usage remains similar to previous implementation
if __name__ == "__main__":
    filename = "meshes/test_mesh.msh"
    try:
        mesh = CustomTwoDimensionMesh(filename=filename)
        mesh.export_enhanced_info()
        mesh.display()
        mesh.display_corner_and_boundary_pairs()
        print(f"{mesh.h = }")
        # Demonstrate new capabilities
    
    except MeshError as e:
        print(f"Mesh processing error: {e}")