import meshio
import numpy as np
import json
import jsbeautifier
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from typing import Optional, Any

import cmasher as cmr
import scienceplots
# plt.style.use('science')

def custom_serializer(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, meshio.CellBlock):
        return {
            "type": obj.type,
            "data": obj.data.tolist(),
            "num_cells": len(obj.data),
            "dim": obj.dim,
            "tags": obj.tags
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")




class CustomTwoDimensionMesh(meshio.Mesh):
    r"""
    Read the information of a 2D gmsh mesh.

    ...

    Attributes
    ----------
    num_nodes: int
        number of mesh's nodes (that we will denote N).
    node_coords: np.ndarray
        coordinates of mesh's nodes : array of shape (N, 2).
    node_refs: np.ndarray
        physical reference tag of the nodes : array of shape (N, 1).
    num_triangles: int
        number of mesh's triangles (that we will denote N_T).
    tri_nodes: np.ndarray
        nodes of each triangle : array of shape (N_T, 3).
    tri_refs: np.ndarray
        physical reference tag of the triangles : array of shape (N_T, 1)

    num_edges: int
        number of mesh's edges (that we will denote N_E)
    edge_nodes: np.ndarray
        nodes of each edge : array of shape (N_E, ...). The ... points are meaning that it depends on the order of the intern approximation
    edge_refs: np.ndarray
        physical reference tag of the edges : array of shape (N_E, 1)

    refs: np.ndarray
        physical references of the subdomains
    labels: dict
        labels of the subdomains
    """
    def __init__(self, filename: str, reordering: bool = True):
        r"""Construct the CustomTwoDimensionMesh object.

        Parameters
        ----------
        filename: str
            file name of the mesh
        reordering: bool, default True
            if True the mesh elements (nodes, edges, and triangles) would be reordered so that the nodes are ordered by their physical reference.
        """
        try:
            mesh = meshio.read(filename)
            super().__init__(
                points = mesh.points,
                cells = mesh.cells,
                point_data = mesh.point_data,
                cell_data = mesh.cell_data,
                field_data = mesh.field_data,
                point_sets = mesh.point_sets,
                cell_sets = mesh.cell_sets,
                gmsh_periodic = mesh.gmsh_periodic,
                info = mesh.info
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} does not exist.")
        except Exception as e:
            raise RuntimeError(f"Error while loading file {filename}: {e}")

        self._validate_mesh()
        self._initialize_attributes()
        self._assign_references_to_nodes()
        if reordering:
            self._reorder_elements()

    def _validate_mesh(self) -> None:
        r"""Validate the mesh dimension.

        Raises
        ------
        ValueError
            if the mesh dimension is not 2.
        """
        elements_max_dimension = np.max([data[1] for data in self.field_data.values()])
        if elements_max_dimension != 2:
            raise ValueError(f"Dimension of the mesh should be 2, but got {elements_max_dimension}.")

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

    def write_info_in_json(self) -> None:
        r"""Save the mesh information in a json file."""
        options = jsbeautifier.default_options()
        options.indent_size = 4
        formatted_json = jsbeautifier.beautify(json.dumps(self.__dict__, default=custom_serializer, sort_keys=False), options)
        with open("mesh_info.json", "w") as json_file:
            json_file.write(formatted_json)

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

    def display_nodes(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        scatter = ax.scatter(*self.node_coords.T, c = colors[self.node_refs - 1], s=8)
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

    def display_triangles(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        tri_colors = colors[self.tri_refs - 1]
        self._plot_elements(ax = ax, element_type = 'triangles', element_nodes = self.tri_nodes, colors=tri_colors, alpha=0.5, lw=0.5)
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

    def display_edges(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        edge_colors = colors[self.edge_refs - 1]
        self._plot_elements(ax=ax, element_type = 'edges', element_nodes=self.edge_nodes, colors=edge_colors, alpha=1, lw=1)

        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

    def _add_legend(self, ax: plt.Axes, colors: np.ndarray):
        custom_lines = [Line2D([0], [0], color=colors[i - 1], lw=2) for i in self.labels.values()]
        ax.legend(custom_lines, 
            list(self.labels.keys()),bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=len(self.labels.keys()),)

        ax.autoscale_view()

    def display(self, ax: Optional[plt.Axes] = None, color_map: str = 'cmr.lavender', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        
        self.display_triangles(ax=ax, color_map=color_map, show=False)
        self.display_edges(ax=ax, color_map=color_map, show=False)
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()


    # def get_corner_and_boundary_pairs(self, boundary_labels: str|list[str] = '$\\partial\\Omega$', tolerance: float = 1e-10) -> tuple[np.ndarray]:
    #     """
    #     Get the indices of corner nodes and pairs of boundary nodes with same x and y values.
        
    #     Parameters
    #     ----------
    #     boundary_labels : str or list of str, default '$\\partial\\Omega$'
    #         List of labels indicating the boundaries.
    #     tolerance: float, default 1e-10
    #         Tolerance for the equality test.

    #     Returns
    #     -------
    #     corner_indices : np.ndarray
    #         Indices of the corner nodes.
    #     pairs_same_x : np.ndarray
    #         Pairs of indices of boundary nodes with the same x value.
    #     pairs_same_y : np.ndarray
    #         Pairs of indices of boundary nodes with the same y value.
    #     """
    #     if isinstance(boundary_labels, str):
    #         boundary_labels = [boundary_labels]
    #     boundary_indices = np.hstack([np.where(self.node_refs == self.labels[label])[0] for label in boundary_labels])
    #     border_nodes = self.node_coords[boundary_indices]

    #     inner_indices= np.setdiff1d(np.arange(self.node_coords.shape[0]), boundary_indices)
    #     # inner_indices = np.where(self.node_refs != boundary_indices)[0]
    #     # print(inner_indices)
    #     x_min = border_nodes[:, 0].min()
    #     x_max = border_nodes[:, 0].max()
    #     y_min = border_nodes[:, 1].min()
    #     y_max = border_nodes[:, 1].max()

    #     left_indices = np.where(self.node_coords[:, 0] == x_min)[0]
    #     right_indices = np.where(self.node_coords[:, 0] == x_max)[0]
    #     bottom_indices = np.where(self.node_coords[:, 1] == y_min)[0]
    #     top_indices = np.where(self.node_coords[:, 1] == y_max)[0]

    #     bottom_left = np.where(np.logical_and(self.node_coords[:, 0] == x_min, self.node_coords[:, 1] == y_min))[0]
    #     bottom_right = np.where(np.logical_and(self.node_coords[:, 0] == x_max, self.node_coords[:, 1] == y_min))[0]
    #     top_left = np.where(np.logical_and(self.node_coords[:, 0] == x_min, self.node_coords[:, 1] == y_max))[0]
    #     top_right = np.where(np.logical_and(self.node_coords[:, 0] == x_max, self.node_coords[:, 1] == y_max))[0]

    #     corner_indices = np.concatenate((bottom_left, bottom_right, top_left, top_right))

    #     non_corner_indices = np.setdiff1d(boundary_indices, corner_indices)
    #     non_corner_left_indices = np.setdiff1d(left_indices, corner_indices)
    #     non_corner_right_indices = np.setdiff1d(right_indices, corner_indices)
    #     non_corner_bottom_indices = np.setdiff1d(bottom_indices, corner_indices)
    #     non_corner_top_indices = np.setdiff1d(top_indices, corner_indices)

    #     pairs_same_x = [(i, j) for i in non_corner_bottom_indices for j in non_corner_top_indices if np.abs(self.node_coords[i, 0] - self.node_coords[j, 0]) <= tolerance]
    #     pairs_same_y = [(i, j) for i in non_corner_left_indices for j in non_corner_right_indices if np.abs(self.node_coords[i, 1] - self.node_coords[j, 1]) <= tolerance]

    #     return corner_indices, np.array(pairs_same_x), np.array(pairs_same_y)
    def get_corner_and_boundary_pairs(self, boundary_labels: str|list[str] = '$\\partial\\Omega$', tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the indices of corner nodes, pairs of boundary nodes with same x and y values, and internal nodes.

        Parameters
        ----------
        boundary_labels : str or list of str, default '$\\partial\\Omega$'
            List of labels indicating the boundaries.
        tolerance : float, default 1e-10
            Tolerance for the equality test.

        Returns
        -------
        corner_indices : np.ndarray
            Indices of the corner nodes.
        pairs_same_x : np.ndarray
            Pairs of indices of boundary nodes with the same x value.
        pairs_same_y : np.ndarray
            Pairs of indices of boundary nodes with the same y value.
        inner_indices : np.ndarray
            Indices of the internal nodes.
        """
        if isinstance(boundary_labels, str):
            boundary_labels = [boundary_labels]

        # Collect boundary indices
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
                mode="expand", borderaxespad=0, ncol=2)
            ax.set(xlabel = "$x$", ylabel = "$y$", aspect = "equal")
            if save_name:
                fig.savefig(f"{save_name}.pdf")
        plt.show()
        
if __name__ == "__main__":
    mesh = CustomTwoDimensionMesh(filename='MS02_Periodic_Poisson_Equation/poisson_equation/geometries/two_domains_rectangle.msh', reordering = True)
    
    mesh.display()