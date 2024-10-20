import meshio
import numpy as np
import json
import jsbeautifier
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from typing import Optional, Dict, Any

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
    def __init__(self, filename: str, reordering: bool = True):
        try:
            mesh = meshio.read(filename)
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
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} does not exist.")
        except Exception as e:
            raise RuntimeError(f"Error while loading file {filename}: {e}")

        self._validate_mesh()
        self._initialize_attributes()
        self.assign_references_to_nodes()
        if reordering:
            self.reorder_elements()

    def _validate_mesh(self):
        elements_max_dimension = np.max([data[1] for data in self.field_data.values()])
        if elements_max_dimension != 2:
            raise ValueError(f"Dimension of the mesh should be 2, but got {elements_max_dimension}.")

    def _initialize_attributes(self):
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

    def reorder_elements(self) -> None:
        """ 
        @brief Reorder the mesh elements (nodes, edges, and triangles) so that the nodes are ordered by their reference.
        
        @detail Steps:
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

        # Optional: Logging instead of printing
        # logging.info(f"Inverse indices map: {inverse_indices}")

        # Step 3: Update the indices in self.tri_nodes and self.edge_nodes
        self.tri_nodes = inverse_indices[self.tri_nodes]
        self.edge_nodes = inverse_indices[self.edge_nodes]
        # print("Reordering completed successfully")
        # logging.info("Reordering completed successfully")
        

    def assign_references_to_nodes(self) -> None:
        """
        @brief Assign a physical reference to nodes.
        """
        for nodes, tag in zip(self.tri_nodes, self.tri_refs):
            self.node_refs[nodes] = tag
        for nodes, tag in zip(self.edge_nodes, self.edge_refs):
            self.node_refs[nodes] = tag

    def _create_colors(self, color_map: str = 'plasma') -> np.ndarray:
        """
        @brief Create a color list from the physical references of the mesh.
        @param color_map: matplotlib color map to use.
        """
        if len(self.refs) == 0:
            raise ValueError("No references found to generate colors.")
        norm = plt.Normalize(self.refs.min(), self.refs.max())
        colors = plt.get_cmap(color_map)(norm(self.refs))
        return colors

    def write_info_in_json(self) -> None:
        """
        @brief Save the mesh information in a json file.
        """
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
        """
        @brief Plot mesh elements (edges or triangles) on a given matplotlib Axes.
        @param ax: Matplotlib Axes to plot on.
        @param element_type: The type of element ('edges' or 'triangles') to plot.
        @param element_nodes: Node indices of the elements (edges or triangles).
        @param colors: Color for each element.
        @param alpha: Transparency level for the elements (used for triangles).
        @param lw: Line width for the edges or triangle borders.
        """
        vertices = np.stack((self.node_coords[element_nodes, 0], self.node_coords[element_nodes, 1]), axis=-1)
        
        match element_type:
            case 'edges':
                collection = LineCollection(vertices, lw=2)
                collection.set_colors(colors)
                
            case 'triangles':
                collection = PolyCollection(vertices, edgecolor='k', alpha=alpha, lw=lw)
                collection.set_facecolor(colors)
            case _ :
                raise ValueError("Element should be edge_nodes or tri_nodes.")
            
        ax.add_collection(collection)

    def display_nodes(self, ax: Optional[plt.Axes] = None, color_map: str = 'plasma', show: bool = True):
        
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        scatter = ax.scatter(*self.node_coords.T, c=colors[self.node_refs - 1], s=8)
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

    def display_triangles(self, ax: Optional[plt.Axes] = None, color_map: str = 'plasma', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        tri_colors = colors[self.tri_refs - 1]
        self._plot_elements(ax = ax, element_type = 'triangles', element_nodes = self.tri_nodes, colors=tri_colors, alpha=0.5, lw=0.5)
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

    def display_edges(self, ax: Optional[plt.Axes] = None, color_map: str = 'plasma', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        edge_colors = colors[self.edge_refs - 1]
        self._plot_elements(ax=ax, element_type = 'edges', element_nodes=self.edge_nodes, colors=edge_colors, alpha=1, lw=2)
        # ax.add_collection(collection)

        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

    def _add_legend(self, ax: plt.Axes, colors: np.ndarray):
        custom_lines = [Line2D([0], [0], color=colors[i - 1], lw=4) for i in self.labels.values()]

        # Crée une figure pour obtenir la position de la légende
        fig = ax.figure

        # Définit la position de la légende en dehors de l'axe
        legend = ax.legend(custom_lines, list(self.labels.keys()), loc='upper center', ncol=len(self.labels.keys()),
                        bbox_to_anchor=(0.5, 0.95), # Positionne la légende en haut au centre de la figure
                        bbox_transform=fig.transFigure)  # Utilise la transformation de la figure

        # Ajuste la taille de la figure pour s'assurer que la légende est visible
        # fig.subplots_adjust(top=0.85)
        ax.autoscale_view()

    def display(self, ax: Optional[plt.Axes] = None, color_map: str = 'viridis', show: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self._create_colors(color_map=color_map)
        
        self.display_triangles(ax=ax, color_map=color_map, show=False)
        self.display_edges(ax=ax, color_map=color_map, show=False)
        if show:
            self._add_legend(ax, colors)
            ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
            plt.show()

if __name__ == "__main__":
    # mesh = CustomTwoDimensionMesh(filename='poisson_equation/geometries/two_domains_rectangle.msh', reordering = True)
    mesh = CustomTwoDimensionMesh(filename='MS02_Periodic_Poisson_Equation/poisson_equation/geometries/rectangle.msh', reordering = True)
    # mesh = CustomTwoDimensionMesh(filename='geometries/asymmetrical_pipe.msh', reordering = True)
    # mesh = CustomTwoDimensionMesh(filename='geometries/square.msh', reordering = True)
    
    mesh.display()
    # mesh.reorder_elements()
    # print(mesh.labels)
    # print(np.where(mesh.node_refs == 1)[0])
    # print(np.where(mesh.node_refs == 2)[0])
    # print(np.where(mesh.node_refs == 3)[0])
    # mesh.display()
    
    # def sigma_by_node(mesh: CustomTwoDimensionMesh, node_index: int) -> int:
    #     """
    #     @brief Return the sigma value for a given node index.
        
    #     @param node_index: Index of the node to evaluate sigma at.
        
    #     @return: 1 if the node belongs to subdomain 1, 2 if it belongs to subdomain 2.
    #     """
    #     if mesh.node_refs[node_index] == 1:
    #         return 1
    #     elif mesh.node_refs[node_index] == 2:
    #         return 2
    #     else:
    #         raise ValueError(f"Node {node_index} does not belong to a recognized subdomain.")
        
    # def sigma_by_trianle(mesh: CustomTwoDimensionMesh, triangle_index: int) -> int:
    #     """
    #     @brief Return the sigma value for a given node index.
        
    #     @param node_index: Index of the node to evaluate sigma at.
        
    #     @return: 1 if the node belongs to subdomain 1, 2 if it belongs to subdomain 2.
    #     """
    #     if mesh.tri_refs[triangle_index] == 1:
    #         return 1
    #     elif mesh.tri_refs[triangle_index] == 2:
    #         return 2
    #     else:
    #         raise ValueError(f"Node {triangle_index} does not belong to a recognized subdomain.")