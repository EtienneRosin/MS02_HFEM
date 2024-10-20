import meshio
import numpy as np
import json
import jsbeautifier

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

def custom_serializer(obj):
    # Si l'objet est un ndarray de NumPy, on le convertit en liste
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Si l'objet est un CellBlock (ou un objet non sérialisable), on peut extraire les informations importantes
    elif isinstance(obj, meshio.CellBlock):
        return {
            "type": obj.type,
            "data": obj.data.tolist(),
            "num_cells": len(obj.data),
            "dim": obj.dim,
            "tags": obj.tags
        }
    
    # Pour tout autre type d'objet non sérialisable directement en JSON
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
# mesh.field_data

class CustomTwoDimensionMesh(meshio.Mesh):
    def __init__(self, filename: str):
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
            raise FileNotFoundError(f"File {filename} doesn't exist.")
        except Exception as e:
            raise Exception(f"Error while loading file {filename} : {str(e)}")

        # for field, data in self.field_data.items():
        #     print(f"{}")
        elements_max_dimension = np.max([data[1] for data in self.field_data.values()])
        
        # print(f"{self.refs = }")
        if elements_max_dimension != 2:
            raise ValueError("Dimension of the mesh should be 2, here it is {elements_max_dimension}.")
        
        self.node_coords = self.points[:, :-1]   # Coordinates of the nodes (num_nodes x 2)
        self.num_nodes = len(self.node_coords)
        self.node_refs = np.zeros(self.num_nodes)     # Node references (num_nodes x 1)
        
        self.tri_nodes = self.cells_dict['triangle'] # List of triangles (num_triangles x 6)
        self.num_triangles = len(self.tri_nodes)
        self.tri_refs = self.cell_data_dict['gmsh:physical']['triangle'].astype(int)    # Triangle references (num_triangles x 1)
        
        self.edge_nodes = self.cells_dict['line'] # List of edge nodes (num_edges x 3)
        self.num_edges = len(self.edge_nodes)
        self.edge_refs = mesh.cell_data_dict['gmsh:physical']['line'].astype(int)      # Edge references (num_edges x 1)
        
        self.refs = np.sort(np.array([data[0] for data in self.field_data.values()]))
        self.labels = {f"${field}$": data[0] for field, data in self.field_data.items()}
        # print(f"{self.labels = }")
        self.assign_references_to_nodes()
    
    def assign_references_to_nodes(self) -> None:
        
        # on commence par etiqueter les triangles comme ces derniers seront nécessairement à l'intérieur du domaine.
        for nodes, tag in zip(self.tri_nodes, self.tri_refs):
            # print(f"{nodes = }, {tag = }")
            self.node_refs[nodes] = tag
        
        # on étiquette ensuite les arrêtes comme ces dernières peuvent définir le bord du domaine (ainsi la priorité de référence d'un noeud sera donnée à l'arrête par rapport au triangle).
        for nodes, tag in zip(self.edge_nodes, self.edge_refs):
            # print(f"{nodes = }, {tag = }")
            self.node_refs[nodes] = tag
            
        self.node_refs = self.node_refs.astype(int)
        pass
        
    def _create_colors(self, color_map: str = 'plasma'):
        norm = plt.Normalize(self.refs.min(), self.refs.max())
        colors = plt.get_cmap(color_map)(norm(self.refs))
        # print(colors)
        return(colors)
        
        
    def write_info_in_json(self):
        with open("sample.json", "w") as outfile: 

            options = jsbeautifier.default_options()
            options.indent_size = 4

            # Format the JSON output
            formatted_json = jsbeautifier.beautify(json.dumps(self.__dict__, default = custom_serializer, sort_keys = False), options)

            with open("mesh_info.json", "w") as json_file:
                json_file.write(formatted_json)
            
    
    def display_nodes(self, ax: plt.axes = None, color_map: str = 'plasma', show: bool = True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            
        colors = self._create_colors(color_map = color_map)
        ax.scatter(*self.node_coords.T, c = colors[self.node_refs - 1], s = 8)
        
        if show:
            custom_lines = [Line2D([0], [0], color= colors[i - 1], lw = 4) for i in self.labels.values()]
            ax_pos = ax.get_position()
            fig.legend(custom_lines, list(self.labels.keys()), loc='upper center', ncols=len(self.labels.keys()),
                    bbox_to_anchor=(0.5, ax_pos.y1 - 0.225))  # Position de la légende au-dessus de l'axe
        
            ax.set(aspect="equal", xlabel = "$x$", ylabel = "$y$")
            plt.show()
    
    def display_triangles(self, ax: plt.axes = None, color_map: str = 'plasma', show: bool = True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            
        colors = self._create_colors(color_map = color_map)
        tri_colors = colors[self.tri_refs - 1]  # Ensure tri_refs is int and 1D
        
        vertices = np.stack((self.node_coords[self.tri_nodes, 0], self.node_coords[self.tri_nodes, 1]), axis=-1)
        collection = PolyCollection(vertices, edgecolor= "k", alpha = 0.5, lw = 0.5)
        collection.set_facecolor(tri_colors)
        
        
        ax.add_collection(collection)
        ax.autoscale_view()
        
        
        if show:
            custom_lines = [Line2D([0], [0], color = colors[i - 1], lw = 4) for i in self.labels.values()]
            ax_pos = ax.get_position()
            fig.legend(custom_lines, list(self.labels.keys()), loc='upper center', ncols=len(self.labels.keys()),
                    bbox_to_anchor=(0.5, ax_pos.y1 - 0.225))  # Position de la légende au-dessus de l'axe
        
            ax.set(aspect = "equal", xlabel = "$x$", ylabel = "$y$")
            plt.show()
    
    def display_edges(self, ax: plt.axes = None, color_map: str = 'plasma', show: bool = True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            
        colors = self._create_colors(color_map = color_map)
        edge_colors = colors[self.edge_refs - 1]
        
        vertices = np.stack((self.node_coords[self.edge_nodes, 0], self.node_coords[self.edge_nodes, 1]), axis=-1)
        collection = LineCollection(vertices, lw = 2)
        collection.set_colors(edge_colors)
        
        ax.add_collection(collection)
        ax.autoscale_view()
        
        if show:
            custom_lines = [Line2D([0], [0], color = colors[i - 1], lw = 4) for i in self.labels.values()]
            ax_pos = ax.get_position()
            fig.legend(custom_lines, list(self.labels.keys()), loc='upper center', ncols=len(self.labels.keys()),
                    bbox_to_anchor=(0.5, ax_pos.y1 - 0.225))  # Position de la légende au-dessus de l'axe
        
            ax.set(aspect = "equal", xlabel = "$x$", ylabel = "$y$")
            plt.show()
    
    
    def display(self, ax: plt.axes = None, color_map: str = 'viridis', show: bool = True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        colors = self._create_colors(color_map = color_map)
        
        # self.display_nodes(ax=ax, color_map=color_map, show=False)
        self.display_edges(ax=ax, color_map=color_map, show=False)
        self.display_triangles(ax=ax, color_map=color_map, show=False)
        
        if show:
            custom_lines = [Line2D([0], [0], color = colors[i - 1], lw = 4) for i in self.labels.values()]
            ax_pos = ax.get_position()
            fig.legend(custom_lines, list(self.labels.keys()), loc='upper center', ncols=len(self.labels.keys()),
                    bbox_to_anchor=(0.5, ax_pos.y1 - 0.225))  # Position de la légende au-dessus de l'axe
        
            ax.set(aspect = "equal", xlabel = "$x$", ylabel = "$y$")
            plt.show()
        # print(np.where(dim_tags[:, 0] == 0))
if __name__ == "__main__":
    # Chargement du maillage et affichage
    mesh = CustomTwoDimensionMesh(filename='geometries/mesh.msh')
    # print(mesh.field_data)
    # mesh.display_nodes()
    # print(mesh.tri_refs )
    # print(mesh.node_refs.shape, mesh.node_refs.dtype)
    # print(mesh.tri_refs.shape, mesh.tri_refs.dtype)
    
    
    
    mesh.display()
    
    
    points = mesh.node_coords[mesh.edge_nodes]
    print(points)
    
    # print(vertices)
    # self.node_coords[self.tri_nodes][:, 0]
    
    
    # mesh.display()
    # print(mesh.tri_nodes)
    
    # mesh.test_assign_ref()
    
    # print(mesh.cell_sets_dict)
    
    
    
    # print(mesh.cell_data_dict['gmsh:physical'])
    # print(mesh.cell_data_dict)
    # print(mesh.cells_dict)
    # print(mesh.cells)
    # print(mesh.points)
    
    # print(mesh.cell_data_dict['gmsh:physical']['triangle'])
    # print(mesh.tri_nodes)
    
    
    # for tags, cell in zip(mesh.cell_data['gmsh:physical'], mesh.cells):
    #     print(tags, cell)
    
    
    
    # print(mesh.__dict__)
    
    # mesh.write_info_in_json()
    
    
    
    # print(dir(mesh))
    # print(mesh.__dict__)
    # for att in dir(mesh):
    #     print (att, getattr(mesh,att))
        
    # cell_data_dict['gmsh:physical'] / cell_data_dict['gmsh:geometrical']
    # print(mesh.gmsh_periodic)
    
    
    
    
    
    # print(mesh.cells_dict['triangle'].shape)
    # print(mesh.cell_data_dict['gmsh:physical']['triangle'].shape)