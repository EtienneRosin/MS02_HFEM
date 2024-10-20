import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.utils import TwoDimensionFunction
from ppph.utils.graphics import display_field_on_mesh

def u_expr(x, y):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

u = TwoDimensionFunction(u_expr)

mesh_fname: str = "mesh_manager/geometries/rectangle.msh"

if __name__ == "__main__":
    
    h = 0.25
    create_rectangle_mesh(h=h, L_x=3, L_y=2, save_name=mesh_fname)
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering=True)
    
    # Extract border nodes
    
    border_ref = mesh.labels['$\\partial\\Omega$']
    border_indices = np.where(mesh.node_refs == border_ref)[0]

    corner_indices, pairs_same_x, pairs_same_y = mesh.get_corner_and_boundary_pairs()

    mesh.display_corner_and_boundary_pairs(save_name="Figures/coordinate_pairs")

    # # Visualize the pairs
    
