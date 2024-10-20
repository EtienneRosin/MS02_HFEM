
from mesh_manager import CustomTwoDimensionMesh

from ppph.utils import TwoDimensionFunction

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import cmasher as cmr

def display_field_on_mesh(mesh: CustomTwoDimensionMesh, field: np.ndarray, label: str = None, save_name: str = None, cmap: str = 'cmr.lavender'):
    with plt.style.context('science' if save_name else 'default'):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
        
        contour = ax.tricontourf(*mesh.node_coords.T, mesh.tri_nodes, field, cmap = cmap)
        fig.colorbar(contour, ax = ax, shrink=0.5, aspect=20, label = fr"{label}")
        if save_name:
            fig.savefig(f"{save_name}.pdf")
        plt.show()


def u_expr(x, y):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

u = TwoDimensionFunction(u_expr)
if __name__ == '__main__':
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    mesh = CustomTwoDimensionMesh(filename=mesh_fname, reordering=True)
    
    field = u(mesh.node_coords)
    
    display_field_on_mesh(mesh=mesh, field=field, label='$u$')