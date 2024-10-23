
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
# from ppph.utils.graphics import cu
from ppph.utils import TwoDimensionFunction

style_path = "ppph/utils/graphics/custom_science.mplstyle"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots
import cmasher as cmr

def display_field_on_mesh(mesh: CustomTwoDimensionMesh, field: np.ndarray, label: str = None, save_name: str = None, cmap: str = 'cmr.lavender'):
    with plt.style.context('science' if save_name else 'default'):
        fig = plt.figure()
        ax = fig.add_subplot()
        
        
        contour = ax.tricontourf(*mesh.node_coords.T, mesh.tri_nodes, field, cmap = cmap)
        # fig.colorbar(contour, ax = ax, shrink=0.5, aspect=20, label = fr"{label}")
        # fig.colorbar(contour, ax=ax, fraction=0.15, pad=0.025, label = fr"{label}")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad="2%")
        plt.colorbar(contour, cax=cax, label = fr"{label}")
        
        ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
        ax.set_box_aspect(None)
        if save_name:
            fig.savefig(f"{save_name}.pdf")
        plt.show()

def display_3d(mesh: CustomTwoDimensionMesh, field: np.ndarray, label: str = None, save_name: str = None, cmap: str = 'cmr.lavender', view_init: tuple = (40, -30)):
    with plt.style.context(style_path if save_name else 'default'):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # Create the trisurf plot
        contour = ax.plot_trisurf(*mesh.node_coords.T, mesh.tri_nodes, field, cmap=cmap)

        # Set labels and view angle
        ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=fr"{label}", aspect="auto")
        ax.view_init(*view_init)

        # Ensure proper layout and save figure
        if save_name:
            # Save with bbox_inches='tight' to ensure all labels are within the saved image
            fig.savefig(f"{save_name}.pdf", bbox_inches='tight')

        # Show the plot
        plt.show()



def u_expr(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)

u = TwoDimensionFunction(u_expr)
if __name__ == '__main__':
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.05
    # create_rectangle_mesh(h = h, L_x = 2, L_y = 1, save_name = mesh_fname)
    mesh = CustomTwoDimensionMesh(filename=mesh_fname, reordering=True)
    
    field = u(mesh.node_coords)
    
    display_field_on_mesh(mesh=mesh, field=field, label='$u$',save_name="hih")
    # display_3d(mesh=mesh, field=field, label='$u$',save_name="udzubuyczbuy")