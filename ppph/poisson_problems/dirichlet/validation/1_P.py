import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.utils import TwoDimensionFunction
from ppph.utils.graphics import display_field_on_mesh

def u_expr(x, y):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

u = TwoDimensionFunction(u_expr)

def _construct_P(mesh: CustomTwoDimensionMesh) -> sp.sparse.csr_matrix:
    r"""
    Create the P matrix that returns the interior nodes of the mesh.

    Parameters
    ----------
    mesh : CustomTwoDimensionMesh
        Considered mesh.

    Returns
    -------
    P : sp.sparse.csr_matrix
        The P matrix.
    """
    on_border_ref = mesh.labels['$\\partial\\Omega$']
    interior_indices = np.where(mesh.node_refs != on_border_ref)[0]
    N_0 = len(interior_indices)
    N = mesh.num_nodes

    P = sp.sparse.lil_matrix((N_0, N), dtype=float)
    for i, j in enumerate(interior_indices):
        P[i, j] = 1

    return P.tocsr()

if __name__ == "__main__":
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.05
    # create_rectangle_mesh(h=h, L_x=2, L_y=1, save_name=mesh_fname)
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering=True)

    # Construct the P matrix
    P = _construct_P(mesh)
    # print(P)

    # # Apply the P matrix to the function values
    field = P @ u(mesh.node_coords)

    # Display the field
    # display_field_on_mesh(mesh=mesh, field=field, label='$u$')
    on_border_ref = mesh.labels['$\\partial\\Omega$']
    # Alternatively, using matplotlib directly
    interior_nodes = mesh.node_coords[np.where(mesh.node_refs != on_border_ref)]
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    
    ax.scatter(*mesh.node_coords.T, u(mesh.node_coords), s = 1, c = "blue", alpha=1)
    ax.scatter(*interior_nodes.T, field, s = 6, c = "red", alpha = 0.75)
    ax.scatter(*mesh.node_coords.T, P.T @ field, s = 6, c = "yellow", alpha = 0.75)
    
    
    
    plt.show()
    # interior_nodes = mesh.node_coords[np.where(mesh.node_refs != on_border_ref)]
    # interior_triangles = mesh.tri_nodes[np.where(mesh.tri_refs != on_border_ref)]
    # contour = ax.tricontourf(interior_nodes[:, 0], interior_nodes[:, 1], interior_triangles, field)
    # fig.colorbar(contour, ax=ax, shrink=0.5, aspect=20)
    # plt.show()
