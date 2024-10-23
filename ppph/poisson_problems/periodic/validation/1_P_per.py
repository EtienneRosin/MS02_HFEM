import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.utils import TwoDimensionFunction
from ppph.utils.graphics import display_field_on_mesh
from pprint import pprint

def u_expr(x, y):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

def u_expr(x, y):
    return 1

u = TwoDimensionFunction(u_expr)

mesh_fname: str = "mesh_manager/geometries/rectangle.msh"



if __name__ == "__main__":
    
    h = 0.5
    create_rectangle_mesh(h=h, L_x=3, L_y=2, save_name=mesh_fname)
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering=True)
    
    # Extract border nodes
    
    border_ref = mesh.labels['$\\partial\\Omega$']
    border_indices = np.where(mesh.node_refs == border_ref)[0]

    corner_indices, pairs_same_x, pairs_same_y, inner_indices = mesh.get_corner_and_boundary_pairs()
    # mesh.display_corner_and_boundary_pairs()
    
    print(corner_indices)
    print(pairs_same_x)
    print(pairs_same_y)
    
    N_inner = len(inner_indices)
    N_corner = 1    # we keep only one corner
    N_pairs_x = len(pairs_same_x)
    N_pairs_y = len(pairs_same_y)
    # print(pairs_same_x[1])
    N_per = N_inner + N_corner + N_pairs_x + N_pairs_y
    print(f"{mesh.num_nodes = }, {N_per = }, {len(inner_indices) = }, {len(pairs_same_x) = }, {len(pairs_same_y) = }")
    
    # 'Projector' of the space V_h of approximation of H^1(\Omega) into V_h^# the approximation of H_#^1(\Omega)
    P = sp.sparse.lil_matrix((N_per, mesh.num_nodes))
    
    for n, i in enumerate(inner_indices):
        P[n,i] = 1
    
    for i in corner_indices:
        P[N_inner,i] = 1
    
    for n, (i,j) in enumerate(pairs_same_x):
        P[n + N_inner + N_corner, i] = P[n + N_inner + N_corner, j] = 1
        
    for n, (i,j) in enumerate(pairs_same_y):
        # print(f"{n = }, {i = }, {j = }")
        P[n + N_inner + N_corner + N_pairs_x, i] = P[n + N_inner + N_corner+ N_pairs_x, j] = 1
    # for n in range(N_pairs_x):
    #     # print(n)
    #     for pair in :
    #         print(f"{pair = }")
            # P[n + N_inner + N_corner, i] = P[n + N_inner + N_corner, j] = 1
    # for (i,j) in pairs_same_x:
    #     P[i,i] = 1
    #     P[i,j] = 1
    
    
    
    # for i in range(N_per):
    #     if i < len(inner_indices):
    #         P[i, i] = 1
        
    #     if i <= len(inner_indices):
    #         for j in corner_indices:
    #             P[corner_indices[0], i] = 1
    
    
    # for (i,j) in pairs_same_y:
    #     P[i,i] = 1
    #     P[i,j] = 1
        
        # P[j,i] = 1
        # P[j,j] = 1
    # for (i,j) in pairs_same_x:
    #     P[i,j] = 1
    #     P[i,i] = 1
    #     P[j,i] = 1
    #     P[j,j] = 1
        
    # for (i,j) in pairs_same_y:
    #     P[i,j] = 1
    #     P[i,i] = 1
    #     P[j,i] = 1
    #     P[j,j] = 1
    
    pprint(P.toarray())
    # print(P)
    
    # # field = u(mesh.node_coords[corner_indices])
    field = P @ u(mesh.node_coords)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.scatter(*mesh.node_coords.T, P.T @ field)
    plt.show()
    
