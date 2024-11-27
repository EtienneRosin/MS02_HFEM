from ppph.diffusion_problem import PoissonPeriodicProblem
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh

import numpy as np

# Mesh ----------------------------------------------------------
mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
h = 0.05
create_rectangle_mesh(h = h, L_x = 2, L_y = 2, save_name = mesh_fname)

mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
# Problem parameters --------------------------------------------
epsilon = 1
def A(x, y):
    return np.diagflat([1, 2], 0)

# print(f"{A(1, 1) = }")

def A_epsilon(x, y):
    return A(x/epsilon, y/epsilon)

def u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def f(x, y): 
    return (3*np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)

# Problem itself ------------------------------------------------
periodic_pb = PoissonPeriodicProblem(
    mesh = mesh, 
    diffusion_tensor = A_epsilon, 
    rhs = f, 
    exact_solution = u)
periodic_pb.solve()
# periodic_pb.display()
periodic_pb.display_3d()
periodic_pb.display_error()


