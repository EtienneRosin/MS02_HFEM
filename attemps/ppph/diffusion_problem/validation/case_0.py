from ppph.diffusion_problem import PoissonPeriodicProblem
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh

import numpy as np

# Mesh ----------------------------------------------------------
mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
h = 0.025
create_rectangle_mesh(h = h, L_x = 2, L_y = 2, save_name = mesh_fname)

mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
# Problem parameters --------------------------------------------
epsilon = 1

def A(x, y):
    return np.eye(2)

def diffusion_tensor(x, y):
    return A(x/epsilon, y/epsilon)

def u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x, y): 
    return 2*(np.pi**2)*u(x,y)

# Problem itself ------------------------------------------------
periodic_pb = PoissonPeriodicProblem(
    mesh = mesh, 
    diffusion_tensor = 
    diffusion_tensor, 
    rhs = f, 
    exact_solution = u)
periodic_pb.solve()
# periodic_pb.display()
periodic_pb.display_3d()
periodic_pb.display_error()


