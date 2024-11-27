from ppph.diffusion_problem.exact_solution import ExactPeriodicDiffusionProblem
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh

import numpy as np

if __name__ == '__main__':
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.05
    # h = 0.025
    create_rectangle_mesh(h = h, L_x = 2, L_y = 2, save_name = mesh_fname)
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    
    # Problem parameters --------------------------------------------
    epsilon = 1
    
    def A(x, y):
        return np.diagflat([1,2], 0)
    
    def diffusion_tensor(x, y):
        return A(x/epsilon, y/epsilon)

    def u(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def f(x, y): 
        return 3*(np.pi**2)*u(x,y)
    
    # Problem itself ------------------------------------------------
    diffusion_pb = ExactPeriodicDiffusionProblem(
        mesh = mesh, 
        diffusion_tensor = diffusion_tensor, 
        rhs = f, 
        exact_solution = u)
    diffusion_pb.solve()
    # diffusion_pb.display()
    diffusion_pb.display_3d()
    diffusion_pb.display_error()