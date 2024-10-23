from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.poisson_problems import NeumannProblem
import numpy as np

if __name__ == "__main__":
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.0125
    create_rectangle_mesh(h=h, L_x=1, L_y=1, save_name=mesh_fname)
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering=True)
    
    # Problem parameters --------------------------------------------
    def v(x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 2

    def diffusion_tensor(x, y):
        return v(x, y) * np.eye(2)

    def u(x, y):
        return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

    def f(x, y):
        return (1 + 16 * (np.pi**2) * (v(x, y) - 1)) * u(x, y)
    
    # Problem itself ------------------------------------------------
    neumann_pb = NeumannProblem(
        mesh = mesh,
        diffusion_tensor = diffusion_tensor,
        rhs = f,
        exact_solution = u
    )
    
    neumann_pb.solve()
    neumann_pb.display()
