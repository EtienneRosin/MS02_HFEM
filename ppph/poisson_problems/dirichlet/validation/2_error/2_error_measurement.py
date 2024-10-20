import numpy as np
import csv
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.poisson_problems import DirichletProblem

h_start = 0.1
N = 10
lst_N = np.arange(start=1, stop=N, step=1)

lst_h = [0.25, 0.125, 0.075, 0.05, 0.025, 0.0125, 0.0075, 0.005, 0.0025, 0.00125]
mesh_fname: str = "mesh_manager/geometries/rectangle.msh"

def diffusion_tensor(x, y):
        return np.eye(2)
    
def u(x, y):
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)

def f(x, y):
    return (1 + 5 * np.pi ** 2) * u(x, y)

folder: str = "ppph/poisson_problems/dirichlet/validation/2_error"
filename: str = "error_measurements.csv"

if __name__ == '__main__':
    with open(f'{folder}/{filename}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["h", "L2_error", "H1_error"])
        for i, h in enumerate(lst_h):
            # h = h_start/N
            print(f"tot: {len(lst_h)}, {i = }, {h = }")
            # Mesh ----------------------------------------------------------
            create_rectangle_mesh(h = h, L_x = 2, L_y = 1, save_name = mesh_fname)
            mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
            # Problem itself ------------------------------------------------
            dirichlet_problem = DirichletProblem(mesh=mesh, diffusion_tensor=diffusion_tensor, rhs=f, exact_solution=u)
            dirichlet_problem.solve()
            # neumann_pb.display()
            writer.writerow([h, dirichlet_problem.relative_L2_error, dirichlet_problem.relative_H1_error])