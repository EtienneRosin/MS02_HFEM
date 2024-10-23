import numpy as np
import csv
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.poisson_problems import NeumannProblem

h_start = 0.1
# N = 2
N = 10
# lst_N = np.array([2**i for i in range(N)])
lst_N = np.arange(start=1, stop=N, step=1)
# lst_h = h_start/lst_N 

lst_h = [0.25, 0.125, 0.075, 0.05, 0.025, 0.0125, 0.0075, 0.005]
mesh_fname: str = "mesh_manager/geometries/rectangle.msh"

# Problem parameters --------------------------------------------
def v(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 2

def diffusion_tensor(x, y):
    return v(x, y) * np.eye(2)

def u(x, y):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

def f(x, y):
    return (1 + 16 * (np.pi**2) * (v(x, y) - 1)) * u(x, y)

folder: str = "ppph/poisson_problems/neumann/validation/4_error_variable_A"
filename: str = "error_measurements.csv"

if __name__ == '__main__':
    with open(f'{folder}/{filename}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["h", "L2_error", "H1_error"])
        for i, h in enumerate(lst_h):
            # h = h_start/N
            print(f"tot: {len(lst_h)}, {i = }, {h = }")
            # Mesh ----------------------------------------------------------
            create_rectangle_mesh(h = h, L_x = 1, L_y = 1, save_name = mesh_fname)
            mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
            # Problem itself ------------------------------------------------
            neumann_pb = NeumannProblem(mesh=mesh, diffusion_tensor=diffusion_tensor, rhs=f, exact_solution=u)
            neumann_pb.solve()
            # neumann_pb.display()
            writer.writerow([h, neumann_pb.relative_L2_error, neumann_pb.relative_H1_error])