from square_mesh import create_square_mesh
from poisson_equation.poisson_problem_2 import PoissonProblem
from poisson_equation.mesh import CustomTwoDimensionMesh
from poisson_equation.geometries.rectangle_mesh import create_rectangle_mesh

import numpy as np

# lst_h = [0.5, 0.25, 0.1, 0.75, 0.05, 0.025]
lst_h = [0.075, 0.05, 0.025, 0.0125]
# lst_h = [0.5, 0.25]

def diffusion_tensor(x, y):
        return np.eye(2)

def u(x,y):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

def f(x, y):
    return (1 + 5 * np.pi ** 2) * u(x, y)

def rho(x, y):
    return 1

for h in lst_h:
    
    create_rectangle_mesh(h = h, save_name="./study_cases/1_Error_analysis/rectangle.msh")
    mesh = CustomTwoDimensionMesh(filename="./study_cases/1_Error_analysis/rectangle.msh", reordering=True)
    
    pb = PoissonProblem(mesh=mesh, f=f, diffusion_tensor=diffusion_tensor, rho=rho, exact_solution=u)
    pb.display_solution()
    pb.display_error()
    
    