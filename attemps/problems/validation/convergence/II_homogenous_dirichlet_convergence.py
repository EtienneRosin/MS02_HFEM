from hfem.problems import HomogenousDirichletPoissonProblem, BasePoissonConfig

from mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh
from mesh_manager.geometries import rectangular_mesh
from hfem.viz import solution_config, error_config, ErrorType

from hfem.problems.validation import ConvergenceData, measure_convergence, save_convergence_data, read_from_file, plot_convergence

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Problem configuration -----------------------------------

    # Geometry ----------------------------------
# rectangular geometry
mesh_sizes = [0.25, 0.125, 0.075, 0.05, 0.025, 0.0125, 0.0075, 0.005]
mesh_config = {'L_x': 1.0, 'L_y': 1.0}


    # Parameters --------------------------------
def v(x,y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + 2

def diffusion_tensor(x, y):
            return np.eye(2)*v(x, y)
        
def exact_solution(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def right_hand_side(x, y):
    return (1 + (16*np.pi**2)*(v(x, y)- 1))*exact_solution(x, y)

pb_config = BasePoissonConfig(
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side,
        exact_solution=exact_solution
    )

if __name__ == '__main__':
    data = measure_convergence(
        problem=HomogenousDirichletPoissonProblem,
        problem_config=pb_config,
        mesh_generator=rectangular_mesh,
        mesh_sizes=mesh_sizes,
        mesh_config=mesh_config
    ) 
    save_dir = Path("results/convergences")
    
    save_convergence_data(data, save_dir)
    
    data = read_from_file(filepath=f"{save_dir}/convergence_dirichlet.csv")
    
    plot_convergence(data, save_name=f"{save_dir}/convergence_dirichlet")