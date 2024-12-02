



from hfem.poisson_problems.solvers.homogenization.cell import CellProblem, CellProblemConfig
from hfem.analysis.convergence import (
    compute_and_save_cell_convergence_rates,
    plot_cell_convergence_from_csv
    )

import numpy as np

def diffusion_tensor(x, y):
    return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)
    
exact_tensor = np.diagflat([4*np.sqrt(3), 2*np.sqrt(15)], 0)

if __name__ == '__main__':
    # Test de convergence
    h_values = np.array([0.075, 0.05, 0.025, 0.02, 0.015, 0.01, 0.0095, 0.009, 0.0085, 0.008, 0.0075, 0.005])
    # h_values = np.array([0.075, 0.05, 0.025, 0.02])
    eta_values = np.array([5e0, 1e0, 5e-2, 5e-4, 5e-6])
    # compute_and_save_cell_convergence_rates(
    #     diffusion_tensor=diffusion_tensor,
    #     exact_homogenized_tensor=exact_tensor,
    #     h_values=h_values,
    #     eta_values=eta_values
    # )
    
    plot_cell_convergence_from_csv(
        csv_path="convergence_results/cell_convergence_results.csv",
        rate = 1,
        save_name="convergence_results/cell_convergence_results"
    )
    
    
    # plot_convergence_from_csv(
    #     csv_path="convergence_results/dirichlet_convergence_results.csv",
    #     # save_name="convergence_results/dirichlet_convergence_results"
    #     )



# def measure_and_save_case_iv(eta_values, mesh_sizes):
#     def diffusion_tensor(x, y):
#         return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)
    
#     exact_tensor = np.diagflat([4*np.sqrt(3), 2*np.sqrt(15)], 0)
    
#     pb_config = PenalizedCellProblemConfig(
#         diffusion_tensor=diffusion_tensor,
#         # exact_correctors=[exact_corrector1, exact_corrector2],
#         exact_homogenized_tensor=exact_tensor,
#         eta=1
#     )
    
#     data = measure_penalized_convergence(
#         base_config=pb_config,
#         eta_values=eta_values,
#         mesh_sizes=mesh_sizes,
#         mesh_generator=rectangular_mesh,
#         mesh_config={'L_x': 1.0, 'L_y': 1.0}
#     )
    
#     save_data(data, save_dir="results/convergences", filename="cell_problem_case_iv.csv")