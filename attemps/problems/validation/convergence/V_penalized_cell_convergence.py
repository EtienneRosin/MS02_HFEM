from hfem.problems import PenalizedCellProblemConfig, PenalizedCellProblems
# from hfem.problems.validation import measure_penalized_convergence, plot_penalized_convergence



from pathlib import Path
from hfem.mesh_manager.geometries import rectangular_mesh
import numpy as np

from hfem.problems.validation import measure_penalized_convergence, PenalizedCellConvergenceData, MultiEtaPenalizedCellConvergenceData, read_data, save_data, plot_corrector_convergence, plot_tensor_convergence


def measure_and_save_case_0(eta_values, mesh_sizes):
    def diffusion_tensor(x, y):
        return np.eye(2)
    
    def exact_corrector1(x, y):
        return np.zeros_like(x)
    
    def exact_corrector2(x, y):
        return np.zeros_like(x)
    
    exact_tensor = np.eye(2)
    
    pb_config = PenalizedCellProblemConfig(
        diffusion_tensor=diffusion_tensor,
        exact_correctors=[exact_corrector1, exact_corrector2],
        exact_homogenized_tensor=exact_tensor,
        eta=1
    )
    
    data = measure_penalized_convergence(
        base_config=pb_config,
        eta_values=eta_values,
        mesh_sizes=mesh_sizes,
        mesh_generator=rectangular_mesh,
        mesh_config={'L_x': 1.0, 'L_y': 1.0}
    )
    
    save_data(data, save_dir="results/convergences", filename="cell_problem_case_0.csv")
    

def measure_and_save_case_ii(eta_values, mesh_sizes):
    def diffusion_tensor(x, y):
        return np.diagflat([2 + np.sin(2 * np.pi * x), 4], 0)
    
    exact_tensor = np.diagflat([np.sqrt(3), 4], 0)
    
    pb_config = PenalizedCellProblemConfig(
        diffusion_tensor=diffusion_tensor,
        # exact_correctors=[exact_corrector1, exact_corrector2],
        exact_homogenized_tensor=exact_tensor,
        eta=1
    )
    
    data = measure_penalized_convergence(
        base_config=pb_config,
        eta_values=eta_values,
        mesh_sizes=mesh_sizes,
        mesh_generator=rectangular_mesh,
        mesh_config={'L_x': 1.0, 'L_y': 1.0}
    )
    
    save_data(data, save_dir="results/convergences", filename="cell_problem_case_ii.csv")
    
def measure_and_save_case_iii(eta_values, mesh_sizes):
    def diffusion_tensor(x, y):
        return np.diagflat([2 + np.sin(2 * np.pi * x), 4 + np.sin(2 * np.pi * x)], 0)
    
    exact_tensor = np.diagflat([np.sqrt(3), 4], 0)
    
    pb_config = PenalizedCellProblemConfig(
        diffusion_tensor=diffusion_tensor,
        # exact_correctors=[exact_corrector1, exact_corrector2],
        exact_homogenized_tensor=exact_tensor,
        eta=1
    )
    
    data = measure_penalized_convergence(
        base_config=pb_config,
        eta_values=eta_values,
        mesh_sizes=mesh_sizes,
        mesh_generator=rectangular_mesh,
        mesh_config={'L_x': 1.0, 'L_y': 1.0}
    )
    
    save_data(data, save_dir="results/convergences", filename="cell_problem_case_iii.csv")

def measure_and_save_case_iv(eta_values, mesh_sizes):
    def diffusion_tensor(x, y):
        return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)
    
    exact_tensor = np.diagflat([4*np.sqrt(3), 2*np.sqrt(15)], 0)
    
    pb_config = PenalizedCellProblemConfig(
        diffusion_tensor=diffusion_tensor,
        # exact_correctors=[exact_corrector1, exact_corrector2],
        exact_homogenized_tensor=exact_tensor,
        eta=1
    )
    
    data = measure_penalized_convergence(
        base_config=pb_config,
        eta_values=eta_values,
        mesh_sizes=mesh_sizes,
        mesh_generator=rectangular_mesh,
        mesh_config={'L_x': 1.0, 'L_y': 1.0}
    )
    
    save_data(data, save_dir="results/convergences", filename="cell_problem_case_iv.csv")
 
    
# def example_convergence_analysis():
#     # Define problem parameters and exact solutions
#     def diffusion_tensor(x, y):
#         return np.array([[2 + np.cos(2*np.pi*x), 0], 
#                         [0, 2 + np.cos(2*np.pi*y)]])
    
#     def exact_corrector1(x, y):
#         return -1/(2*np.pi) * np.sin(2*np.pi*x)
    
#     def exact_corrector2(x, y):
#         return -1/(2*np.pi) * np.sin(2*np.pi*y)
    
#     exact_tensor = np.array([[1.5, 0], [0, 1.5]])
    
#     # Create problem configuration
#     config = PenalizedCellProblemConfig(
#         diffusion_tensor=diffusion_tensor,
#         exact_correctors=[exact_corrector1, exact_corrector2],
#         exact_homogenized_tensor=exact_tensor,
#         eta=1e-6
#     )
    
#     # Setup convergence test parameters
#     mesh_sizes = [0.2, 0.1, 0.05, 0.025]
#     mesh_config = {'L_x': 1.0, 'L_y': 1.0}
    
#     # Run convergence analysis
#     data = measure_penalized_convergence(
#         problem_config=config,
#         mesh_generator=rectangular_mesh,
#         mesh_sizes=mesh_sizes,
#         mesh_config=mesh_config
#     )
    
    
    
#     # Plot results
#     plot_penalized_convergence(
#         data=data,
#         # save_name='penalized_convergence_test',
#         show=True
#     )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # example_convergence_analysis()
    # eta_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    eta_values = [5e0, 1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    # mesh_sizes = [0.01, 0.0075, 0.005, 0.0025]
    # mesh_sizes = [0.1, 0.075, 0.05, 0.025, 0.0125, 0.0075, 0.005]
    # mesh_sizes = [0.1, 0.075, 0.05, 0.025, 0.0125, 0.0075, 0.005]
    mesh_sizes = [0.075, 0.05, 0.025, 0.02, 0.015, 0.01, 0.0095, 0.009, 0.0085, 0.008, 0.0075]
    
    
    
    # mesh_sizes = [0.075, 0.05, 0.025]
    # mesh_sizes = [0.25, 0.125, 0.075, 0.05]
    
    # measure_and_save_case_0(eta_values=eta_values, mesh_sizes=mesh_sizes)
    # measure_and_save_case_ii(eta_values=eta_values, mesh_sizes=mesh_sizes)
    # measure_and_save_case_iii(eta_values=eta_values, mesh_sizes=mesh_sizes)
    # measure_and_save_case_iv(eta_values=eta_values, mesh_sizes=mesh_sizes)
    
    fname = "cell_problem_case_ii"
    
    data = read_data(filepath=f"results/convergences/{fname}.csv", data_type="multi-eta penalized cell")
    
    # plot_corrector_convergence(data)
    # plt.show()
    plot_tensor_convergence(data, rate=2, save_name=f"results/convergences/convergence_{fname}")
    plt.show()
    