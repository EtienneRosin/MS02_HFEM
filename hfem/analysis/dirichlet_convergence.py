
from hfem.poisson_problems.solvers.standard.dirichlet import DirichletHomogeneousProblem
from hfem.poisson_problems.configs.standard.dirichlet import DirichletConfig
from hfem.analysis.convergence import compute_convergence_rates, plot_convergence_from_csv

import numpy as np

def v(x,y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + 2

def diffusion_tensor(x, y):
            return np.eye(2)*v(x, y)
        
def exact_solution(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def right_hand_side(x, y):
    return (1 + (16*np.pi**2)*(v(x, y)- 1))*exact_solution(x, y)

if __name__ == '__main__':
    # Test de convergence
    h_values = np.array([0.075, 0.05, 0.025, 0.02, 0.015, 0.01, 0.0095, 0.009, 0.0085, 0.008, 0.0075, 0.005])
    # compute_convergence_rates(
    #     DirichletHomogeneousProblem,
    #     DirichletConfig,
    #     exact_solution,
    #     diffusion_tensor,
    #     right_hand_side,
    #     h_values,
    #     domain_size = (1, 1)
    # )
    
    plot_convergence_from_csv(
        csv_path="convergence_results/dirichlet_convergence_results.csv",
        save_name="convergence_results/dirichlet_convergence_results")