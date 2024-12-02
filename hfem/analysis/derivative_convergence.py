from hfem.analysis.convergence import (
    compute_and_save_derivatives_convergence_rates,
    plot_derivatives_convergence_from_csv)

import numpy as np

def exact_solution(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)
    
def right_hand_side(x, y):
    return -2*(np.pi**2)*(-np.sqrt(15) - 2*np.sqrt(3))*exact_solution(x,y)
    
effective_tensor = np.array([
    [4*np.sqrt(3), 0],
    [0, 2*np.sqrt(15)]
])

def exact_derivatives(x, y):
    # ∂x(sin(πx)sin(πy)) = πcos(πx)sin(πy)
    dx = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
    # ∂y(sin(πx)sin(πy)) = sin(πx)πcos(πy)
    dy = np.sin(np.pi*x) * np.pi * np.cos(np.pi*y)
    return dx, dy

if __name__ == '__main__':
    h_values = np.array([0.075, 0.05, 0.025, 0.02, 0.015, 0.01, 0.0095, 0.009, 0.0085, 0.008, 0.0075, 0.005])
    # h_values = np.array([0.075, 0.05, 0.025])
    
    # compute_and_save_derivatives_convergence_rates(
    #     effective_tensor=effective_tensor,
    #     exact_derivatives=exact_derivatives,
    #     right_hand_side=right_hand_side,
    #     domain_size=(2, 2),
    #     h_values=h_values
    #     )
    plot_derivatives_convergence_from_csv(
        csv_path="convergence_results/homogenized_convergence_results.csv",
        save_name="convergence_results/homogenized_convergence_results"
        )