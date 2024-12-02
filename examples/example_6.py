from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from hfem.core.io import FEMDataManager
from hfem.viz.conditional_style_context import conditional_style_context
from hfem.analysis.convergence import add_convergence_triangle
import re
from functools import lru_cache
from hfem.poisson_problems.solvers.homogenization.full_diffusion import HomogenizationAnalysis, HomogenizationConfig

class A:
    """Y-periodic diffusion tensor"""
    # @lru_cache(maxsize=1024)
    def __call__(self, x, y):
        return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)

class AEpsilon:
    """εY-periodic diffusion tensor"""
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.A = A()
    
    # @lru_cache(maxsize=1024)
    def __call__(self, x, y):
        return self.A(x/self.epsilon, y/self.epsilon)

class RightHandSide:
    """Right hand side function"""
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    # @lru_cache(maxsize=1024)
    # def __call__(self, x, y):
    #     return -(2*np.pi**2/self.epsilon)*(-self.epsilon*(np.sin(2*np.pi*x/self.epsilon) + 2)*(np.sin(2*np.pi*y/self.epsilon) + 4)*np.sin(np.pi*x)*np.sin(np.pi*y) \
    #     + (np.sin(2*np.pi*x/self.epsilon) + 2)*np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(2*np.pi*y/self.epsilon) \
    #     + (np.sin(2*np.pi*y/self.epsilon) + 4)*np.sin(np.pi*y)*np.cos(np.pi*x)*np.cos(2*np.pi*x/self.epsilon))
        
    def __call__(self, x, y):
        return -(2*np.pi**2)*(-(np.sin(2*np.pi*x) + 2)*(np.sin(2*np.pi*y) + 4)*np.sin(np.pi*x)*np.sin(np.pi*y) \
        + (np.sin(2*np.pi*x) + 2)*np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(2*np.pi*y) \
        + (np.sin(2*np.pi*y) + 4)*np.sin(np.pi*y)*np.cos(np.pi*x)*np.cos(2*np.pi*x))
        
    # def __call__(self, x, y):
    #     return 

def run_batch_analysis(epsilons, mesh_size=None, force_recompute=False):
    """Execute analysis for multiple epsilon values with a common mesh."""
    if mesh_size is None:
        mesh_size = min(epsilons)/4

    results = {}
    print(f"Running analysis for {len(epsilons)} epsilon values with h={mesh_size}")
    
    # Create a single analyzer instance and reuse it
    base_config = HomogenizationConfig(
        epsilon=epsilons[0],  # Will be updated
        mesh_size=mesh_size,
        A=A(),
        A_epsilon=None,  # Will be updated
        right_hand_side=None  # Will be updated
    )
    
    analyzer = HomogenizationAnalysis(base_config)
    
    for eps in epsilons:
        print(f"\nAnalyzing for epsilon = {eps}")
        
        # Update only the epsilon-dependent components
        analyzer.config = HomogenizationConfig(
            epsilon=eps,
            mesh_size=mesh_size,
            A=base_config.A,
            A_epsilon=AEpsilon(eps),
            right_hand_side=RightHandSide(eps)
        )
        
        results[eps] = analyzer.analyze(force_recompute)
        print(f"Results for epsilon = {eps}:")
        print(f"L2 error: {results[eps]['errors']['l2']}")
        print(f"H1 error: {results[eps]['errors']['h1']}")
        print(f"H1 error (corrected): {results[eps]['errors']['h1_corrected']}")
    
    return results

@lru_cache(maxsize=None)
def load_all_analyses(folder_path="simulation_data/homogenization_analysis"):
    """Load all analysis results from the specified folder with caching."""
    results = {}
    manager = FEMDataManager()
    
    analysis_path = Path(folder_path)
    analysis_files = list(analysis_path.glob("*.h5"))
    
    for file_path in analysis_files:
        match = re.search(r'eps_([0-9.]+)', file_path.stem)
        if match:
            epsilon = float(match.group(1))
            try:
                solution, mesh, _ = manager.load(file_path)
                results[epsilon] = {
                    'solutions': {
                        'u_epsilon': solution.data['u_epsilon'],
                        'u_0': solution.data['u_0'],
                        'u_1': solution.data['u_1']
                    },
                    'errors': solution.metadata['errors'],
                    'mesh': mesh
                }
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
    
    return dict(sorted(results.items()))

@conditional_style_context()
def plot_errors(results, save_name=None, show_convergence_rates=True):
    """Plot error analysis with optional convergence rates."""
    import cmasher as cmr
    epsilons = np.array(list(results.keys()))
    l2_errors = np.array([result['errors']['l2'] for result in results.values()])
    h1_errors = np.array([result['errors']['h1'] for result in results.values()])
    h1_corrected = np.array([result['errors']['h1_corrected'] for result in results.values()])
    
    colors = cmr.lavender(np.linspace(0, 1, 3))
    line_props = dict(marker='o', linestyle="--", markersize=4)
    
    fig, ax = plt.subplots()
    x_vals = np.log(1/epsilons)
    
    # Plot errors
    plots = [
        (np.log(l2_errors), r"$\log\left(\frac{\|e_{\varepsilon, h}\|_{L^2(\Omega)}}{\|u_{0,h}\|_{L^2(\Omega)}}\right)$", colors[0], 1),
        (np.log(h1_errors), r"$\log\left(\frac{|e_{\varepsilon, h}|_{H^1(\Omega)}}{|u_{0,h}|_{H^1(\Omega)}}\right)$", colors[1], None),
        (np.log(h1_corrected), r"$\log\left(\frac{|\tilde{e}_{\varepsilon, h}|_{H^1(\Omega)}}{|u_{\varepsilon, h}^*|_{H^1(\Omega)}}\right)$", colors[2], 0.5)
    ]
    
    for y_vals, label, color, rate in plots:
        ax.plot(x_vals, y_vals, label=label, color=color, **line_props)
        if show_convergence_rates and rate is not None:
            add_convergence_triangle(ax, x_vals, y_vals, rate=rate, color=color)
    
    # Add reference lines
    ref_line_props = {**line_props, 'color': 'gray', 'alpha': 0.5}
    # ax.plot(x_vals, np.log(epsilons), label=r'$O(\varepsilon)$', **ref_line_props)
    # ax.plot(x_vals, np.log(np.sqrt(epsilons)), label=r'$O(\sqrt{\varepsilon})$', **ref_line_props)
    
    ax.set_xlabel(r"$\log\left(1/\varepsilon\right)$")
    ax.legend()
    
    if save_name:
        plt.savefig(save_name)
    plt.show()

if __name__ == '__main__':
    lst_n = np.arange(start=25, stop=40 + 1, step=1)
    lst_n = np.arange(start=1, stop=40 + 1, step=2)
    epsilons = 1/lst_n
    # run_batch_analysis(epsilons, 
    #                 #    force_recompute=True
    #                    )
    
    # Run new analysis or load existing results
    # try:
    #     results = load_all_analyses()
    #     print("Loaded existing analysis results")
    # except Exception:
    #     print("Computing new analysis...")
    #     results = run_batch_analysis(epsilons)
    
    # plot_errors(
    #     results,
    #     # save_name="convergence_results/homogenized_convergence_comparison_cas.pdf"
    # )
    
    
    folder = "simulation_data/homogenization_analysis"
    file_path = "homogenization_analysis_eps_0.2_analysis.h5"
    # homogenization_analysis_eps_1.0_analysis.h5
    # homogenization_analysis_eps_0.14285714285714285_analysis.h5
    manager = FEMDataManager()
    solution, mesh, _ = manager.load(f"{folder}/{file_path}")
                # results[epsilon] = {
                #     'solutions': {
                #         'u_epsilon': solution.data['u_epsilon'],
                #         'u_0': solution.data['u_0'],
                #         'u_1': solution.data['u_1']
                #     },
                #     'errors': solution.metadata['errors'],
                #     'mesh': mesh
                # }
    # print(f"{solution.meta['epsilon']}")
    epsilon = solution.metadata['epsilon']
    print(f"{epsilon = }")
    u_eps = solution.data['u_epsilon']
    u_0 = solution.data['u_0']
    u_1 = solution.data['u_1']
    mesh.display_field(
        field=u_eps,
        field_label=r'$u_\varepsilon$',
        save_name="full_u_eps"
    )
    
    mesh.display_field(
        field=u_eps - u_0,
        field_label=r'$u_\varepsilon - u_0$',
        save_name="full_u_eps_u_0",
        # kind = 'trisurface'
    )
    
    mesh.display_field(
        field=u_eps - u_0 - epsilon * u_1,
        field_label=r'$u_\varepsilon - u_0 - \varepsilon u_1$',
        save_name="full_u_eps_u_0_u_1",
        # kind = 'trisurface'
    )
                
    # mesh.display_field(
    #     field=solution.data['u_0'],
    #     field_label=r'$u_0$'
    # )
    
    # mesh.display_field(
    #     field=solution.data['u_1'],
    #     field_label=r'$u_1$'
    # )