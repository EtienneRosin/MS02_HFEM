from hfem.poisson_problems.solvers.homogenization.full_diffusion import HomogenizationAnalysis, HomogenizationConfig
from hfem.viz.conditional_style_context import conditional_style_context
import numpy as np
from hfem.analysis.convergence import add_convergence_triangle





class A:
    """Y-periodic diffusion tensor"""
    def __call__(self, x, y):
        return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)

class AEpsilon:
    """εY-periodic diffusion tensor"""
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.A = A()
    
    def __call__(self, x, y):
        return self.A(x/self.epsilon, y/self.epsilon)

class RightHandSide:
    """Right hand side function"""
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self, x, y):
        return -(2*np.pi**2/self.epsilon)*(
            -self.epsilon*(np.sin(2*np.pi*x/self.epsilon) + 2)*
            (np.sin(2*np.pi*y/self.epsilon) + 4)*np.sin(np.pi*x)*np.sin(np.pi*y) +
            (np.sin(2*np.pi*x/self.epsilon) + 2)*np.sin(np.pi*x)*np.cos(np.pi*y)*
            np.cos(2*np.pi*y/self.epsilon) +
            (np.sin(2*np.pi*y/self.epsilon) + 4)*np.sin(np.pi*y)*np.cos(np.pi*x)*
            np.cos(2*np.pi*x/self.epsilon)
        )
        

def run_batch_analysis(epsilons, mesh_size=None, force_recompute=False):
    """Exécute l'analyse pour plusieurs valeurs d'epsilon avec un maillage commun."""
    if mesh_size is None:
        mesh_size = min(epsilons)/3

    results = {}
    print(f"Running analysis for {len(epsilons)} epsilon values with h={mesh_size}")
    
    for eps in epsilons:
        print(f"\nAnalyzing for epsilon = {eps}")
        config = HomogenizationConfig(
            epsilon=eps,
            mesh_size=mesh_size,
            A=A(),
            A_epsilon=AEpsilon(eps),
            right_hand_side=RightHandSide(eps)
        )
        
        analyzer = HomogenizationAnalysis(config)
        results[eps] = analyzer.analyze(force_recompute)
        
        print(f"Results for epsilon = {eps}:")
        print(f"L2 error: {results[eps]['errors']['l2']}")
        print(f"H1 error: {results[eps]['errors']['h1']}")
        print(f"H1 error (corrected): {results[eps]['errors']['h1_corrected']}")
    
    return results

from pathlib import Path
from hfem.core.io import FEMDataManager
import re

def load_all_analyses(folder_path="simulation_data/homogenization_analysis"):
    """
    Charge tous les résultats d'analyse stockés dans le dossier spécifié.
    
    Returns:
        dict: Dictionnaire {epsilon: résultats} où epsilon est extrait du nom du fichier
    """
    results = {}
    manager = FEMDataManager()
    
    # Liste tous les fichiers .h5 dans le dossier
    analysis_path = Path(folder_path)
    analysis_files = list(analysis_path.glob("*.h5"))
    
    for file_path in analysis_files:
        # Extraire epsilon du nom du fichier
        # Supposant que le nom est de la forme "analysis_eps_0.1_analysis.h5"
        match = re.search(r'eps_([0-9.]+)', file_path.stem)
        if match:
            epsilon = float(match.group(1))
            
            try:
                # Charger les résultats
                solution, mesh, _ = manager.load(file_path)
                
                # Reconstruire le dictionnaire de résultats
                results[epsilon] = {
                    'solutions': {
                        'u_epsilon': solution.data['u_epsilon'],
                        'u_0': solution.data['u_0'],
                        'u_1': solution.data['u_1']
                    },
                    'errors': solution.metadata['errors'],
                    'mesh': mesh
                }
                # print(f"Loaded results for epsilon = {epsilon}")
                
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
    
    return dict(sorted(results.items()))  # Retourne les résultats triés par epsilon


import matplotlib.pyplot as plt
import numpy as np
@conditional_style_context()
def plot_errors(results, save_name = None):
    import cmasher as cmr
    epsilons = np.array(list(results.keys()))
    l2_errors = np.array([result['errors']['l2'] for result in results.values()])
    h1_errors = np.array([result['errors']['h1'] for result in results.values()])
    h1_corrected = np.array([result['errors']['h1_corrected'] for result in results.values()])
    colors = cmr.lavender(np.linspace(0, 1, 3))
    # print(f"{len(epsilons) = }")
    # plt.figure(figsize=(10, 6))
    line_props = dict(marker = 'o', linestyle = "--", markersize = 4, 
                    #   alpha = 0.75
                      )
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # plt.plot(np.log(1/epsilons))
    
    ax.plot(
        np.log(1/epsilons), np.log(l2_errors), 
        label=r"$\log\left(\frac{\|e_{\varepsilon, h}\|_{L^2(\Omega)}}{\|u_{0,h}\|_{L^2(\Omega)}}\right)$", 
        color = colors[0], **line_props)
    ax.plot(
        np.log(1/epsilons), np.log(h1_errors), 
        label=r"$\log\left(\frac{|e_{\varepsilon, h}|_{L^2(\Omega)}}{|u_{0,h}|_{H^1(\Omega)}}\right)$", 
        color = colors[1], **line_props)
    ax.plot(
        np.log(1/epsilons), np.log(h1_corrected), 
        label=r"$\log\left(\frac{\|\tilde{e}_{\varepsilon, h}\|_{L^2(\Omega)}}{\|u_{\varepsilon, h}^*\|_{L^2(\Omega)}}\right)$", 
        color = colors[2], **line_props)
    
    add_convergence_triangle(ax, np.log(1/epsilons), np.log(l2_errors), rate=1, color=colors[0])
    add_convergence_triangle(ax, np.log(1/epsilons), np.log(h1_corrected), rate=0.5, color=colors[2])
    ax.plot(
        np.log(1/epsilons), 
        np.log(epsilons)
        , **line_props)
    ax.plot(
        np.log(1/epsilons), 
        np.log(np.sqrt(epsilons))
        , **line_props)
    
    
    ax.set(
        xlabel = r"$\log\left(1/\varepsilon\right)$",
        # ylabel = r"$\log\left(\frac{\|e_h\|}{\|u\|}\right)$",
    )
    
    ax.legend()
    # r"$\log\left(\frac{\|e_{\varepsilon, h}\|_{L^2(\Omega)}}{\|u_{0,h}\|_{L^2(\Omega)}}\right)$"
    # e_{\varepsilon, h} = u_{\varepsilon, h} - u_{0, h}
    # \tilde{e}_{\varepsilon, h} = u_0 + \varepsilon u_1(\cdot, \cdot/\varepsilon)
    # u_{\varepsilon, h}^*
    
    
    # plt.loglog(epsilons, l2_errors, 'o-', label='L2 error')
    # plt.loglog(epsilons, h1_errors, 's-', label='H1 error')
    # plt.loglog(epsilons, h1_corrected, '^-', label='H1 corrected error')
    # plt.grid(True)
    # plt.xlabel('ε')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.title('Convergence analysis')
    if save_name:
        plt.savefig(f"{save_name}")
    plt.show()



if __name__ == '__main__':
    # lst_n = np.arange(start=1, stop=30 + 1, step=1)
    lst_n = np.arange(start=25, stop=30 + 1, step=1)
    epsilons = 1/lst_n
    results = run_batch_analysis(epsilons)
    
    
    
    
    # print(f"{epsilons = }")
    
    # Utilisation
    results = load_all_analyses()
    # print("\nEpsilon values found:", list(results.keys()))

    # # Exemple d'utilisation des résultats
    # for eps, result in results.items():
    #     print(f"\nResults for epsilon = {eps}:")
    #     print(f"L2 error: {result['errors']['l2']}")
    #     print(f"H1 error: {result['errors']['h1']}")
    #     print(f"H1 error (corrected): {result['errors']['h1_corrected']}")
    
    plot_errors(
        results,
        # save_name="convergence_results/homogenized_convergence_comparison.pdf"
        )