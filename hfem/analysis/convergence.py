import numpy as np
from pathlib import Path
from typing import Callable, List, Tuple, Union, Optional
from tqdm.auto import tqdm
import pandas as pd
import cmasher as cmr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
from hfem.viz.conditional_style_context import conditional_style_context
from hfem.poisson_problems.solvers.homogenization.cell import CellProblem, CellProblemConfig
from hfem.poisson_problems.solvers.homogenization.homogenized import HomogenizedConfig, HomogenizedProblem



def compute_and_save_derivatives_convergence_rates(
    effective_tensor: np.ndarray,
    exact_derivatives: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    right_hand_side: Callable[[np.ndarray, np.ndarray], np.ndarray],
    h_values: List[float],
    domain_size: Tuple[float, float] = (1.0, 1.0),
    save_dir: Path = Path("convergence_results")
    ) -> None:
    L_x, L_y = domain_size
    l2_errors_x = []
    l2_errors_y = []
    h1_errors_x = []
    h1_errors_y = []
    
    for h in tqdm(h_values, desc=f"Computing convergence", leave=False):
        mesh_file = f"meshes/conv_mesh_h{h}.msh"
        rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
        mesh = CustomTwoDimensionMesh(mesh_file)
        config = HomogenizedConfig(
            mesh=mesh,
            mesh_size=h,
            effective_tensor=effective_tensor,
            right_hand_side=right_hand_side
        )
        problem = HomogenizedProblem(config=config)
        problem.solve_and_save(save_name=f"conv_h{h}")
        
        # Calcul des erreurs
        dx, dy = exact_derivatives(*problem.config.mesh.node_coords.T)
        
        error_x = dx - problem.solution_derivatives[0]
        error_y = dy - problem.solution_derivatives[1]
        
        # Erreur L2
        l2_error_x = np.sqrt(error_x.T @ problem.mass_matrix @ error_x)/np.sqrt(dx.T @ problem.mass_matrix @ dx)
        l2_errors_x.append(l2_error_x)
        
        l2_error_y = np.sqrt(error_y.T @ problem.mass_matrix @ error_y)/np.sqrt(dy.T @ problem.mass_matrix @ dy)
        l2_errors_y.append(l2_error_y)
        
        # Erreur H1
        h1_error_x = np.sqrt(error_x.T @ problem.stiffness_matrix @ error_x)/np.sqrt(dx.T @ problem.stiffness_matrix @ dx)
        h1_errors_x.append(h1_error_x)
        
        h1_error_y = np.sqrt(error_y.T @ problem.stiffness_matrix @ error_y)/np.sqrt(dy.T @ problem.stiffness_matrix @ dy)
        h1_errors_y.append(h1_error_y)
    
    l2_errors_x = np.array(l2_errors_x)
    l2_errors_y = np.array(l2_errors_y)
    h1_errors_x = np.array(h1_errors_x)
    h1_errors_y = np.array(h1_errors_y)
    
    
    # Sauvegarde des résultats dans un fichier csv
    results = pd.DataFrame({
        'h': h_values,
        'L2_error_x': l2_errors_x,
        'L2_error_y': l2_errors_y,
        'H1_error_x': h1_errors_x,
        'H1_error_y': h1_errors_y,
    })
    results.to_csv(save_dir / f"{str(config.problem_type)}_convergence_results.csv", index=False)
    
@conditional_style_context()
def plot_derivatives_convergence_from_csv(csv_path: Union[str, Path], save_name: Optional[str] = None) -> None:
    # Lecture des données
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(layout='constrained')
    colors = cmr.lavender(np.linspace(0, 1, 2))
    line_props = dict(linestyle="--", markersize=4)
    
    # Plot des erreurs
    log_1_h = np.log(1/df['h'])
    ax.plot(log_1_h, np.log(df['L2_error_x']), label=r'$\|\partial_x \cdot\|_{L^2(\Omega)}$', color=colors[0], **line_props, marker='o')
    ax.plot(log_1_h, np.log(df['L2_error_y']), label=r'$\|\partial_y\cdot\|_{L^2(\Omega)}$', color=colors[0], **line_props, marker='^')
    ax.plot(log_1_h, np.log(df['H1_error_x']), label=r'$|\partial_x\cdot|_{H^1(\Omega)}$', color=colors[1], **line_props, marker='o')
    ax.plot(log_1_h, np.log(df['H1_error_y']), label=r'$|\partial_y\cdot|_{H^1(\Omega)}$', color=colors[1], **line_props, marker='^')
    
    # Ajouter le triangle de convergence
    add_convergence_triangle(ax, log_1_h, np.log(df['L2_error_x']), rate=2, color=colors[0])
    add_convergence_triangle(ax, log_1_h, np.log(df['H1_error_x']), rate=1, color=colors[1])
    
    ax.set_xlabel(r'$\log(1/h)$')
    ax.set_ylabel(r'$\log\left(\frac{\|\partial_i u_0 - \partial_i u_{0,h}\|}{\|\partial_i u_0\|}\right)$')
    ax.legend()
    
    if save_name:
        plt.savefig(f"{save_name}")
    plt.show()

def compute_and_save_cell_convergence_rates(
    diffusion_tensor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    exact_homogenized_tensor: np.ndarray,
    h_values: List[float],
    eta_values: List[float],
    save_dir: Path = Path("convergence_results")
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    L_x, L_y = 1, 1
    
    # Liste pour stocker tous les résultats
    results_data = []
    
    # Pour chaque valeur de eta
    for eta in tqdm(eta_values, desc="Testing eta values"):
        # Pour chaque taille de maille
        for h in tqdm(h_values, desc=f"Computing convergence for eta={eta}", leave=False):
            # Création du maillage
            # print(f"{eta = }, {h = }")
            mesh_file = f"meshes/conv_mesh_h{h}.msh"
            rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
            mesh = CustomTwoDimensionMesh(mesh_file)
            
            # Configuration et résolution
            config = CellProblemConfig(
                mesh=mesh,
                mesh_size=h,
                diffusion_tensor=diffusion_tensor,
                eta=eta
            )
            
            problem = CellProblem(config=config)
            problem.solve_and_save(save_name=f"conv_h{h}_eta{eta}")
            
            # Calcul des erreurs
            diff = exact_homogenized_tensor - problem.homogenized_tensor
            abs_error = np.linalg.norm(diff)
            rel_error = abs_error/np.linalg.norm(exact_homogenized_tensor)
            
            # Stockage des résultats
            results_data.append({
                'eta': eta,
                'h': h,
                'absolute_error': abs_error,
                'relative_error': rel_error
            })
    
    # Sauvegarde dans un unique fichier csv
    df = pd.DataFrame(results_data)
    df.to_csv(save_dir / "cell_convergence_results.csv", index=False)


@conditional_style_context()
def plot_cell_convergence_from_csv(csv_path: Union[str, Path], save_name: Optional[str] = None, rate: Optional[float] = None) -> None:
    # Lecture des données
    df = pd.read_csv(csv_path)
    eta_values = sorted(df['eta'].unique())
    colors = cmr.lavender(np.linspace(0, 1, len(eta_values)))

    fig, ax = plt.subplots(layout='constrained')
    line_props = dict(marker = 'o', linestyle = "--", markersize = 4, 
                    #   alpha = 0.75
                      )
    # Plot pour chaque eta
    eta_handles = []
    for eta, color in zip(eta_values[::-1], colors[::-1]):
        # eta_handles.append(Line2D([0], [0], color=color, lw=2, label = fr"${eta:.2g}$"))
        df_eta = df[df['eta'] == eta]
        # log_h = np.log(1/df_eta['h'])
        # log_error = np.log(df_eta['absolute_error'])
        h = df_eta['h']
        error = df_eta['absolute_error']
        
        ax.plot(np.log(1/h), np.log(error), color=color, label=fr"${eta:1.2g}$", **line_props)
    
    # Ajout des pentes de référence
    # h_ref = np.array([min(df['h']), max(df['h'])])
    # error_scale = df['absolute_error'].max() * 0.5
    # ax.plot(h_ref, error_scale * (h_ref/h_ref[0])**2, 'k:', alpha=0.5, label=r'$O(h^2)$')
    
    ax.set_xlabel(r'$\log(1/h)$')
    ax.set_ylabel(r'$\log\left(\|A^* - A_{h, \eta}^*\|_F\right)$')
    # fig.legend(handles=eta_handles, loc = 'outside right center', frameon = True, title = r"$\eta$ values")
    fig.legend(loc = 'outside right center', 
            #    frameon = True, 
               title = r"$\eta$ values")
    
    if rate:
        min_eta = np.array(eta_values).min()
        df_eta = df[df['eta'] == min_eta]
        
        # data_last_eta = eta_sorted_data[0][1]
        color = colors[0]
        h = df_eta['h']
        error = df_eta['absolute_error']
        add_convergence_triangle(
            ax=ax, 
            x = np.log(1/np.array(df_eta['h'])),
            y = np.log(np.array(df_eta['absolute_error'])), 
            rate=rate, color = color)
    
    
    if save_name:
        plt.savefig(f"{save_name}")
    
    plt.show()
    return fig, ax



def compute_convergence_rates(
    problem_class,
    config_class,
    exact_solution: Callable[[np.ndarray, np.ndarray], np.ndarray],
    diffusion_tensor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    right_hand_side: Callable[[np.ndarray, np.ndarray], np.ndarray],
    h_values: List[float],
    domain_size: Tuple[float, float] = (1.0, 1.0),
    save_dir: Path = Path("convergence_results")
) -> None:
    """
    Calcule les taux de convergence pour un problème donné.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    l2_errors = []
    h1_errors = []
    L_x, L_y = domain_size
    
    for h in tqdm(h_values, desc="Computing convergence"):
        # Création du maillage
        mesh_file = f"meshes/conv_mesh_h{h}.msh"
        rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
        mesh = CustomTwoDimensionMesh(mesh_file)
        
        # Configuration et résolution
        config = config_class(
            mesh=mesh,
            mesh_size=h,
            diffusion_tensor=diffusion_tensor,
            right_hand_side=right_hand_side
        )
        
        problem = problem_class(config=config)
        problem.solve_and_save(save_name=f"conv_h{h}")
        
        # Calcul des erreurs
        exact = exact_solution(*problem.config.mesh.node_coords.T)
        error = exact - problem.solution
        
        # Erreur L2
        l2_error = np.sqrt(error.T @ problem.mass_matrix @ error)/np.sqrt(exact.T @ problem.mass_matrix @ exact)
        l2_errors.append(l2_error)
        
        # Erreur H1
        h1_error = np.sqrt(error.T @ problem.stiffness_matrix @ error)/np.sqrt(exact.T @ problem.stiffness_matrix @ exact)
        h1_errors.append(h1_error)
    
    l2_errors = np.array(l2_errors)
    h1_errors = np.array(h1_errors)
    
    
    # Sauvegarde des résultats dans un fichier csv
    results = pd.DataFrame({
        'h': h_values,
        'L2_error': l2_errors,
        'H1_error': h1_errors
    })
    results.to_csv(save_dir / f"{str(config.problem_type)}_convergence_results.csv", index=False)


@conditional_style_context()
def plot_convergence_from_csv(csv_path: Union[str, Path], save_name: Optional[str] = None):
    """
    Affiche les résultats de convergence à partir d'un fichier CSV.
    
    Le fichier CSV doit contenir les colonnes : h, L2_error, H1_error
    """
    import matplotlib.pyplot as plt
    # Lecture des données
    results = pd.read_csv(csv_path)
    h_values = results['h'].values
    l2_errors = results['L2_error'].values
    h1_errors = results['H1_error'].values
    colors = cmr.lavender(np.linspace(0, 1, 2))
    fig, ax = plt.subplots()
    # print(colors)
    # Plot des erreurs
    log_1_h = np.log(1/h_values)
    line_props = dict(marker = 'o', linestyle = "--", markersize = 4)
    ax.plot(log_1_h, np.log(l2_errors), label=r'$\|\cdot\|_{L^2(\Omega)}$', color = colors[0], **line_props)
    ax.plot(log_1_h, np.log(h1_errors), label=r'$|\cdot|_{H^1(\Omega)}$', color = colors[1], **line_props)
    add_convergence_triangle(ax, log_1_h, np.log(l2_errors), rate=2, color=colors[0])
    add_convergence_triangle(ax, log_1_h, np.log(h1_errors), rate=1, color=colors[1])
    
    ax.set_ylabel(r'$\log\left(\frac{\|u - u_h\|}{\|u\|} \right)$')
    ax.set_xlabel(r'$\log\left(1/h\right)$')
    ax.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(f"{save_name}.pdf")
    plt.show()

# @conditional_style_context()
# def plot_standard_convergence(data: StandardConvergenceData, 
#                             save_name: Optional[str] = None) -> tuple[plt.Figure, plt.Axes]:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     log_h = np.log(1/np.array(data.h))
    
#     # Plot L2 and H1 errors
#     ax.plot(log_h, np.log(data.l2_errors), 'o--', label=r'$L^2$ Error', color='#D1453D')
#     ax.plot(log_h, np.log(data.h1_errors), 'o--', label=r'$H^1$ Error', color='#5B9276')
    
#     # Add expected convergence triangles
#     add_convergence_triangle(ax, log_h, np.log(data.l2_errors), rate=2, color='#D1453D')
#     add_convergence_triangle(ax, log_h, np.log(data.h1_errors), rate=1, color='#5B9276')
    
#     ax.set_xlabel(r'$\log(1/h)$')
#     ax.set_ylabel(r'$\log(\textrm{Error})$')
#     ax.legend()
    
#     if save_name:
#         plt.savefig(f"{save_name}.pdf")
#     return fig, ax

def add_convergence_triangle(ax: plt.Axes, x: np.ndarray, y: np.ndarray, 
                        rate: float, color: str) -> None:
    """Add convergence rate triangle to plot."""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    tri_x = np.array([x_max-1, x_max, x_max-1])
    tri_y = np.array([y_min, y_min, y_min+rate])
    ax.add_patch(plt.Polygon(np.column_stack((tri_x, tri_y)), 
                           facecolor=color, alpha=0.3))
    # ax.text(tri_x[0] + 0.5, tri_y[0] + rate/2, str(rate),
    #         color=color, fontsize=12, weight='bold')
    
    ax.annotate(text = r'$\boldsymbol{' + str(rate) + r'}$',
        xy=((tri_x[-1] + tri_x[0])/2, 
            (tri_y[-1] + tri_y[0])/2),
        xytext=(2*rate, -2*rate), textcoords='offset points',
        va='center', ha='left', color=color, fontsize=12, weight='bold')









def v(x,y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + 2

def diffusion_tensor(x, y):
            return np.eye(2)*v(x, y)
        
def exact_solution(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def right_hand_side(x, y):
    return (1 + (16*np.pi**2)*(v(x, y)- 1))*exact_solution(x, y)


# Exemple d'utilisation
if __name__ == "__main__":
    from hfem.poisson_problems.solvers.standard.dirichlet import DirichletHomogeneousProblem
    from hfem.poisson_problems.configs.standard.dirichlet import DirichletConfig
    
    # Test de convergence
    # h_values = np.array([0.075, 0.05, 0.025, 0.02, 0.015, 0.01, 0.0095, 0.009, 0.0085, 0.008, 0.0075])
    # l2_errors, h1_errors, l2_rates, h1_rates = compute_convergence_rates(
    #     DirichletHomogeneousProblem,
    #     DirichletConfig,
    #     exact_solution,
    #     diffusion_tensor,
    #     right_hand_side,
    #     h_values,
    #     domain_size = (1, 1)
    # )
    
    
    # plot_convergence_from_csv("convergence_results/dirichlet_convergence_results.csv",
    #                         #   save_name="convergence_results/dirichlet_convergence_results"
    #                           )
    colors = cmr.lavender(np.linspace(0, 1, 4))
    print(colors)
    print(cmr.lavender_r(np.linspace(0, 1, 4)))
    

    cmr.create_cmap_overview(cmaps=[cmr.lavender, cmr.iceburn_r, cmr.wildfire])