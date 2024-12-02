from typing import Union, Optional
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import cmasher as cmr

from hfem.viz.conditional_style_context import conditional_style_context
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


@dataclass
class StandardConvergenceData:
    """Standard Poisson problem convergence data."""
    h: list[float]
    n_nodes: list[int]
    l2_errors: list[float]
    h1_errors: list[float]
    boundary_type: str

@dataclass
class PenalizedCellConvergenceData:
    """Penalized cell problems convergence data."""
    eta: float
    h: list[float]
    n_nodes: list[int]
    l2_errors_corrector1: list[float]
    h1_errors_corrector1: list[float]
    l2_errors_corrector2: list[float]
    h1_errors_corrector2: list[float]
    tensor_errors: list[float]

@dataclass
class MultiEtaPenalizedCellConvergenceData:
    data: dict[float, PenalizedCellConvergenceData]   

def save_data(data: Union[StandardConvergenceData, PenalizedCellConvergenceData, MultiEtaPenalizedCellConvergenceData], 
              save_dir: Union[str, Path], 
              filename: Optional[str] = None) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, StandardConvergenceData):
        if filename is None:
            filename = f"convergence_{data.boundary_type}.csv"
        save_path = save_dir / filename
        pd.DataFrame(data.__dict__).to_csv(save_path, index=False)
    elif isinstance(data, PenalizedCellConvergenceData):
        if filename is None:
            filename = f"penalized_cell_convergence.csv"
        save_path = save_dir / filename
        pd.DataFrame(data.__dict__).to_csv(save_path, index=False)
    else:
        if filename is None:
            filename = "multi_eta_convergence.csv"
        save_path = save_dir / filename
        
        # Combine all eta data into one DataFrame
        all_data = []
        for eta, eta_data in data.data.items():
            df = pd.DataFrame(eta_data.__dict__)
            all_data.append(df)
        pd.concat(all_data).to_csv(save_path, index=False)
    
    return save_path

def read_data(filepath: Union[str, Path], data_type: str = 'standard') -> Union[StandardConvergenceData, MultiEtaPenalizedCellConvergenceData]:
    df = pd.read_csv(filepath)
    
    if data_type == 'standard':
        return StandardConvergenceData(**df.to_dict('list'))
    elif data_type == 'penalized cell':
        return PenalizedCellConvergenceData(**df.to_dict('list'))
    elif data_type == 'multi-eta penalized cell':
        data_dict = {}
        for eta in df['eta'].unique():
            eta_df = df[df['eta'] == eta]
            data_dict[eta] = PenalizedCellConvergenceData(**eta_df.to_dict('list'))
        return MultiEtaPenalizedCellConvergenceData(data=data_dict)
    
    raise ValueError("data_type must be 'standard', 'penalized cell' or 'multi-eta penalized cell'")



@conditional_style_context()
def plot_standard_convergence(data: StandardConvergenceData, 
                            save_name: Optional[str] = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))
    log_h = np.log(1/np.array(data.h))
    
    # Plot L2 and H1 errors
    ax.plot(log_h, np.log(data.l2_errors), 'o--', label=r'$L^2$ Error', color='#D1453D')
    ax.plot(log_h, np.log(data.h1_errors), 'o--', label=r'$H^1$ Error', color='#5B9276')
    
    # Add expected convergence triangles
    add_convergence_triangle(ax, log_h, np.log(data.l2_errors), rate=2, color='#D1453D')
    add_convergence_triangle(ax, log_h, np.log(data.h1_errors), rate=1, color='#5B9276')
    
    ax.set_xlabel(r'$\log(1/h)$')
    ax.set_ylabel(r'$\log(\textrm{Error})$')
    ax.legend()
    
    if save_name:
        plt.savefig(f"{save_name}.pdf")
    return fig, ax

@conditional_style_context()
def plot_corrector_convergence(data: MultiEtaPenalizedCellConvergenceData,
                             save_name: Optional[str] = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(layout = 'constrained'
                           )
    # colors = plt.cm.viridis(np.linspace(0, 1, len(data.data)))
    colors = cmr.lavender(np.linspace(0, 1, len(data.data)))
    
    # custom_handles = [
    #     Line2D([0], [0], color="black", lw=2, linestyle="--", label = r"$\|\cdot\|_{L^2(Y)}$"),
    #     Line2D([0], [0], color="black", lw=2, linestyle=":", label = r"$\|\cdot\|_{H^1(Y)}$"),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label=r'$e_{1,h}^\eta$'),
    #     Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8, label=r'$e_{2,h}^\eta$')
    #     ]
    custom_handles = [
        # Line2D([0], [0], color="black", lw=2, linestyle="--", label = r"$\|\cdot\|_{L^2(Y)}$"),
        Line2D([0], [0], color="black", lw=2, linestyle=":", label = r"$\|\cdot\|_{H^1(Y)}$"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label=r'$e_{1,h}^\eta$'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8, label=r'$e_{2,h}^\eta$')
        ]
    

    fig.legend(
        handles=custom_handles, 
        loc = 'outside right upper',
        frameon = True, title = r"Symbols"
        # title = "iubc", 
        # ncols = 2
        )
    
    eta_handles = []
    for (eta, conv_data), color in zip(sorted(data.data.items()), colors):
        eta_handles.append(Line2D([0], [0], color=color, lw=2, label = fr"${eta:.2g}$"))
        log_h = np.log(1/np.array(conv_data.h))
        
        # First corrector
        # ax.plot(log_h, np.log(conv_data.l2_errors_corrector1), 'o--', color=color, alpha=0.7)
        # ax.plot(log_h, np.log(conv_data.h1_errors_corrector1), 'o:', color=color, alpha=0.3)
        
        # # Second corrector
        # ax.plot(log_h, np.log(conv_data.l2_errors_corrector2), '^--', color=color, alpha=0.7)
        # ax.plot(log_h, np.log(conv_data.h1_errors_corrector2), '^:', color=color, alpha=0.3)
        
        #____________________
        ax.plot(conv_data.h, conv_data.l2_errors_corrector1, 'o--', color=color, alpha=0.7)
        ax.plot(conv_data.h, conv_data.h1_errors_corrector1, 'o:', color=color, alpha=0.7)
        
        # Second corrector
        ax.plot(conv_data.h, conv_data.l2_errors_corrector2, '^--', color=color, alpha=0.7)
        ax.plot(conv_data.h, conv_data.h1_errors_corrector2, '^:', color=color, alpha=0.7)
    
    # ax.set_xlabel(r'$\log(1/h)$')
    # ax.set_ylabel(r'$\log(\|e_{i,h}^\eta\|)$')
    
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\|e_{i,h}^\eta\|$')
    
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handles=eta_handles, 
    #         #   title = "$\eta$"
    #           )
    fig.legend(handles=eta_handles, loc = 'outside right center', frameon = True, title = r"$\eta$ values")
    # ax.legend(bbox_to_anchor=(0.05, 1.05), loc='upper left')
    
    # plt.tight_layout()
    
    if save_name:
        plt.savefig(f"{save_name}")
    return fig, ax

@conditional_style_context()
def plot_tensor_convergence(data: MultiEtaPenalizedCellConvergenceData,
                          save_name: Optional[str] = None, rate = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(layout = 'constrained')
    ax.set(xlabel = r'$\log\left(\frac{1}{h}\right)$', ylabel = r'$\log\left(\frac{\left\|A^* - A^*_\eta\right\|_F}{\left\|A^*\right\|_F}\right)$')
    colors = cmr.lavender(np.linspace(0, 1, len(data.data)))
    
    eta_sorted_data = sorted(data.data.items())
    
    for i, (eta, conv_data) in enumerate(eta_sorted_data):
        log_h = np.log(1/np.array(conv_data.h))
        ax.plot(log_h, np.log(conv_data.tensor_errors), 'o--', 
                label=fr'${eta:.2g}$', color = colors[i], markersize = 4)
    
    if rate:
        data_last_eta = eta_sorted_data[0][1]
        color = colors[0]
        add_convergence_triangle(
            ax=ax, 
            x = np.log(1/np.array(data_last_eta.h)),
            y = np.log(np.array(data_last_eta.tensor_errors)), 
            rate=rate, color = color)
    
    fig.legend(loc = 'outside right center', frameon = True, title = r"$\eta$ values")
    
    if save_name:
        plt.savefig(f"{save_name}")
    return fig, ax

def add_convergence_triangle(ax: plt.Axes, x: np.ndarray, y: np.ndarray, 
                           rate: float, color: str) -> None:
    """Add convergence rate triangle to plot."""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    tri_x = np.array([x_max-1, x_max, x_max-1])
    tri_y = np.array([y_min, y_min, y_min+rate])
    
        # **annotation_props

    
    
    
    ax.add_patch(plt.Polygon(np.column_stack((tri_x, tri_y)), 
                           facecolor=color, alpha=0.3))
    # ax.text(tri_x[0] + 0.5, tri_y[0] + rate/2, str(rate),
    #         color=color, fontsize=12, weight='bold')
    
    ax.annotate(text = r'$\boldsymbol{' + str(rate) + r'}$',
        xy=((tri_x[-1] + tri_x[0])/2, 
            (tri_y[-1] + tri_y[0])/2),
        xytext=(2*rate, -2*rate), textcoords='offset points',
        va='center', ha='left', color=color, fontsize=12, weight='bold')
    
    
    
from dataclasses import replace
from typing import Callable, Dict, List, Optional
import numpy as np
from tqdm import tqdm

from hfem.problems import BasePoissonConfig, BasePoissonProblem, PenalizedCellProblemConfig, PenalizedCellProblems
from hfem.mesh_manager import CustomTwoDimensionMesh

def measure_poisson_convergence(
    config: BasePoissonConfig,
    mesh_sizes: List[float],
    mesh_generator: Callable,
    mesh_config: Optional[Dict] = None
) -> StandardConvergenceData:
    """Mesure la convergence pour un problème de Poisson standard."""
    h_values = []
    n_nodes = []
    l2_errors = []
    h1_errors = []
    
    for h in tqdm(mesh_sizes, desc="Measuring convergence"):
        mesh_file = f"temp_mesh_{h:.6f}.msh"
        mesh_generator(h=h, save_name=mesh_file, **(mesh_config or {}))
        
        try:
            mesh = CustomTwoDimensionMesh(mesh_file)
            solver = BasePoissonProblem(mesh, config)
            solver.solve()
            
            h_values.append(h)
            n_nodes.append(mesh.num_nodes)
            l2_errors.append(solver.l2_error)
            h1_errors.append(solver.h1_error)
            
        finally:
            Path(mesh_file).unlink(missing_ok=True)
            
    return StandardConvergenceData(
        h=h_values,
        n_nodes=n_nodes,
        l2_errors=l2_errors,
        h1_errors=h1_errors,
        boundary_type='Dirichlet'
    )

def measure_penalized_convergence(
    base_config: PenalizedCellProblemConfig,
    eta_values: List[float],
    mesh_sizes: List[float],
    mesh_generator: Callable,
    mesh_config: Optional[Dict] = None
) -> MultiEtaPenalizedCellConvergenceData:
    """Mesure la convergence pour différentes valeurs de eta."""
    eta_data = {}
    # print(f"{mesh_config = }")
    for eta in eta_values:
        config = replace(base_config, eta=eta)
        
        h_values = []
        n_nodes = []
        l2_c1, h1_c1 = [], []
        l2_c2, h1_c2 = [], []
        tensor_errors = []
        
        for h in mesh_sizes:
            mesh_file = f"temp_mesh_{h:.6f}.msh"
            mesh_generator(h=h, save_name=mesh_file, **(mesh_config or {}))
            
            try:
                mesh = CustomTwoDimensionMesh(mesh_file)
                # mesh.display()
                solver = PenalizedCellProblems(mesh, config)
                solver.solve()
                # solver.display_corrector_errors()
                # plt.show()
                
                h_values.append(h)
                n_nodes.append(mesh.num_nodes)
                if solver.l2_errors:
                    l2_c1.append(solver.l2_errors[0])
                    h1_c1.append(solver.h1_errors[0])
                    l2_c2.append(solver.l2_errors[1])
                    h1_c2.append(solver.h1_errors[1])
                else:
                    l2_c1.append(None)
                    h1_c1.append(None)
                    l2_c2.append(None)
                    h1_c2.append(None)
                
                tensor_errors.append(solver.homogenized_tensor_error)
                
                # tensor_error = np.linalg.norm(
                #     solver.homogenized_tensor - config.exact_homogenized_tensor
                # ) / np.linalg.norm(config.exact_homogenized_tensor)
                
                
            finally:
                Path(mesh_file).unlink(missing_ok=True)
                
        eta_data[eta] = PenalizedCellConvergenceData(
            eta=eta,
            h=h_values,
            n_nodes=n_nodes,
            l2_errors_corrector1=l2_c1,
            h1_errors_corrector1=h1_c1,
            l2_errors_corrector2=l2_c2,
            h1_errors_corrector2=h1_c2,
            tensor_errors=tensor_errors
        )
        
    return MultiEtaPenalizedCellConvergenceData(data=eta_data)

if __name__ == '__main__':
    
    
    # A = PenalizedCellConvergenceData(
    #     eta = 3e-6,
    #     h = [0.1, 0.05, 0.025],
    #     n_nodes= [10, 100, 1000],
    #     l2_errors_corrector1=[0.5, 0.1, 0.02],
    #     h1_errors_corrector1=[0.5, 0.1, 0.02],
    #     l2_errors_corrector2=[0.5, 0.1, 0.02],
    #     h1_errors_corrector2=[0.5, 0.1, 0.02],
    #     tensor_errors=[0.5, 0.1, 0.02]
    # )
    # B = PenalizedCellConvergenceData(
    #     eta = 2,
    #     h = [0.1, 0.05, 0.025],
    #     n_nodes= [10, 100, 1000],
    #     l2_errors_corrector1=[0.5, 0.1, 0.02],
    #     h1_errors_corrector1=[0.5, 0.1, 0.02],
    #     l2_errors_corrector2=[0.5, 0.1, 0.02],
    #     h1_errors_corrector2=[0.5, 0.1, 0.02],
    #     tensor_errors=[0.5, 0.1, 0.02]
    # )
    # # print(A.__dict__)
    # # for key, value in A.__dict__.items():
    # #     print(f"{key}: {value}")
    
    # # C = MultiEtaPenalizedCellConvergenceData({3: A, 2: B})
    
    # # save_data(C, save_dir=".")
    
    # # data_1 = read_data("multi_eta_convergence.csv", data_type="multi-eta penalized cell")
    
    # data_2 = read_data("multi_eta_convergence.csv", data_type="multi-eta penalized cell")
    # # plot_tensor_convergence(data_2)
    # plot_corrector_convergence(data_2, 
    #                         #    save_name="caca.pdf"
    #                            )
    # plt.show()
    
    
    def generate_test_data() -> MultiEtaPenalizedCellConvergenceData:
        h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
        n_nodes = [25, 100, 400, 1600, 6400]
        eta_values = [1e-1, 1e-2, 1e-3, 1e-4]
        
        test_data = {}
        for eta in eta_values:
            # Simuler des taux de convergence quadratique/linéaire avec bruit
            base_rate = np.array(h_values) 
            l2_rate = base_rate**2 * (1 + 0.1*np.random.randn(len(h_values)))
            h1_rate = base_rate * (1 + 0.1*np.random.randn(len(h_values)))
            
            # Effet de η sur l'erreur
            eta_factor = np.sqrt(eta)
            
            test_data[eta] = PenalizedCellConvergenceData(
                eta=eta,
                h=h_values,
                n_nodes=n_nodes,
                l2_errors_corrector1=l2_rate * eta_factor,
                h1_errors_corrector1=h1_rate * eta_factor,
                l2_errors_corrector2=l2_rate * eta_factor * 1.2,  # Légèrement différent pour C2
                h1_errors_corrector2=h1_rate * eta_factor * 1.2,
                tensor_errors=l2_rate * eta * 0.5  # Erreur tenseur proportionnelle à η
            )
        
        return MultiEtaPenalizedCellConvergenceData(data=test_data)

    # Test
    test_data = generate_test_data()
    plot_corrector_convergence(test_data, 
                               save_name="test_correctors.pdf"
                               )
    plt.show()
    plot_tensor_convergence(test_data, 
                            save_name="test_tensor.pdf"
                            )
    plt.show()
    # print(BaseConvergenceData.__annotations__)
    # print(StandardConvergenceData.__annotations__)
    # print(PenalizedConvergenceData.__annotations__)
    #  dir(), var(), and getattr()