"""
Module for finite element convergence analysis.

This module provides tools to measure, analyze, and visualize convergence rates
of finite element solutions with different boundary conditions. It supports:
- Convergence measurement for different mesh sizes
- Data storage and retrieval
- Convergence rate analysis
- Visualization with theoretical convergence rates
"""

from typing import Union, List, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from hfem.problems import BasePoissonConfig, BasePoissonProblem, PenalizedCellProblemConfig, PenalizedCellProblems
from mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh
from hfem.viz.conditional_style_context import conditional_style_context


@dataclass
class ConvergenceData:
    """
    Storage class for finite element convergence data.
    
    Attributes
    ----------
    h : list[float]
        Mesh sizes
    n_nodes : list[int]
        Number of nodes for each mesh
    l2_errors : list[float]
        L² norm relative errors
    h1_errors : list[float]
        H¹ norm relative errors
    boundary_type : str
        Boundary condition type ('Dirichlet', 'Neumann', 'Periodic')
    """
    h: list[float]
    n_nodes: list[int]
    l2_errors: list[float]
    h1_errors: list[float]
    boundary_type: str


def measure_convergence(
    problem: BasePoissonProblem,
    problem_config: BasePoissonConfig,
    mesh_generator: callable,
    mesh_sizes: List[float],
    mesh_config: Optional[dict] = None
) -> ConvergenceData:
    """
    Measure finite element convergence across multiple mesh sizes.
    
    Parameters
    ----------
    problem : BasePoissonProblem
        Problem class (not instance) to solve
    problem_config : BasePoissonConfig
        Problem configuration
    mesh_generator : callable
        Function to generate meshes
    mesh_sizes : List[float]
        Mesh sizes to test
    mesh_config : dict, optional
        Additional mesh generation parameters
        
    Returns
    -------
    ConvergenceData
        Measured convergence data
    """
    if mesh_config is None:
        mesh_config = {}
    
    h_values = []
    n_nodes_values = []
    l2_errors = []
    h1_errors = []
    
    for h in mesh_sizes:
        mesh_file = f"meshes/temp_mesh_{h:.6f}.msh"
        mesh_generator(h=h, save_name=mesh_file, **mesh_config)
        
        try:
            mesh = CustomTwoDimensionMesh(mesh_file)
            solver = problem(mesh, problem_config)
            solver.solve()
            l2_error, h1_error = solver.l2_error, solver.h1_error
            
            h_values.append(h)
            n_nodes_values.append(mesh.num_nodes)
            l2_errors.append(l2_error)
            h1_errors.append(h1_error)
            
        finally:
            Path(mesh_file).unlink(missing_ok=True)
    
    boundary_type = problem.__name__.lower()
    if "dirichlet" in boundary_type:
        boundary_type = "Dirichlet"
    elif "neumann" in boundary_type:
        boundary_type = "Neumann"
    elif "periodic" in boundary_type:
        boundary_type = "Periodic"
    else:
        boundary_type = "Unknown"
    
    return ConvergenceData(
        h=h_values,
        n_nodes=n_nodes_values,
        l2_errors=l2_errors,
        h1_errors=h1_errors,
        boundary_type=boundary_type
    )


def save_convergence_data(
    data: ConvergenceData,
    save_dir: Union[str, Path],
    filename: Optional[str] = None
) -> Path:
    """
    Save convergence data to CSV.
    
    Parameters
    ----------
    data : ConvergenceData
        Data to save
    save_dir : str or Path
        Save directory
    filename : str, optional
        Custom filename (default: 'convergence_{boundary_type}.csv')
        
    Returns
    -------
    Path
        Path to saved file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"convergence_{data.boundary_type}.csv"
    
    save_path = save_dir / filename
    df = pd.DataFrame({
        'h': data.h,
        'n_nodes': data.n_nodes,
        'L2_error': data.l2_errors,
        'H1_error': data.h1_errors
    })
    df.to_csv(save_path, index=False)
    
    return save_path


def read_from_file(
    filepath: Union[str, Path],
    boundary_type: Optional[str] = None
) -> ConvergenceData:
    """
    Read convergence data from CSV.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    boundary_type : str, optional
        Override boundary condition type
        
    Returns
    -------
    ConvergenceData
        Loaded convergence data
    
    Raises
    ------
    ValueError
        If required columns are missing
    FileNotFoundError
        If file doesn't exist
    """
    filepath = Path(filepath)
    
    if boundary_type is None:
        filename = filepath.stem
        if 'dirichlet' in filename.lower():
            boundary_type = 'Dirichlet'
        elif 'neumann' in filename.lower():
            boundary_type = 'Neumann'
        elif 'periodic' in filename.lower():
            boundary_type = 'Periodic'
        else:
            boundary_type = 'Unknown'
    
    try:
        df = pd.read_csv(filepath)
        required_columns = {'h', 'n_nodes', 'L2_error', 'H1_error'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        return ConvergenceData(
            h=df['h'].tolist(),
            n_nodes=df['n_nodes'].tolist(),
            l2_errors=df['L2_error'].tolist(),
            h1_errors=df['H1_error'].tolist(),
            boundary_type=boundary_type
        )
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


@conditional_style_context()
def plot_convergence(
    data: Union[ConvergenceData, List[ConvergenceData]],
    save_name: Optional[str] = None,
    show: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot convergence analysis with theoretical rates.
    
    Creates a log-log plot showing numerical errors and their theoretical
    convergence rates. L² errors should converge at rate 2, and H¹ errors
    at rate 1, indicated by triangles on the plot.
    
    Parameters
    ----------
    data : ConvergenceData or List[ConvergenceData]
        Convergence data to plot
    save_name : str, optional
        Base name for saved plots (.pdf and .svg extensions added)
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects for customization
    """
    if not isinstance(data, list):
        data = [data]
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    
    line_props = dict(marker="o", linewidth=1, linestyle="--")
    annotation_props = dict(fontsize=15, weight='extra bold')
    
    for d in data:
        log_h = np.log(1/np.array(d.h))
        log_l2 = np.log(np.array(d.l2_errors))
        log_h1 = np.log(np.array(d.h1_errors))
        
        suffix = f" ({d.boundary_type})"
        
        # Plot error curves
        ax.plot(log_h, log_l2, 
                label=r'$L^2\left(\Omega\right)$' + f' error{suffix}',
                c="#D1453D", **line_props)
        ax.plot(log_h, log_h1,
                label=r'$H^1\left(\Omega\right)$' + f' error{suffix}',
                c="#5B9276", **line_props, markersize=3)
        
        # Add L² rate triangle
        x_max, y_min = log_h.max(), log_l2.min()
        l2_tri_x = np.array([x_max-1, x_max, x_max-1])
        l2_tri_y = np.array([y_min, y_min, y_min+2])
        
        ax.add_patch(plt.Polygon(
            np.column_stack((l2_tri_x, l2_tri_y)),
            facecolor='#D1453D', alpha=0.3,
            label=r'$L^2$ Expected rate'
        ))
        
        ax.annotate(r'$\boldsymbol{2}$',
            xy=((l2_tri_x[-1] + l2_tri_x[0])/2, 
                (l2_tri_y[-1] + l2_tri_y[0])/2),
            xytext=(4, -4), textcoords='offset points',
            va='center', ha='left', color="#D1453D",
            **annotation_props
        )
        
        # Add H¹ rate triangle
        x_min, y_max = log_h.min(), log_h1.max()
        h1_tri_x = np.array([x_min, x_min + 1, x_min])
        h1_tri_y = np.array([y_max-1, y_max-1, y_max])
        
        ax.add_patch(plt.Polygon(
            np.column_stack((h1_tri_x, h1_tri_y)),
            facecolor='#5B9276', alpha=0.3,
            label=r'$H^1$ Expected rate'
        ))
        
        ax.annotate(r'$\boldsymbol{1}$',
            xy=((h1_tri_x[-1] + h1_tri_x[0])/2,
                (h1_tri_y[-1] + h1_tri_y[0])/2),
            xytext=(4, -4), textcoords='offset points',
            va='center', ha='left', color="#5B9276",
            **annotation_props
        )
    
    ax.set_xlabel(r'$\log(1/h)$')
    ax.set_ylabel(r'$\log\left(\frac{\|u_h - u\|}{\|u\|}\right)$')
    ax.legend()
    
    plt.tight_layout()
    
    if save_name is not None:
        fig.savefig(f"{save_name}.pdf")
        fig.savefig(f"{save_name}.svg")
    
    if show:
        plt.show()
    
    return fig, ax


@dataclass
class PenalizedConvergenceData:
    """Storage for penalized cell problems convergence data."""
    h: list[float]
    n_nodes: list[int]
    l2_errors_corrector1: list[float]
    h1_errors_corrector1: list[float]
    l2_errors_corrector2: list[float]
    h1_errors_corrector2: list[float]
    tensor_errors: list[float]
    eta: float

def measure_penalized_convergence(
    problem_config: PenalizedCellProblemConfig,
    mesh_generator: callable,
    mesh_sizes: List[float],
    mesh_config: Optional[dict] = None
) -> PenalizedConvergenceData:
    """Measure convergence for penalized cell problems."""
    if mesh_config is None:
        mesh_config = {}
    
    data = {
        'h': [], 'n_nodes': [],
        'l2_c1': [], 'h1_c1': [],
        'l2_c2': [], 'h1_c2': [],
        'tensor': []
    }
    
    for h in mesh_sizes:
        mesh_file = f"meshes/temp_mesh_{h:.6f}.msh"
        mesh_generator(h=h, save_name=mesh_file, **mesh_config)
        
        try:
            mesh = CustomTwoDimensionMesh(mesh_file)
            solver = PenalizedCellProblems(mesh, problem_config)
            solver.solve()
            
            data['h'].append(h)
            data['n_nodes'].append(mesh.num_nodes)
            data['l2_c1'].append(solver.l2_errors[0])
            data['h1_c1'].append(solver.h1_errors[0])
            data['l2_c2'].append(solver.l2_errors[1])
            data['h1_c2'].append(solver.h1_errors[1])
            
            tensor_error = np.linalg.norm(
                solver.homogenized_tensor - problem_config.exact_homogenized_tensor
            ) / np.linalg.norm(problem_config.exact_homogenized_tensor)
            data['tensor'].append(tensor_error)
            
        finally:
            Path(mesh_file).unlink(missing_ok=True)
    
    return PenalizedConvergenceData(
        h=data['h'],
        n_nodes=data['n_nodes'],
        l2_errors_corrector1=data['l2_c1'],
        h1_errors_corrector1=data['h1_c1'],
        l2_errors_corrector2=data['l2_c2'],
        h1_errors_corrector2=data['h1_c2'],
        tensor_errors=data['tensor'],
        eta=problem_config.eta
    )

@conditional_style_context()
def plot_penalized_convergence(
    data: PenalizedConvergenceData,
    save_name: Optional[str] = None,
    show: bool = True
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot convergence for penalized cell problems."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    line_props = dict(marker="o", linewidth=1, linestyle="--")
    annotation_props = dict(fontsize=12, weight='bold')
    
    log_h = np.log(1/np.array(data.h))
    
    # Correctors plot
    for i, (l2_err, h1_err, label) in enumerate([
        (data.l2_errors_corrector1, data.h1_errors_corrector1, "Corrector 1"),
        (data.l2_errors_corrector2, data.h1_errors_corrector2, "Corrector 2")
    ]):
        log_l2 = np.log(np.array(l2_err))
        log_h1 = np.log(np.array(h1_err))
        
        color = "#D1453D" if i == 0 else "#5B9276"
        ax1.plot(log_h, log_l2, label=f"L² error ({label})", c=color, **line_props)
        ax1.plot(log_h, log_h1, label=f"H¹ error ({label})", c=color, **line_props, markersize=3)
    
    # Add rate triangles (similar to original plot_convergence)
    ax1.set_xlabel(r'$\log(1/h)$')
    ax1.set_ylabel(r'$\log\left(\frac{\|u_h - u\|}{\|u\|}\right)$')
    ax1.legend()
    
    # Tensor convergence plot
    log_tensor = np.log(np.array(data.tensor_errors))
    ax2.plot(log_h, log_tensor, label="Tensor error", c="#1A5F7A", **line_props)
    ax2.set_xlabel(r'$\log(1/h)$')
    ax2.set_ylabel(r'$\log\left(\frac{\|A^* - A^*_\eta\|}{\|A^*\|}\right)$')
    ax2.legend()
    
    plt.suptitle(f"Convergence Analysis (η = {data.eta})")
    plt.tight_layout()
    
    if save_name is not None:
        fig.savefig(f"{save_name}.pdf")
        fig.savefig(f"{save_name}.svg")
    
    if show:
        plt.show()
    
    return fig, [ax1, ax2]

if __name__ == '__main__':
    # Example usage
    pass
    # from hfem.problems import HomogenousDirichletPoissonProblem
    # from mesh_manager.geometries import rectangular_mesh
    
    # def v(x, y):
    #     return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 2

    # def diffusion_tensor(x, y):
    #     return v(x, y) * np.eye(2)

    # def exact_solution(x, y):
    #     return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    # def right_hand_side(x, y): 
    #     return (1 + 16*(np.pi**2)*(v(x,y) - 1))*exact_solution(x,y)
    
    # # Create problem configuration
    # problem_config = BasePoissonConfig(
    #     diffusion_tensor=diffusion_tensor,
    #     right_hand_side=right_hand_side,
    #     exact_solution=exact_solution
    # )
    
    # # mesh_sizes = [0.25, 0.125, 0.075, 0.05, 0.025, 0.0125, 0.0075, 0.005]
    # # mesh_sizes = [0.25, 0.125, 0.075, 0.05, 0.025, 0.0125]
    # mesh_sizes = [0.25, 0.125, 0.075, 0.05, 0.025]
    # mesh_config = {'L_x': 1.0, 'L_y': 1.0}
    
    # # Run convergence study
    # data = measure_convergence(
    #     problem=HomogenousDirichletPoissonProblem,
    #     problem_config=problem_config,
    #     mesh_generator=rectangular_mesh,
    #     mesh_sizes=mesh_sizes,
    #     mesh_config=mesh_config
    # ) 
    
    # # Save data
    # save_dir = Path("convergence_results")
    # save_convergence_data(data, save_dir)
    
    # # # Analyze and plot
    # # analysis = analyze_convergence(data)
    # # print("\nConvergence analysis:")
    # # print(analysis)
    
    # data = read_from_file(filepath=f"{save_dir}/convergence_dirichlet.csv")
    
    # # plot_convergence(data, save_dir)
    # plot_convergence(data, save_name='ububobuibygv')
    # # plot_convergence(data, save_dir)