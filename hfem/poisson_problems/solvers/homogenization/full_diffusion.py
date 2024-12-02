from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Union
from hfem.core.io import FEMDataManager, Solution, MeshData
from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
from hfem.poisson_problems.configs import *
from hfem.poisson_problems.solvers import *
from scipy.interpolate import LinearNDInterpolator
from hfem.core.aliases import ScalarField, TensorField
from functools import cached_property, lru_cache

@dataclass(frozen=True)
class HomogenizationConfig:
    """Configuration for homogenization analysis"""
    A: TensorField
    A_epsilon: TensorField
    right_hand_side: ScalarField
    epsilon: float
    mesh_size: float
    L_x: float = 2.0
    L_y: float = 2.0
    eta: float = 1e-5
    save_prefix: str = "homogenization_analysis"
    
    def to_dict(self):
        return {
            'L_x': self.L_x,
            'L_y': self.L_y,
            'mesh_size': self.mesh_size,
            'epsilon': self.epsilon,
            'eta': self.eta
        }

class HomogenizationAnalysis:
    """Class for performing homogenization analysis with efficient caching of solutions."""
    
    def __init__(self, config: HomogenizationConfig):
        """Initialize the analysis with given configuration."""
        self.config = config
        self.manager = FEMDataManager()
        self._ensure_mesh_directories()
        
    @staticmethod
    def _ensure_mesh_directories():
        """Ensure required directories exist."""
        for dir_name in ['meshes', 'simulation_data/cell', 'simulation_data/homogenized', 'simulation_data/diffusion']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    @cached_property
    def problem_mesh(self):
        """Lazily create or load problem mesh."""
        mesh_path = Path(f"meshes/rectangle_{self.config.mesh_size}.msh")
        if not mesh_path.is_file():
            rectangular_mesh(
                h=self.config.mesh_size,
                L_x=self.config.L_x,
                L_y=self.config.L_y,
                save_name=str(mesh_path)
            )
        return CustomTwoDimensionMesh(filename=str(mesh_path))

    @cached_property
    def cell_mesh(self):
        """Lazily create or load cell mesh."""
        mesh_path = Path(f"meshes/periodicity_cell_{self.config.mesh_size}.msh")
        if not mesh_path.is_file():
            rectangular_mesh(
                h=self.config.mesh_size,
                L_x=1,
                L_y=1,
                save_name=str(mesh_path)
            )
        return CustomTwoDimensionMesh(filename=str(mesh_path))

    def _get_cell_solution(self, force_recompute: bool = False) -> tuple:
        """Get cell problem solution, using cache if possible."""
        save_file = f"{self.config.save_prefix}_cell_h_{self.config.mesh_size}"
        cell_problem_save_file = f"simulation_data/cell/{save_file}.h5"
        
        if not force_recompute and Path(cell_problem_save_file).is_file():
            solution, mesh, _ = self.manager.load(cell_problem_save_file)
            if (solution.metadata['problem_params']['eta'] == self.config.eta and 
                solution.metadata['problem_params']['mesh_size'] == self.config.mesh_size):
                return solution, mesh

        print("Computing cell problems...")
        cell_problem_config = CellProblemConfig(
            mesh=self.cell_mesh,
            mesh_size=self.config.mesh_size,
            diffusion_tensor=self.config.A,
            eta=self.config.eta
        )
        cell_problem = CellProblem(config=cell_problem_config)
        cell_problem.solve_and_save(save_name=save_file)
        
        return self.manager.load(cell_problem_save_file)[:2]

    # @lru_cache
    def _get_interpolators(self, cell_solution, cell_mesh):
        """Cache interpolators for performance."""
        return [
            LinearNDInterpolator(
                cell_mesh.nodes,
                cell_solution.data[f'corrector_{ax}'],
                fill_value=0
            ) for ax in ['x', 'y']
        ]

    def analyze(self, force_recompute: bool = False) -> dict:
        """Run complete homogenization analysis."""
        # Get solutions
        cell_solution, cell_mesh = self._get_cell_solution(force_recompute)
        homogenized_solution, _, homogenized_matrices = self._solve_homogenized_problem(cell_solution)
        diffusion_solution, diffusion_mesh, _ = self._solve_diffusion_problem()
        
        # Compute solutions and errors
        interpolated_correctors = self._compute_correctors(cell_solution, cell_mesh, diffusion_mesh)
        u_1 = (homogenized_solution.data['x_derivative']*interpolated_correctors[0] + 
               homogenized_solution.data['y_derivative']*interpolated_correctors[1])
        u_0 = homogenized_solution.data['solution']
        u_epsilon = diffusion_solution.data
        
        errors = self._compute_errors(u_epsilon, u_0, u_1, homogenized_matrices)
        results = {
            'solutions': {
                'u_epsilon': u_epsilon,
                'u_0': u_0,
                'u_1': u_1
            },
            'errors': {
                'l2': errors[0],
                'h1': errors[1],
                'h1_corrected': errors[2]
            },
            'mesh': diffusion_mesh
        }
        
        self._save_analysis(results)
        return results

    def _solve_homogenized_problem(self, cell_solution) -> tuple:
        """Solve homogenized problem with caching."""
        save_file = f"{self.config.save_prefix}_homogenized_eps_{self.config.epsilon}"
        save_path = Path(f"simulation_data/homogenized/{save_file}.h5")
        
        if save_path.is_file():
            return self.manager.load(str(save_path))
        
        homogenized_config = HomogenizedConfig(
            mesh=self.problem_mesh,
            mesh_size=self.config.mesh_size,
            effective_tensor=cell_solution.data['homogenized_tensor'],
            right_hand_side=self.config.right_hand_side
        )
        homogenized_problem = HomogenizedProblem(config=homogenized_config)
        homogenized_problem.solve_and_save(save_name=save_file)
        return self.manager.load(str(save_path))

    def _solve_diffusion_problem(self) -> tuple:
        """Solve diffusion problem with caching."""
        save_file = f"{self.config.save_prefix}_diffusion_eps_{self.config.epsilon}"
        save_path = Path(f"simulation_data/diffusion/{save_file}.h5")
        
        if save_path.is_file():
            return self.manager.load(str(save_path))
            
        diffusion_config = DiffusionProblemConfig(
            mesh=self.problem_mesh,
            mesh_size=self.config.mesh_size,
            epsilon=self.config.epsilon,
            diffusion_tensor=self.config.A_epsilon,
            right_hand_side=self.config.right_hand_side
        )
        diffusion_problem = DiffusionProblem(config=diffusion_config)
        diffusion_problem.solve_and_save(save_file)
        return self.manager.load(str(save_path))
    
    def _compute_correctors(self, cell_solution, cell_mesh, diffusion_mesh):
        """Compute correctors through interpolation."""
        interpolators = self._get_interpolators(cell_solution, cell_mesh)
        
        X, Y = self._get_cell_coordinates(*diffusion_mesh.nodes.T)
        points = np.column_stack([X.ravel(), Y.ravel()])
        return [interpolator(points).reshape(X.shape) for interpolator in interpolators]

    def _get_cell_coordinates(self, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute coordinates in the unit cell ]0,1[²."""
        y1 = (x1/self.config.epsilon) % 1
        y2 = (x2/self.config.epsilon) % 1
        return y1, y2
    
    def _compute_errors(self, u_epsilon, u_0, u_1, matrices):
        """Compute normalized L2 and H1 errors."""
        M, K = matrices.mass_matrix, matrices.stiffness_matrix
        
        l2_error = np.sqrt((u_epsilon - u_0).T @ M @ (u_epsilon - u_0))/np.sqrt(u_0.T @ M @ u_0)
        h1_error = np.sqrt((u_epsilon - u_0).T @ K @ (u_epsilon - u_0))/np.sqrt(u_0.T @ K @ u_0)
        h1_error_corrected = np.sqrt(
            (u_epsilon - u_0 - self.config.epsilon*u_1).T @ 
            K @ 
            (u_epsilon - u_0 - self.config.epsilon*u_1)
        )/np.sqrt((u_0 + self.config.epsilon*u_1).T @ K @ (u_0 + self.config.epsilon*u_1))
        
        return l2_error, h1_error, h1_error_corrected
    
    def _save_analysis(self, results: dict) -> None:
        """Save analysis results."""
        solution = Solution(
            data={
                'u_epsilon': results['solutions']['u_epsilon'],
                'u_0': results['solutions']['u_0'],
                'u_1': results['solutions']['u_1']
            },
            problem_type="homogenization_analysis",
            metadata={
                'mesh_size': self.config.mesh_size,
                'epsilon': self.config.epsilon,
                'errors': {
                    'l2': float(results['errors']['l2']),
                    'h1': float(results['errors']['h1']),
                    'h1_corrected': float(results['errors']['h1_corrected'])
                },
                'problem_params': self.config.to_dict()
            }
        )
        
        self.manager.save(
            name=f"{self.config.save_prefix}_eps_{self.config.epsilon}_analysis",
            solution=solution,
            mesh=results['mesh'],
            matrices=None
        )