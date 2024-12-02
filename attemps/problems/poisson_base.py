"""
Base classes for Poisson problems.

This module provides abstract base classes for solving different types of Poisson problems:
- Neumann boundary conditions
- Homogeneous Dirichlet boundary conditions 
- Periodic boundary conditions
- Homogenization
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum


from hfem.core import QuadratureFactory, QuadratureRule, BarycentricTransformation
from hfem.problems import BasePoissonConfig
from hfem.viz import FEMVisualizer, solution_config, error_config, ErrorType, VisualizationConfig
from hfem.mesh_manager import CustomTwoDimensionMesh





class BasePoissonProblem(ABC):
    """Abstract base solver for all Poisson problems."""
    
    def __init__(self, 
                 mesh: CustomTwoDimensionMesh, 
                 config: BasePoissonConfig):
        """
        Initialize the solver.
        
        Parameters
        ----------
        mesh : CustomTwoDimensionMesh
            Triangular mesh of the domain
        config : BasePoissonConfig
            Problem configuration
        """
        self.mesh = mesh
        self.config = config
        self.solution = None
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.l2_error = None
        self.h1_error = None
        
        # Get quadrature rule
        self.quadrature = config.quadrature_strategy.get_rule()

    def _build_elementary_mass_matrix(self, triangle: np.ndarray) -> np.ndarray:
        """
        Construct elementary mass matrix for a triangle.
        
        Parameters
        ----------
        triangle : np.ndarray
            Triangle vertex indices
            
        Returns
        -------
        np.ndarray
            3x3 elementary mass matrix
        """
        nodes = self.mesh.node_coords[triangle]
        det_j = np.abs(np.linalg.det(BarycentricTransformation.compute_jacobian(nodes)))
        mass_matrix = np.ones((3, 3)) * det_j / 24
        np.fill_diagonal(mass_matrix, mass_matrix[0,0] * 2)
        return mass_matrix

    def _build_elementary_stiffness_matrix(self, triangle: np.ndarray) -> np.ndarray:
        """
        Construct elementary stiffness matrix using quadrature.
        
        Parameters
        ----------
        triangle : np.ndarray
            Triangle vertex indices
            
        Returns
        -------
        np.ndarray
            3x3 elementary stiffness matrix
        """
        nodes = self.mesh.node_coords[triangle]
        jacobian = BarycentricTransformation.compute_jacobian(nodes)
        inv_jac = np.linalg.inv(jacobian).T
        det_j = np.abs(np.linalg.det(jacobian))

        stiffness_matrix = np.zeros((3, 3))
        
        for w_q, x_q in zip(self.quadrature.weights, self.quadrature.points):
            point = np.dot(jacobian, x_q) + nodes[0]
            A_local = self.config.diffusion_tensor(*point)
            
            for i in range(3):
                for j in range(3):
                    grad_ref_i = BarycentricTransformation.compute_reference_gradient(i)
                    grad_ref_j = BarycentricTransformation.compute_reference_gradient(j)
                    
                    grad_i = inv_jac @ grad_ref_i
                    grad_j = inv_jac @ grad_ref_j
                    
                    stiffness_matrix[i, j] += w_q * np.dot(A_local @ grad_i, grad_j)
        
        return stiffness_matrix * det_j

    def assemble_system(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Assemble global system matrix and RHS.
        
        Returns
        -------
        Tuple[sp.csr_matrix, np.ndarray]
            System matrix and right-hand side vector
        """
        n = self.mesh.num_nodes
        mass_matrix = sp.lil_matrix((n, n), dtype=float)
        stiffness_matrix = sp.lil_matrix((n, n), dtype=float)
        
        # Assemble matrices
        for triangle in tqdm(self.mesh.tri_nodes, desc="Assembling matrices"):
            mass_elem = self._build_elementary_mass_matrix(triangle)
            stiffness_elem = self._build_elementary_stiffness_matrix(triangle)
            
            for i in range(3):
                for j in range(3):
                    I, J = triangle[i], triangle[j]
                    mass_matrix[I, J] += mass_elem[i, j]
                    stiffness_matrix[I, J] += stiffness_elem[i, j]
        
        self.mass_matrix = mass_matrix.tocsr()
        self.stiffness_matrix = stiffness_matrix.tocsr()
        
        # Compute RHS
        rhs = self.mass_matrix @ self.config.right_hand_side(*self.mesh.node_coords.T)
        
        return self.mass_matrix + self.stiffness_matrix, rhs

    # @abstractmethod
    # def _apply_boundary_conditions(self, 
    #                              matrix: sp.csr_matrix, 
    #                              rhs: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray]:
    #     """
    #     Apply boundary conditions to system.
        
    #     Parameters
    #     ----------
    #     matrix : sp.csr_matrix
    #         System matrix
    #     rhs : np.ndarray
    #         Right-hand side vector
            
    #     Returns
    #     -------
    #     Tuple[sp.csr_matrix, np.ndarray]
    #         Modified system matrix and RHS
    #     """
    #     pass

    def solve(self) -> np.ndarray:
        """
        Solve the Poisson problem.
        
        Returns
        -------
        np.ndarray
            Solution vector
        """
        # Assemble system
        matrix, rhs = self.assemble_system()
        
        # Apply boundary conditions
        # matrix, rhs = self._apply_boundary_conditions(matrix, rhs)
        
        # Solve system
        self.solution = spla.spsolve(matrix, rhs)
        
        # Compute errors if exact solution available
        if self.config.exact_solution is not None:
             self._compute_errors()
        
        return self.solution

    def _compute_errors(self) -> Tuple[float, float]:
        """
        Compute L2 and H1 errors.
        
        Returns
        -------
        Tuple[float, float]
            L2 error and H1 error
        """
        exact = self.config.exact_solution(*self.mesh.node_coords.T)
        error = exact - self.solution
        
        # L2 error
        l2_error = np.sqrt(error.T @ self.mass_matrix @ error) / \
                   np.sqrt(exact.T @ self.mass_matrix @ exact)
        
        # H1 error
        h1_error = np.sqrt(error.T @ self.stiffness_matrix @ error) / \
                   np.sqrt(exact.T @ self.stiffness_matrix @ exact)
        self.l2_error, self.h1_error = l2_error, h1_error
        print(f"L2 Error: {l2_error:.3e}, H1 Error: {h1_error:.3e}")
        # print(f"")
        
        return l2_error, h1_error
    
    def display_solution(self, config: Optional[VisualizationConfig] = None, **kwargs):
        """Display the computed solution."""
        if self.solution is None:
            self.solve()
            
        if config is None:
            config = solution_config(**kwargs)
        
        visualizer = FEMVisualizer(self.mesh.node_coords, self.mesh.tri_nodes)
        return visualizer.plot_solution(self.solution, config)

    def display_error(self, config: Optional[VisualizationConfig] = None, **kwargs):
        """Display error between computed and exact solutions."""
        if self.solution is None:
            self.solve()
        
        if self.config.exact_solution is None:
            raise ValueError("No exact solution available for error computation")
        
        if config is None:
            config = error_config(**kwargs)
        
        exact = self.config.exact_solution(*self.mesh.node_coords.T)
        
        visualizer = FEMVisualizer(self.mesh.node_coords, self.mesh.tri_nodes)
        return visualizer.plot_error(self.solution, exact, config)
    
    
    
# class SimplePoissonSolver(BasePoissonProblem):
#     """Simple concrete implementation of BasePoissonProblem for testing."""
    
#     def _apply_boundary_conditions(self, matrix, rhs):
#         """No boundary conditions for this test."""
#         return matrix, rhs

def test_poisson_solver():
    """Test the Poisson solver with a simple case."""
    
    # Create a test mesh
    mesh_file = "test_mesh.msh"
    from mesh_manager.geometries import rectangular_mesh
    rectangular_mesh(h=0.01, L_x=1, L_y=1, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    a = 2
    def v(x, y):
        return np.sin(a * np.pi * x) * np.sin(a * np.pi * y) + 2
    
    def diffusion_tensor(x, y):
        return np.eye(2)*v(x,y)

    def exact_solution(x, y):
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y)

    def right_hand_side(x, y):
        return (1 + (16*np.pi**2)*(v(x,y) -1))*exact_solution(x,y)
    
    
    
    # Create configuration
    config = BasePoissonConfig(
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side,
        exact_solution=exact_solution
    )
    
    # Create and solve
    solver = BasePoissonProblem(mesh, config)
    solution = solver.solve()
    
    solver.display_solution(
        solution_config(
            kind='trisurface',
            # title='Numerical Solution',
            # cmap='viridis',
            # cbar=True,
            save_name = 'hihi.pdf'
        )
    )
    plt.show()
    # Visualize error
    solver.display_error(
        error_config(
            kind='contourf',
            error_type=ErrorType.ABSOLUTE,
            cbar = True
            # title='Absolute Error',
            # cmap='RdBu_r'
        )
    )
    plt.show()
    
    solver.display_error(
        error_config(
            kind='trisurface',
            error_type=ErrorType.ABSOLUTE,
            # title='Absolute Error',
            # cmap='RdBu_r',
            cbar = False
        )
    )
    plt.show()
    return solver

if __name__ == "__main__":
    solver = test_poisson_solver()