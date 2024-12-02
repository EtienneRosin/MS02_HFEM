from hfem.core import BarycentricTransformation
from hfem.mesh_manager import CustomTwoDimensionMesh
from hfem.viz import FEMVisualizer, solution_config, error_config, ErrorType, VisualizationConfig
from hfem.problems import PenalizedCellProblemConfig

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from dataclasses import replace



from hfem.core import BarycentricTransformation
from hfem.core.related_matrices import assemble_P_per
from hfem.viz import FEMVisualizer, solution_config, error_config, ErrorType, VisualizationConfig
from hfem.problems import PenalizedCellProblemConfig

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, List
from dataclasses import replace
from scipy.interpolate import LinearNDInterpolator
    

class PenalizedCellProblems:
    def __init__(self, 
                 mesh: CustomTwoDimensionMesh, 
                 config: PenalizedCellProblemConfig):
        """
        Initialize the solver.
        
        Parameters
        ----------
        mesh : CustomTwoDimensionMesh
            Triangular mesh of the domain
        config : PenalizedCellProblemConfig
            Problem configuration
        """
        self.mesh = mesh
        self.config = config
        self.correctors = None
        self.homogenized_tensor = None
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.l2_errors = None
        self.h1_errors = None
        self.homogenized_tensor_error = None
        
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

    def assemble_system(self) -> Tuple[sp.sparse.csr_matrix, np.ndarray]:
        """
        Assemble global system matrix and RHS.
        
        Returns
        -------
        Tuple[sp.csr_matrix, np.ndarray]
            System matrix and right-hand side vector
        """
        n = self.mesh.num_nodes
        mass_matrix = sp.sparse.lil_matrix((n, n), dtype=float)
        stiffness_matrix = sp.sparse.lil_matrix((n, n), dtype=float)
        
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
        rhs = [-self.stiffness_matrix @ coordinate for coordinate in self.mesh.node_coords.T]
        # rhs = [self.stiffness_matrix @ coordinate for coordinate in self.mesh.node_coords.T]
        return self.stiffness_matrix + self.config.eta * self.mass_matrix, rhs

    def solve(self) -> np.ndarray:
        """
        Solve the penalized cell problems.
        
        Returns
        -------
        np.ndarray
            Solution vector
        """
        # Assemble system
        A, L = self.assemble_system()
        P_per = assemble_P_per(self.mesh)
        A_per = P_per @ A @ P_per.T
        
        
        try:
            # U_per = [sp.sparse.linalg.spsolve(A_per, P_per @ L_i) for L_i in L]
            # # Get back to V_h
            # self.correctors = [P_per.T @ U_i for U_i in U_per]
            self.correctors = [P_per.T @ sp.sparse.linalg.spsolve(A_per, P_per @ L_i) for L_i in L]
            
            self.homogenized_tensor = self.compute_homogenized_tensor()            
            self._compute_errors()
            
            return self.correctors
            
        except sp.linalg.LinAlgError as e:
            raise ValueError(f"Failed to solve system: {str(e)}. Try increasing eta.") from e
        
    def compute_homogenized_tensor(self) -> np.ndarray:
        if self.correctors is None:
            raise ValueError("Need to solve the cell problems first")
        
        A_eff = np.zeros((2,2))
        for j in range(2):
            for k in range(2):
                # A_eff[i,j] = (self.mesh.node_coords[:, j] + self.correctors[j]).T @ self.stiffness_matrix @ (self.mesh.node_coords[:, i] + self.correctors[i])
                # A_eff[j, k] = (self.mesh.node_coords[:, k] + self.correctors[j]).T @ self.stiffness_matrix @ (self.mesh.node_coords[:, i] + self.correctors[i])

                A_eff[j,k] = np.dot(self.stiffness_matrix @ (self.mesh.node_coords[:, k] + self.correctors[k]), self.mesh.node_coords[:, j] + self.correctors[j])
                
        return A_eff
        # return 4*A_eff

    def _compute_errors(self) -> None:
        """Compute and display errors for correctors and homogenized tensor."""
        if self.config.exact_correctors is not None:
            self.l2_errors = []
            self.h1_errors = []
            print("\n╔══ Corrector Errors ═════════════════")
            for i, (corrector, exact_corrector) in enumerate(zip(self.correctors, self.config.exact_correctors)):
                error = exact_corrector(*self.mesh.node_coords.T) - corrector
                
                
                l2_error = np.sqrt(np.dot(self.mass_matrix @ error, error))
                h1_error = np.sqrt(error.T @ self.stiffness_matrix @ error)
                
                self.l2_errors.append(l2_error)
                self.h1_errors.append(h1_error)
                
                print(f"║  Corrector {i+1}")
                print(f"║    ├ L2: {l2_error:.2e} (abs)")
                print(f"║    └ H1: {h1_error:.2e} (abs)")
            print("╚" + "═" * 35)
        if self.config.exact_homogenized_tensor is not None:
            diff = self.config.exact_homogenized_tensor - self.homogenized_tensor
            rel_error = np.linalg.norm(diff) / np.linalg.norm(self.config.exact_homogenized_tensor)
            self.homogenized_tensor_error = rel_error
            print("\n╔══ Homogenized Tensor Analysis ═════")
            print(f"║  Frobenius Error: {rel_error:.2e} (rel)")
            print("║  Component-wise differences (A* - A*η):")
            for i in range(diff.shape[0]):
                row_str = " ".join(f"{val:+.2e}" for val in diff[i])
                print(f"║    {row_str}")
            print("╚" + "═" * 35)

    def display_correctors(self, config: Optional[VisualizationConfig] = None, **kwargs) -> Tuple[plt.Figure, list[plt.Axes]]:
        """
        Display both correctors side by side.
        
        Parameters
        ----------
        config : Optional[VisualizationConfig]
            Visualization configuration. If None, uses default solution_config
        **kwargs
            Additional arguments passed to solution_config
            
        Returns
        -------
        Tuple[Figure, list[Axes]]
            Figure and axes objects
        """
        if self.correctors is None:
            raise ValueError("Need to solve the problems first")
            
        if config is None:
            config = solution_config(**kwargs)
            
        visualizer = FEMVisualizer(self.mesh.node_coords, self.mesh.tri_nodes)
        return visualizer.plot_correctors(self.correctors, config)

    def display_corrector_errors(self, config: Optional[VisualizationConfig] = None, **kwargs) -> Tuple[plt.Figure, list[plt.Axes]]:
        """
        Display error between computed and exact correctors side by side.
        
        Parameters
        ----------
        config : Optional[VisualizationConfig]
            Visualization configuration. If None, uses default error_config
        **kwargs
            Additional arguments passed to error_config
            
        Returns
        -------
        Tuple[Figure, list[Axes]]
            Figure and axes objects
        """
        if self.correctors is None or self.config.exact_correctors is None:
            raise ValueError("Need both computed and exact correctors")
            
        if config is None:
            config = error_config(**kwargs)
            
        # Get exact values at nodes
        exact_values = [
            exact_corrector(*self.mesh.node_coords.T)
            for exact_corrector in self.config.exact_correctors
        ]
            
        visualizer = FEMVisualizer(self.mesh.node_coords, self.mesh.tri_nodes)
        return visualizer.plot_corrector_errors(self.correctors, exact_values, config)

    def create_corrector_interpolators(self) -> List[LinearNDInterpolator]:
        """
        Create interpolators for both correctors.
        
        Returns
        -------
        List[LinearNDInterpolator]
            List of interpolators, one for each corrector
        """
        if self.correctors is None:
            raise ValueError("Need to solve the problems first")
        
        interpolators = []
        for corrector in self.correctors:
            interpolator = LinearNDInterpolator(
                self.mesh.node_coords,  # Points (x,y)
                corrector,              # Values at these points
                fill_value=0
            )
            interpolators.append(interpolator)
        
        return interpolators
    
    

if __name__ == '__main__':
    mesh_file = "meshes/rectangle_mesh.msh"
    h = 0.075   # mesh size
    L_x = 1     # rectangle width
    L_y = 1     # rectangle height

    # Parameters
    eta = 1e-5
        
    def diffusion_tensor(x, y):
        return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)

    # Configuration
    pb_config = PenalizedCellProblemConfig(
        eta=eta,
        diffusion_tensor=diffusion_tensor,
        exact_homogenized_tensor=np.diagflat([4*np.sqrt(3), 2*np.sqrt(15)], 0)
    )
    from hfem.mesh_manager import rectangular_mesh
    
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # Instanciate and solve
    cell_pb = PenalizedCellProblems(mesh, pb_config)
    cell_pb.solve()
    
    
    # Configure visualization
    solution_cfg = solution_config(
        kind='contourf',
        cbar=True,
        num_levels=30,
        # save_name = 'study_cases/IV_Microstructured_Periodic/II_homogenized/cell_problems/case_iv.pdf'
        # cbar_props={'pad': 0.05, 'fraction': 0.046},
    )
    fig_correctors, axes_correctors = cell_pb.display_correctors(solution_cfg)
    plt.show()
    
    interpolators = cell_pb.create_corrector_interpolators()
    
    # Exemple d'utilisation :
    def get_cell_coordinates(x1: np.ndarray, x2: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """Retourne les coordonnées dans la cellule unité ]0,1[²."""
        y1 = (x1/epsilon) % 1
        y2 = (x2/epsilon) % 1
        return y1, y2
    
    # Test sur une grille
    x = np.linspace(0, 3*L_x, 100)
    y = np.linspace(0, 3*L_y, 100)
    X, Y = np.meshgrid(x, y)
    epsilon = 3
    
    # Obtenir les coordonnées dans la cellule unité
    Y1, Y2 = get_cell_coordinates(X, Y, epsilon)
    points = np.column_stack([Y1.ravel(), Y2.ravel()])
    
    # Interpoler les correcteurs
    interpolated_correctors = [
        interpolator(points).reshape(X.shape)
        for interpolator in interpolators
    ]
    
    # Visualiser les résultats interpolés
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, corr in zip(axes, interpolated_correctors):
        im = ax.contourf(X, Y, corr, levels=30)
        plt.colorbar(im, ax=ax)
    plt.show()