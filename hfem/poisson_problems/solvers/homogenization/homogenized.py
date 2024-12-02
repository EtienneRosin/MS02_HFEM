from hfem.poisson_problems.solvers.base import PoissonProblem
from hfem.poisson_problems.configs.homogenization.homogenized import HomogenizedConfig
from hfem.core.io import Solution, FEMDataManager, MeshData, FEMMatrices
from hfem.core.related_matrices import assemble_P_0, assemble_elementary_derivatives_matrices

import scipy.sparse as sparse
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from tqdm.auto import tqdm

class HomogenizedProblem(PoissonProblem):
    def __init__(self, config: HomogenizedConfig):
        super().__init__(config)
        self.solution_derivatives = None
        
    def _compute_rhs(self) -> np.ndarray:
        """Calcule le second membre pour le problème homogénéisé."""
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
        
    def assemble_system(self) -> None:
        # D'abord on assemble les matrices standard (masse et rigidité)
        super().assemble_system()
        # Puis on assemble les matrices de dérivées
        self._assemble_derivative_matrices()
    
    def _solve_system(self) -> np.ndarray:
        """Résout le problème homogénéisé."""
        
        P_0 = assemble_P_0(mesh=self.config.mesh)
        A_0 = P_0 @ self.stiffness_matrix @ P_0.T
        L_0 = P_0 @ self._compute_rhs()
        return P_0.T @ sparse.linalg.spsolve(A_0, L_0)
        
        # return sparse.linalg.spsolve(
        #     self.stiffness_matrix,
        #     self.rhs
        # )
        
    def _assemble_derivative_matrices(self) -> None:
        # print("Assembling derivative matrices...")
        n = self.config.mesh.num_nodes
        n_elements = len(self.config.mesh.tri_nodes)
        
        # Pré-allocation des arrays
        rows = np.zeros(9 * n_elements, dtype=np.int32)
        cols = np.zeros(9 * n_elements, dtype=np.int32)
        deriv_data_1 = np.zeros(9 * n_elements)
        deriv_data_2 = np.zeros(9 * n_elements)
        
        # Assemblage séquentiel
        for idx, triangle in enumerate(tqdm(self.config.mesh.tri_nodes, leave=False)):
            nodes_coords = self.config.mesh.node_coords[triangle]
            
            # Calcul des matrices dérivées élémentaires
            G1_elem, G2_elem = assemble_elementary_derivatives_matrices(nodes_coords)
            
            # Remplissage des arrays
            start = 9 * idx
            for i in range(3):
                for j in range(3):
                    pos = start + 3 * i + j
                    rows[pos] = triangle[i]
                    cols[pos] = triangle[j]
                    deriv_data_1[pos] = G1_elem[i, j]
                    deriv_data_2[pos] = G2_elem[i, j]
        
        # Construction des matrices creuses
        self.derivative_matrices = (
            sparse.csr_matrix(
                (deriv_data_1, (rows, cols)),
                shape=(n, n)
            ),
            sparse.csr_matrix(
                (deriv_data_2, (rows, cols)),
                shape=(n, n)
            )
        )
        
    def compute_directional_derivatives(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule les dérivées directionnelles de la solution.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Les dérivées selon x et y de la solution
        """
        if self.solution is None:
            raise ValueError("La solution doit être calculée avant les dérivées")
            
        if self.derivative_matrices is None:
            raise ValueError("Les matrices de dérivées doivent être assemblées")
            
        G1, G2 = self.derivative_matrices
        # U = self.solution
        # P_0 = assemble_P_0(mesh=self.config.mesh)
        # M_0 =  P_0 @ self.mass_matrix @ P_0.T
        # return P_0.T @ sparse.linalg.spsolve(M_0, - P_0 @ G1 @ U), P_0.T @ sparse.linalg.spsolve(M_0, - P_0 @ G2 @ U)
    
        return sparse.linalg.spsolve(self.mass_matrix, -G1 @ self.solution), sparse.linalg.spsolve(self.mass_matrix, -G2 @ self.solution)
        
    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Résout le problème et calcule le gradient avant la sauvegarde."""
        self.assemble_system()
        self.solution = self._solve_system()
        # self.solution_gradient = self.compute_gradient()
        self.solution_derivatives = self.compute_directional_derivatives()
        self._save_solution(save_name)
    
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Sauvegarde la solution et son gradient."""
        solution = Solution(
            data={
                'solution': self.solution,
                'x_derivative': self.solution_derivatives[0],
                'y_derivative': self.solution_derivatives[1],
            },
            problem_type=str(self.config.problem_type),
            metadata={
                'mesh_size': self.config.mesh_size,
                'problem_params': self.config.to_dict(),
                # 'effective_tensor': self.config.effective_tensor.tolist()
            }
        )
        
        manager = FEMDataManager()
        manager.save(
            name=save_name,
            solution=solution,
            mesh=MeshData.from_mesh(self.config.mesh),
            matrices=FEMMatrices(
                mass_matrix=self.mass_matrix,
                stiffness_matrix=self.stiffness_matrix
            )
        )
        