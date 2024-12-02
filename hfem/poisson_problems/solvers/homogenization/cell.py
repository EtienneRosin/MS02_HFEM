from hfem.poisson_problems.solvers.base import PoissonProblem
from hfem.core.related_matrices import assemble_P_per
from hfem.poisson_problems.configs.homogenization.cell import CellProblemConfig
from hfem.core.io import Solution, FEMDataManager, MeshData, FEMMatrices

import scipy.sparse as sparse
import numpy as np
from pathlib import Path
from typing import Union, Tuple

class CellProblem(PoissonProblem):
    def __init__(self, config: CellProblemConfig):
        super().__init__(config)
        self.corrector_x = None
        self.corrector_y = None
        self.homogenized_tensor = None

    def _compute_rhs(self, direction: str) -> np.ndarray:
        """Calcul du second membre pour le problème de cellule."""
        if direction not in ['x', 'y']:
            raise ValueError("direction must be either 'x' or 'y'")
            
        # On retourne -div(A * e_i) où e_i est le vecteur de base canonique
        return -self.stiffness_matrix @ self.config.mesh.node_coords.T[
            0 if direction == 'x' else 1
        ]
    
    def _solve_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Résolution des deux problèmes de cellule."""
        P_per = assemble_P_per(mesh=self.config.mesh)
        A_per = P_per @ (self.stiffness_matrix + self.config.eta*self.mass_matrix) @ P_per.T
        
        # Résoudre pour les deux directions
        corrector_x = P_per.T @ sparse.linalg.spsolve(
            A_per, 
            P_per @ self._compute_rhs('x')
        )
        corrector_y = P_per.T @ sparse.linalg.spsolve(
            A_per, 
            P_per @ self._compute_rhs('y')
        )
        
        return corrector_x, corrector_y
        
    def compute_homogenized_tensor(self) -> np.ndarray:
        """Calcule le tenseur homogénéisé à partir des correcteurs."""
        if self.corrector_x is None or self.corrector_y is None:
            raise ValueError("Need to solve the cell problems first")
        
        correctors = [self.corrector_x, self.corrector_y]
        A_eff = np.zeros((2, 2))
        
        # Calculer les composantes du tenseur homogénéisé
        for i in range(2):
            for j in range(2):
                base_i = self.config.mesh.node_coords[:, i]
                corrector_i = correctors[i]
                base_j = self.config.mesh.node_coords[:, j]
                corrector_j = correctors[j]
                
                A_eff[i, j] = np.dot(
                    self.stiffness_matrix @ (base_j + corrector_j),
                    base_i + corrector_i
                )
        return A_eff
        
    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Résout les problèmes de cellule et sauvegarde les résultats."""
        self.assemble_system()
        self.corrector_x, self.corrector_y = self._solve_system()
        self.homogenized_tensor = self.compute_homogenized_tensor()
        self._save_solution(save_name)
    
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Sauvegarde les correcteurs et le tenseur homogénéisé."""
        solution = Solution(
            data={
                'corrector_x': self.corrector_x,
                'corrector_y': self.corrector_y,
                'homogenized_tensor': self.homogenized_tensor
            },
            problem_type=str(self.config.problem_type),
            metadata={
                'mesh_size': self.config.mesh_size,
                'problem_params': self.config.to_dict()
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
        print("on a save solution cell")