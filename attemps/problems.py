from hfem.poisson_problems.problem_configs import (
    CorePoissonProblemsConfig, 
    PeriodicConfig, 
    NeumannConfig, 
    DirichletConfig, 
    CellProblemConfig,
    HomogenizedConfig
)
from hfem.core.io import (
    FEMDataManager, 
    FEMMatrices, 
    Solution, 
    MeshData
)

from hfem.core.related_matrices import (
    assemble_elementary_mass_matrix, 
    assemble_elementary_stiffness_matrix,
    assemble_P_0,
    assemble_P_per
)

import scipy.sparse as sparse
from tqdm import tqdm
from pathlib import Path
from typing import Union
import numpy as np
from abc import ABC, abstractmethod



class PoissonProblem(ABC):  # Classe abstraite de base
    """Classe abstraite de base pour tous les problèmes de Poisson."""
    
    def __init__(self, config: CorePoissonProblemsConfig):
        self.config = config
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.solution = None
        self.rhs = None
        
    def assemble_system(self) -> None:
        """Assemblage générique des matrices."""
        n = self.config.mesh.num_nodes
        mass_matrix = sparse.lil_matrix((n, n))
        stiffness_matrix = sparse.lil_matrix((n, n))
        
        for triangle in tqdm(self.config.mesh.tri_nodes):
            mass_elem = assemble_elementary_mass_matrix(
                self.config.mesh.node_coords[triangle]
            )
            stiff_elem = assemble_elementary_stiffness_matrix(
                triangle_nodes=self.config.mesh.node_coords[triangle],
                diffusion_tensor=self.config.diffusion_tensor
            )
            
            for i in range(3):
                for j in range(3):
                    I, J = triangle[i], triangle[j]
                    mass_matrix[I, J] += mass_elem[i, j]
                    stiffness_matrix[I, J] += stiff_elem[i, j]
        
        self.mass_matrix = mass_matrix.tocsr()
        self.stiffness_matrix = stiffness_matrix.tocsr()
        self.rhs = self._compute_rhs()
    
    @abstractmethod
    def _compute_rhs(self) -> np.ndarray:
        """À implémenter dans les classes filles."""
        pass

    @abstractmethod
    def _solve_system(self) -> np.ndarray:
        """À implémenter dans les classes filles."""
        pass

    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Méthode commune à tous les problèmes."""
        self.assemble_system()
        self.solution = self._solve_system()
        self._save_solution(save_name)
    
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Méthode commune de sauvegarde."""
        solution = Solution(
            data=self.solution,
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


class DirichletHomogeneousProblem(PoissonProblem):
    def _compute_rhs(self) -> np.ndarray:
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        P_0 = assemble_P_0(mesh=self.config.mesh)
        A_0 = P_0 @ (self.mass_matrix + self.stiffness_matrix) @ P_0.T
        L_0 = P_0 @ self.rhs
        return P_0.T @ sparse.linalg.spsolve(A_0, L_0)


class NeumannHomogeneousProblem(PoissonProblem):
    def _compute_rhs(self) -> np.ndarray:
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        return sparse.linalg.spsolve(
            self.mass_matrix + self.stiffness_matrix,
            self.rhs
        )


class PeriodicProblem(PoissonProblem):
    def _compute_rhs(self) -> np.ndarray:
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        P_per = assemble_P_per(mesh=self.config.mesh)
        A_per = P_per @ (self.mass_matrix + self.stiffness_matrix) @ P_per.T
        L_per = P_per @ self.rhs
        return P_per.T @ sparse.linalg.spsolve(A_per, L_per)


class CellProblem(PoissonProblem):
    def __init__(self, config: CellProblemConfig):
        super().__init__(config)
        # Correcteurs dans chaque direction
        self.corrector_x = None  # Pour la direction e₁
        self.corrector_y = None  # Pour la direction e₂
        # Tenseur homogénéisé calculé à partir des correcteurs
        self.homogenized_tensor = None
        
    def compute_homogenized_tensor(self) -> np.ndarray:
        if self.correctors is None:
            raise ValueError("Need to solve the cell problems first")
        
        A_eff = np.zeros((2,2))
        for j in range(2):
            for k in range(2):
                # A_eff[i,j] = (self.mesh.node_coords[:, j] + self.correctors[j]).T @ self.stiffness_matrix @ (self.mesh.node_coords[:, i] + self.correctors[i])
                # A_eff[j, k] = (self.mesh.node_coords[:, k] + self.correctors[j]).T @ self.stiffness_matrix @ (self.mesh.node_coords[:, i] + self.correctors[i])

                A_eff[j,k] = np.dot(self.stiffness_matrix @ (self.config.mesh.node_coords[:, k] + self.correctors[k]), self.config.mesh.node_coords[:, j] + self.correctors[j])
                
        return A_eff
    
    def _compute_rhs(self, direction: str) -> np.ndarray:
        if direction == 'x':
            return -self.stiffness_matrix @ self.config.mesh.node_coords.T[0]
        elif direction == 'y':
            return -self.stiffness_matrix @ self.config.mesh.node_coords.T[1]
    
    def _solve_system(self) -> np.ndarray:
        """Résolution du système pour une direction donnée."""
        P_per = assemble_P_per(mesh=self.config.mesh)
        A_per = P_per @ (self.mass_matrix + self.stiffness_matrix) @ P_per.T
        
        
        # Calculer le second membre spécifique aux problèmes de cellule
        # rhs = self._compute_rhs(direction)
        
        corrector_x = P_per.T @ sparse.linalg.spsolve(A_per, P_per @ self._compute_rhs(direction='x'))
        corrector_y = P_per.T @ sparse.linalg.spsolve(A_per, P_per @ self._compute_rhs(direction='y'))
        # L_per = P_per @ rhs
        return corrector_x, corrector_y
        # return P_per.T @ sparse.linalg.spsolve(A_per, L_per)
        
    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Override pour sauvegarder les correcteurs et le tenseur homogénéisé."""
        self.assemble_system()
        self.corrector_x, self.corrector_y = self._solve_system()
        # self.corrector_y = self._solve_system(direction='y')
        self.homogenized_tensor = self.compute_homogenized_tensor()
        self._save_solution(save_name)
    
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Adaptation pour sauvegarder les correcteurs et le tenseur homogénéisé."""
        solution = Solution(
            data={'corrector_x': self.corrector_x,
                  'corrector_y': self.corrector_y},
            problem_type=str(self.config.problem_type),
            metadata={
                'mesh_size': self.config.mesh_size,
                'problem_params': self.config.to_dict(),
                'homogenized_tensor': self.homogenized_tensor.tolist()
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

class HomogenizedProblem(PoissonProblem):
    def __init__(self, config: HomogenizedConfig):
        super().__init__(config)
        self.solution_gradient = None
        
    def compute_gradient(self) -> np.ndarray:
        """Calcule le gradient de la solution."""
        # À implémenter selon votre méthode de calcul du gradient
        pass
        
    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Override pour inclure le calcul du gradient."""
        self.assemble_system()
        self.solution = self._solve_system()
        self.solution_gradient = self.compute_gradient()
        self._save_solution(save_name)
    
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Adaptation pour sauvegarder la solution et son gradient."""
        solution = Solution(
            data={
                'solution': self.solution,
                'gradient': self.solution_gradient
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