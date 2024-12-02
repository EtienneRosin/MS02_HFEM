from hfem.poisson_problems.problem_configs import CorePoissonProblemsConfig
from hfem.core.io import FEMDataManager, FEMMatrices, Solution, MeshData

from hfem.core.related_matrices import assemble_elementary_mass_matrix, assemble_elementary_stiffness_matrix

import scipy.sparse as sparse
from tqdm import tqdm
from pathlib import Path
from typing import Union
import numpy as np

class PoissonProblem:
    """Classe de base pour tous les problèmes de Poisson."""
    
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

    def _compute_rhs(self) -> np.ndarray:
        """À implémenter dans les classes filles."""
        raise NotImplementedError

    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Résout et sauvegarde avec nom de fichier standard."""
        self.assemble_system()
        self.solution = self._solve_system()
        self._save_solution(save_name)

    def _compute_rhs(self) -> np.ndarray:
        """À implémenter dans les classes filles."""
        raise NotImplementedError

    # def _solve_system(self) -> np.ndarray:
    #     """Résolution du système."""
    #     return sparse.linalg.spsolve(
    #         self.mass_matrix + self.stiffness_matrix,
    #         self.rhs
    #     )

    # def _save_solution(self, save_name: Union[str, Path]) -> None:
    #     """Sauvegarde la solution."""
    #     manager = FEMDataManager()
    #     manager.save_solution(
    #         save_name=save_name,
    #         solution=StandardPoissonSolution(
    #             solution=self.solution,
    #             boundary_type=self._get_boundary_type()
    #         ),
    #         mesh=MeshData.from_mesh(self.config.mesh),
    #         matrices=FEMMatrices(
    #             mass_matrix=self.mass_matrix,
    #             stiffness_matrix=self.stiffness_matrix
    #         )
    #     )
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        raise NotImplementedError
        
    # def _get_boundary_type(self) -> str:
    #     """À implémenter dans les classes filles."""
    #     raise NotImplementedError
    

class CorePoissonProblem:
    """Classe de base pour tous les problèmes de Poisson de base (Dirichlet et Neumann Homogène, périodique et problème homogénéisé)."""
    
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

    def _compute_rhs(self) -> np.ndarray:
        """À implémenter dans les classes filles."""
        raise NotImplementedError

    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Résout et sauvegarde avec nom de fichier standard."""
        self.assemble_system()
        self.solution = self._solve_system()
        self._save_solution(save_name)

    def _compute_rhs(self) -> np.ndarray:
        """À implémenter dans les classes filles."""
        raise NotImplementedError

    # def _solve_system(self) -> np.ndarray:
    #     """Résolution du système."""
    #     return sparse.linalg.spsolve(
    #         self.mass_matrix + self.stiffness_matrix,
    #         self.rhs
    #     )

    # def _save_solution(self, save_name: Union[str, Path]) -> None:
    #     """Sauvegarde la solution."""
    #     manager = FEMDataManager()
    #     manager.save_solution(
    #         save_name=save_name,
    #         solution=StandardPoissonSolution(
    #             solution=self.solution,
    #             boundary_type=self._get_boundary_type()
    #         ),
    #         mesh=MeshData.from_mesh(self.config.mesh),
    #         matrices=FEMMatrices(
    #             mass_matrix=self.mass_matrix,
    #             stiffness_matrix=self.stiffness_matrix
    #         )
    #     )
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        raise NotImplementedError