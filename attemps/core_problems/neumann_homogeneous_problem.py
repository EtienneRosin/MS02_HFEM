from pathlib import Path
from typing import Union
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

from hfem.poisson_problems.poisson_problem import PoissonProblem
from hfem.core.related_matrices import (
    assemble_elementary_mass_matrix,
    assemble_elementary_stiffness_matrix
)
from hfem.core.io import FEMDataManager, FEMMatrices, MeshData
from hfem.poisson_problems.problem_configs import CorePoissonProblemsConfig

class NeumannHomogeneousProblem(PoissonProblem):
    def _compute_rhs(self) -> np.ndarray:
        """Calcul du second membre."""
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        """Résolution du système."""
        return sparse.linalg.spsolve(
            self.mass_matrix + self.stiffness_matrix,
            self.rhs
        )

    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Sauvegarde la solution avec la nouvelle structure."""
        from hfem.core.io.solution import Solution  # Import de la nouvelle classe
        
        solution = Solution(
            data=self.solution,
            problem_type='neumann',
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

if __name__ == '__main__':
    from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
    
    # Configuration du problème
    h = 0.1     # taille du maillage
    L_x = L_y = 2  # dimensions du rectangle
    
    def v(x, y):
        return np.sin(2*np.pi*x)*np.sin(2*np.pi*y) + 2

    def diffusion_tensor(x, y):
        return np.eye(2)*v(x, y)
            
    def exact_solution(x, y):
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y)

    def right_hand_side(x, y):
        return (1 + (16*np.pi**2)*(v(x, y)- 1))*exact_solution(x, y)
    
    # Création du maillage
    mesh_file = "meshes/rectangle_mesh.msh"
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    
    # Configuration et résolution
    config = CorePoissonProblemsConfig(
        mesh=CustomTwoDimensionMesh(mesh_file),
        mesh_size=h,
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side
    )
    
    # Résolution
    pb = NeumannHomogeneousProblem(config=config)
    pb.solve_and_save(save_name="tentative/test_2")
    
    # Chargement et affichage des résultats
    manager = FEMDataManager()
    solution, mesh, matrices = manager.load("simulation_data/neumann/test_2.h5")
    # result, mesh, matrices = manager.load("simulation_data/results/neumann/test.h5")
    # Affichage
    mesh.display_field(solution.data)  # Notez qu'on utilise .data au lieu de .solution
    
    # import h5py

    # # Ouvre le fichier et affiche sa structure
    # with h5py.File("simulation_data/neumann/test.h5", 'r') as f:
    #     def print_structure(name, obj):
    #         print(name)
    #         if isinstance(obj, h5py.Group):
    #             print("  Attributes:", dict(obj.attrs))
        
    #     f.visititems(print_structure)