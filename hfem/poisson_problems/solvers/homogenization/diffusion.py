
from hfem.poisson_problems.configs.homogenization.diffusion import DiffusionProblemConfig

from hfem.poisson_problems.solvers.base import PoissonProblem
from hfem.core.related_matrices import assemble_P_0
from hfem.poisson_problems.configs.homogenization.cell import CellProblemConfig
from hfem.core.io import Solution, FEMDataManager, MeshData, FEMMatrices

import scipy.sparse as sparse
import numpy as np
from pathlib import Path
from typing import Union, Tuple

# class CellProblem(PoissonProblem):
#     def __init__(self, config: CellProblemConfig):
#         super().__init__(config)
        
#         self.config.diff
#         self.corrector_x = None
#         self.corrector_y = None
#         self.homogenized_tensor = None
        
        

class DiffusionProblem(PoissonProblem):
    def __init__(self, config: DiffusionProblemConfig):
        super().__init__(config)
        # self.config.diffusion_tensor = lambda x,y : self.config.diffusion_tensor(x/self.config.epsilon, y/self.config.epsilon)
    pass
    
    def _compute_rhs(self) -> np.ndarray:
        """Calcul du second membre avec conditions de Dirichlet homogènes."""
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        """Résolution du système avec conditions de Dirichlet homogènes."""
        P_0 = assemble_P_0(mesh=self.config.mesh)
        A_0 = P_0 @ self.stiffness_matrix @ P_0.T
        L_0 = P_0 @ self._compute_rhs()
        return P_0.T @ sparse.linalg.spsolve(A_0, L_0)

epsilon = 1/100
def diffusion_tensor(x, y):
    return (2 + np.sin(2 * np.pi * x/epsilon))*(4 + np.sin(2 * np.pi * y/epsilon))*np.eye(2)


if __name__ == '__main__':
    mesh_file = "meshes/rectangle_mesh.msh"
    # rectangular geometry
    h = 0.0075   # mesh size
    h = 0.0125
    L_x = 2     # rectangle width
    L_y = 2     # rectangle height

        # Parameters --------------------------------
    

    def exact_solution(x,y):
        return np.sin(np.pi*x)*np.sin(np.pi*y)
    
    def right_hand_side(x, y):
        return -(2*np.pi**2/epsilon)*(-epsilon*(np.sin(2*np.pi*x/epsilon) + 2)*(np.sin(2*np.pi*y/epsilon) + 4)*exact_solution(x,y) \
            + (np.sin(2*np.pi*x/epsilon) + 2)*np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(2*np.pi*y/epsilon) \
            + (np.sin(2*np.pi*y/epsilon) + 4)*np.sin(np.pi*y)*np.cos(np.pi*x)*np.cos(2*np.pi*x/epsilon))
        
    from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
    mesh_file = "meshes/periodic_square.msh"
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # 4. Configuration du problème
    config = DiffusionProblemConfig(
        mesh=mesh,
        mesh_size=h,
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side,
        epsilon=epsilon
    )
    
    
    # 5. Création et résolution du problème
    problem = DiffusionProblem(config=config)
    problem.solve_and_save(save_name="diffusion_test")
    
    # 6. Chargement et visualisation des résultats
    from hfem.core.io import FEMDataManager
    
    manager = FEMDataManager()
    solution, mesh, matrices = manager.load(
        f"simulation_data/{str(config.problem_type)}/diffusion_test.h5"
    )
    # print(f"{type(solution.data) = }")
    
    mesh.display_field(
        field=solution.data,
        field_label=r"$u_0$",
        # kind='trisurface'
    )