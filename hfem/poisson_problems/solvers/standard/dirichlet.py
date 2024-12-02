from hfem.poisson_problems.solvers.base import PoissonProblem
from hfem.core.related_matrices import assemble_P_0
from hfem.poisson_problems.configs.standard.dirichlet import DirichletConfig
import scipy.sparse as sparse
import numpy as np

class DirichletHomogeneousProblem(PoissonProblem):
    def __init__(self, config: DirichletConfig):
        super().__init__(config)
    
    def _compute_rhs(self) -> np.ndarray:
        """Calcul du second membre avec conditions de Dirichlet homogènes."""
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        """Résolution du système avec conditions de Dirichlet homogènes."""
        P_0 = assemble_P_0(mesh=self.config.mesh)
        A_0 = P_0 @ (self.mass_matrix + self.stiffness_matrix) @ P_0.T
        L_0 = P_0 @ self._compute_rhs()
        return P_0.T @ sparse.linalg.spsolve(A_0, L_0)