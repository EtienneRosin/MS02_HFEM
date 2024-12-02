from hfem.poisson_problems.solvers.base import PoissonProblem
from hfem.core.related_matrices import assemble_P_per
from hfem.poisson_problems.configs.standard.periodic import PeriodicConfig
import scipy.sparse as sparse
import numpy as np

class PeriodicProblem(PoissonProblem):
    def __init__(self, config: PeriodicConfig):
        super().__init__(config)
        # if not config.mesh.is_periodic_compatible():
        #     raise ValueError("Mesh must be compatible with periodic conditions")
    
    def _compute_rhs(self) -> np.ndarray:
        """Calcul du second membre avec conditions périodiques."""
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        """Résolution du système avec conditions périodiques."""
        P_per = assemble_P_per(mesh=self.config.mesh)
        A_per = P_per @ (self.mass_matrix + self.stiffness_matrix) @ P_per.T
        L_per = P_per @ self._compute_rhs()
        return P_per.T @ sparse.linalg.spsolve(A_per, L_per)