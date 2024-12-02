from hfem.poisson_problems.solvers.base import PoissonProblem
from hfem.poisson_problems.configs.standard.neumann import NeumannConfig
import scipy.sparse as sparse
import numpy as np

class NeumannHomogeneousProblem(PoissonProblem):
    def __init__(self, config: NeumannConfig):
        super().__init__(config)
    
    def _compute_rhs(self) -> np.ndarray:
        """Calcul du second membre avec conditions de Neumann homogènes."""
        return self.mass_matrix @ self.config.right_hand_side(
            *self.config.mesh.node_coords.T
        )
    
    def _solve_system(self) -> np.ndarray:
        """Résolution du système avec conditions de Neumann homogènes."""
        return sparse.linalg.spsolve(
            self.mass_matrix + self.stiffness_matrix,
            self._compute_rhs()
        )