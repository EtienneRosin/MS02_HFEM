

from hfem.problems import HomogenousDirichletPoissonProblem, MicrostructuredPoissonConfig
from hfem.mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh

import numpy as np
import scipy as sp
from dataclasses import dataclass
from typing import Callable



class MicrostructuredPeriodicPoissonProblem(HomogenousDirichletPoissonProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, config: MicrostructuredPoissonConfig):
        super().__init__(mesh, config)
        
    def solve(self) -> np.ndarray:
        """
        Solve the Poisson problem.
        
        Returns
        -------
        np.ndarray
            Solution vector
        """
        # Assemble system
        A, L = self.assemble_system()
        
        # Assemble the projection matrix P for Dirichlet boundary conditions
        P_0 = self._construct_P_0()
        # Project onto V_h^0
        A_0 = P_0 @ self.stiffness_matrix @ P_0.T
        L_0 = P_0 @ L
        
        # Solve system
        U_0 = sp.sparse.linalg.spsolve(A_0, L_0)
        
        # Get back to V_h
        self.solution = P_0.T @ U_0
        # Compute errors if exact solution available
        if self.config.exact_solution is not None:
            # print(f"sol calculée, on calcule l'erreur")
            self._compute_errors()
        
        return self.solution
    
if __name__ == '__main__':
    
    epsilon = 1

    def diffusion_tensor(x, y):
        return np.eye(2)

    def u(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def f(x, y): 
        return 2*(np.pi**2)*u(x,y)