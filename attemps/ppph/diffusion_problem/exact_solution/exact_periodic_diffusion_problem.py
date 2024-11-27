

from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.poisson_problems import PoissonDirichletProblem

import scipy as sp
import numpy as np

class ExactPeriodicDiffusionProblem(PoissonDirichletProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, diffusion_tensor: callable, rhs: callable, exact_solution: callable = None) -> None:
        super().__init__(mesh, diffusion_tensor, rhs, exact_solution)
        
    def solve(self):
        r"""Solve he linear system of the discrete variationnal formulation.
        
        Returns
        -------
        U: np.ndarray
            Approximate solution.
        """
        if (self.M is None) or (self.K is None):
            self._construct_A()
        L = self._construct_L()
        L_0 = self.P @ L
        # A_0 = self.P @ (self.M + self.K) @ self.P.T
        A_0 = self.P @ self.K @ self.P.T
        U_0 = sp.sparse.linalg.spsolve(A_0, L_0)
        U = self.P.T @ U_0
        self.U = U
        if not (self.exact_function is None):
            self.U_exact = self.exact_function(self.mesh.node_coords)
            self.relative_L2_error = self.compute_L2_error(self.U_exact)
            print(f"L^2 relative error : {self.relative_L2_error}")
            self.relative_H1_error = self.compute_H1_error(self.U_exact)
            print(f"H^1 relative error : {self.relative_H1_error}")
        return(U)
    

if __name__ == '__main__':
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.05
    h = 0.025
    create_rectangle_mesh(h = h, L_x = 2, L_y = 2, save_name = mesh_fname)
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    # mesh.display()
    # Problem parameters --------------------------------------------
    epsilon = 1e-1
    # def A(x, y):
    #     return np.diagflat([1,2], 0)
    
    def A(x, y):
        return np.eye(2)
    
    def diffusion_tensor(x, y):
        return A(x/epsilon, y/epsilon)

    def u(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def f(x, y): 
        return 2*(np.pi**2)*u(x,y)
    
    # Problem itself ------------------------------------------------
    periodic_pb = ExactPeriodicDiffusionProblem(
        mesh = mesh, 
        diffusion_tensor = diffusion_tensor, 
        rhs = f, 
        exact_solution = u)
    periodic_pb.solve()
    # periodic_pb.display()
    periodic_pb.display_3d()
    periodic_pb.display_error()