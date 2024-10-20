import numpy as np
import scipy as sp

from ppph.poisson_problems import NeumannProblem
from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh

class DirichletProblem(NeumannProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, diffusion_tensor: callable, rhs: callable, exact_solution: callable = None) -> None:
        super().__init__(mesh, diffusion_tensor, rhs, exact_solution)
        self.P = self._construct_P(mesh)
    
    def _construct_P(self, mesh: CustomTwoDimensionMesh) -> sp.sparse.csr_matrix:
        r"""
        Create the P matrix that returns the interior nodes of the mesh.

        Parameters
        ----------
        mesh : CustomTwoDimensionMesh
            Considered mesh.

        Returns
        -------
        P : sp.sparse.csr_matrix
            The P matrix.
        """
        on_border_ref = mesh.labels['$\\partial\\Omega$']
        interior_indices = np.where(mesh.node_refs != on_border_ref)[0]
        N_0 = len(interior_indices)
        N = mesh.num_nodes

        P = sp.sparse.lil_matrix((N_0, N), dtype=float)
        for i, j in enumerate(interior_indices):
            P[i, j] = 1

        return P.tocsr()
    
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
        A_0 = self.P @ (self.M + self.K) @ self.P.T
        U_0 = sp.sparse.linalg.spsolve(A_0, L_0)
        U = self.P.T @ U_0
        self.U = U
        if not (self.exact_function is None):
            U_exact = self.exact_function(self.mesh.node_coords)
            self.relative_L2_error = np.sqrt(np.dot(self.M @ (U_exact - U), U_exact - U))/ np.sqrt(np.dot(self.M @ U_exact, U_exact))
            print(f"L^2 relative error : {self.relative_L2_error}")
            self.relative_H1_error = np.sqrt(np.dot(self.K @ (U_exact - U), U_exact - U))/ np.sqrt(np.dot(self.K @ U_exact, U_exact))
            print(f"H^1 relative error : {self.relative_H1_error}")
        return(U)
        

if __name__ == "__main__":
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.05
    # create_rectangle_mesh(h = h, L_x = 2, L_y = 1, save_name = mesh_fname)
    
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    # Problem parameters --------------------------------------------
    def diffusion_tensor(x, y):
        return np.eye(2)
    
    def u(x, y):
        return np.sin(np.pi * x) * np.sin(2 * np.pi * y)

    def f(x, y):
        return (1 + 5 * np.pi ** 2) * u(x, y)
    
    # Problem itself ------------------------------------------------
    dirichlet_pb = DirichletProblem(mesh=mesh, diffusion_tensor=diffusion_tensor, rhs=f, exact_solution=u)
    # neumann_pb._construct_A()
    dirichlet_pb.solve()
    dirichlet_pb.display()
    # triangle = mesh.tri_nodes[0]
    # neumann_pb._construct_elementary_rigidity_matrix(triangle=triangle)