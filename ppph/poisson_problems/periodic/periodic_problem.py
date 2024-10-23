from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
from ppph.poisson_problems import NeumannProblem
from ppph.utils.matrix_info import display_matrix_info

import numpy as np
import scipy as sp

class PeriodicProblem(NeumannProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, diffusion_tensor: callable, rhs: callable, exact_solution: callable = None, boundary_labels: str | list[str] = '$\\partial\\Omega$') -> None:
        super().__init__(mesh, diffusion_tensor, rhs, exact_solution)
        self.P_per = self._construct_P_per(boundary_labels)
        
    def _construct_P_per(self, boundary_labels: str | list[str] = '$\\partial\\Omega$', tolerance: float = 1e-10) -> np.ndarray:
        r"""
        Construct the P matrix that projects the approximation space V_h of H^1(\Omega) into V_h^per, the approximation space of H^per(\Omega).

        Parameters
        ----------
        boundary_labels : str or list of str, default '$\partial\Omega$'
            List of labels indicating the boundaries.
        tolerance : float, default 1e-10
            Tolerance for the equality test.

        Returns
        -------
        P : np.ndarray
            The projection matrix.
        """
        if isinstance(boundary_labels, str):
            boundary_labels = [boundary_labels]
            
        corner_indices, pairs_same_x, pairs_same_y, inner_indices = self.mesh.get_corner_and_boundary_pairs(boundary_labels, tolerance)
        # Initialize the projector matrix P
        N_inner = len(inner_indices)
        N_corner = 1    # we keep only one corner
        N_pairs_x = len(pairs_same_x)
        N_pairs_y = len(pairs_same_y)
        N_per = N_inner + N_corner + N_pairs_x + N_pairs_y
        P = sp.sparse.lil_matrix((N_per, self.mesh.num_nodes))

        # Assign values to P based on inner indices and corner indices
        for n, i in enumerate(inner_indices):
            P[n,i] = 1
        
        for i in corner_indices:
            P[N_inner,i] = 1
        
        for n, (i,j) in enumerate(pairs_same_x):
            P[n + N_inner + N_corner, i] = P[n + N_inner + N_corner, j] = 1
            
        for n, (i,j) in enumerate(pairs_same_y):
            P[n + N_inner + N_corner + N_pairs_x, i] = P[n + N_inner + N_corner+ N_pairs_x, j] = 1
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
        L_per = self.P_per @ L
        # display_matrix_info((self.M + self.K).toarray(), name="A")
        A_per = self.P_per @ (self.M + self.K) @ self.P_per.T
        # display_matrix_info(A_per.toarray(), name="A_per")
        U_per = sp.sparse.linalg.spsolve(A_per, L_per)
        U = self.P_per.T @ U_per
        self.U = U
        if not (self.exact_function is None):
            U_exact = self.exact_function(self.mesh.node_coords)
            self.relative_L2_error = self.compute_L2_error(U_exact)
            print(f"L^2 relative error : {self.relative_L2_error}")
            self.relative_H1_error = self.compute_H1_error(U_exact)
            print(f"H^1 relative error : {self.relative_H1_error}")
        return(U)


if __name__ == "__main__":
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.025
    create_rectangle_mesh(h = h, L_x = 2, L_y = 1, save_name = mesh_fname)
    
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    # Problem parameters --------------------------------------------
    def diffusion_tensor(x, y):
        return np.eye(2)
    
    def u(x, y):
        return np.sin(np.pi * x) * np.sin(2 * np.pi * y)

    def f(x, y):
        return (1 + 5 * np.pi ** 2) * u(x, y)
    
    # Problem itself ------------------------------------------------
    
    periodic_pb = PeriodicProblem(mesh = mesh, diffusion_tensor = diffusion_tensor, rhs = f, exact_solution = u)
    # neumann_pb._construct_A()
    periodic_pb.solve()
    # print(periodic_pb.U)
    periodic_pb.display()
    # triangle = mesh.tri_nodes[0]
    # neumann_pb._construct_elementary_rigidity_matrix(triangle=triangle)