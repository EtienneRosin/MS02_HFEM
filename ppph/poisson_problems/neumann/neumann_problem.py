"""Module providing the NeumannProblem class.

References
----------
.. [1] https://perso.ensta-paris.fr/~fliss/teaching-an201.html
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
# from ppph.utils.quadratures.gauss_lobatto_4_points import quadrature_weights, quadrature_points
from ppph.utils.quadratures.gauss_legendre_6_points import quadrature_weights, quadrature_points
from ppph.utils import ReferenceElementBarycentricCoordinates, DiffusionTensor, TwoDimensionFunction
from ppph.utils.graphics import display_field_on_mesh, display_3d

bc = ReferenceElementBarycentricCoordinates()

class NeumannProblem:
    r"""
    Define a Poisson problem with variable coefficients and Neumann boundary 
    conditions.

    ...

    Attributes
    ----------
    mesh: CustomTwoDimensionMesh
        Mesh of the problem.
    diffusion_tensor: callable
        Diffusion tensor :math:`\boldsymbol{A}` of the problem.
    rhs: TwoDimensionFunction
        Right-hand side member of the problem.
    M: sp.sparse._csr.csr_array
        Mass matrix :math:`\mathbb{M}` of the problem (None if not constructed).
    K: sp.sparse._csr.csr_array
        Rigidity matrix :math:`\mathbb{K}` of the problem (None if not constructed).
    U: np.ndarray
        Approximate solution.
    exact_solution: TwoDimensionFunction, default None
        Exact solution function.
    U_exact: np.ndarray
        Exact solution on the domain
    relative_L2_error: float
        relative :math:`L^2(\Omega)` error (defined if the exact solution is provided)
    relative_H1_error: float
        relative :math:`H^1(\Omega)` error (defined if the exact solution is provided)

    Notes
    -----
    The problem is the following:
    
    Find :math:`u \in H^1(\Omega)` such that:
    .. math:: \begin{case} \quad u - \nabla \cdot (\boldsymbol{A}\nabla u) &= rhs,\ & \text{in } \Omega\\
    \quad \boldsymbol{A}\nabla u \cdot \boldsymbol{n} &= 0,\ & \text{on } \partial\Omega \end{cases}
    """
    def __init__(self, mesh: CustomTwoDimensionMesh, diffusion_tensor: callable, rhs: callable, exact_solution: callable = None) -> None:
        r"""Construct the NeumannProblem object.

        Parameters
        ----------
        mesh : CustomTwoDimensionMesh
            Mesh of the problem.
        diffusion_tensor : callable
            Expression of the problem's diffusion tensor.
        rhs: callable
            Expression of the problem's right-hand side member of the problem.
        exact_solution : callable, default None
            Expression of the problem's exact solution.
        """
        self.mesh: CustomTwoDimensionMesh = mesh
        self.rhs: TwoDimensionFunction = TwoDimensionFunction(expr = rhs)
        self.diffusion_tensor: DiffusionTensor = DiffusionTensor(expr = diffusion_tensor)
        
        self.M: sp.sparse._csr.csr_array = None
        self.K: sp.sparse._csr.csr_array = None
        
        self.U: np.ndarray = None
        
        self.exact_function: TwoDimensionFunction = None if exact_solution is None else TwoDimensionFunction(expr = exact_solution)
        self.U_exact = None
        self.relative_L2_error: float = 0.0
        self.relative_H1_error: float = 0.0
    
    
    def _construct_elementary_mass_matrix(self, triangle: np.ndarray) -> np.ndarray:
        r"""Construct the P1 lagrange elementary mass matrix M^l on the considered triangle.
        
        Parameters
        ----------
        triangle: list
            considered triangle (list of 3 indices).

        Returns
        -------
        M_l: np.ndarray
            (3x3) elementary mass matrix M^l on the considered triangle.
        """
        triangle_nodes = self.mesh.node_coords[triangle]
        D_l = np.abs(bc.D_l(*triangle_nodes))
        M_l = np.ones((3,3), dtype = float)
        for i in range(3):
            M_l[i,i] += 1
        M_l *= D_l / 24
        return M_l
    
    def _construct_elementary_rigidity_matrix(self, triangle: np.ndarray) -> np.ndarray:
        r"""Construct the P1 lagrange elementary rigidity matrix K^l on the considered triangle by a quadrature.
        
        Parameters
        ----------
        triangle: list
            considered triangle (list of 3 indices).

        Returns
        -------
        K_l: np.ndarray
            (3x3) elementary rigidity matrix K^l on the considered triangle.
        """
        triangle_nodes = self.mesh.node_coords[triangle]
        D_l = np.abs(bc.D_l(*triangle_nodes))
        A_l = bc.A_l(*triangle_nodes)
        K_l = np.zeros((3,3))
        
        for i in range(3):
            for j in range(3):
                for omega_q, S_q in zip(quadrature_weights, quadrature_points):
                    K_l[i,j] += omega_q * np.dot(self.diffusion_tensor(bc.F_l(S_q, *triangle_nodes)) @ (A_l @ bc.grad_w_tilde(i+1, S_q)), A_l @ bc.grad_w_tilde(j+1, S_q)) 
        # # Normalize the rigidity matrix
        K_l /= D_l
        return K_l

    def _construct_A(self) -> tuple[np.ndarray]:
        r"""Construct the A matrix (K + M) of the linear system of the discrete variationnal formulation.
        
        Returns
        -------
        M: np.ndarray
            Mass matrix.
        K: np.ndarray
            Rigidity matrix.
        """
        M = sp.sparse.lil_matrix((self.mesh.num_nodes, self.mesh.num_nodes), dtype = float)
        K = sp.sparse.lil_matrix((self.mesh.num_nodes, self.mesh.num_nodes), dtype = float)
        
        print(f"Constructing the A matrix")
        for triangle in tqdm(self.mesh.tri_nodes):
            M_elem = self._construct_elementary_mass_matrix(triangle)
            K_elem = self._construct_elementary_rigidity_matrix(triangle)
            
            for i in range(0, 3):
                I = triangle[i]
                for j in range(0, 3):
                    J = triangle[j]
                    M[I, J] += M_elem[i, j]
                    K[I, J] += K_elem[i, j]
                    
        # Conversion to CSR format for better performance
        M = M.tocsr()
        K = K.tocsr()
        self.M = M
        self.K = K
        return M, K
    
    def _construct_L(self):
        r"""Construct the L vector (RHS of the linear system of the discrete variationnal formulation).
        
        Returns
        -------
        L: np.ndarray
            Right-hand side of the linear system.
        
        Raises
        ------
        ValueError
            If called before _construct_A()
        """
        if self.M is None:
            raise ValueError("Should construct the mass matrix before constructing the RHS.")
        L = self.M @ self.rhs(self.mesh.node_coords)
        return L
    
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
        U = sp.sparse.linalg.spsolve(self.M + self.K, L)
        self.U = U
        if not (self.exact_function is None):
            self.U_exact = self.exact_function(self.mesh.node_coords)
            self.relative_L2_error = self.compute_L2_error(self.U_exact)
            print(f"L^2 relative error : {self.relative_L2_error}")
            self.relative_H1_error = self.compute_H1_error(self.U_exact)
            print(f"H^1 relative error : {self.relative_H1_error}")
        return(U)
    
    def compute_L2_error(self, U_exact: np.ndarray) -> float:
        if self.U is None:
            self.solve()
        U = self.U
        return np.sqrt(np.dot(self.M @ (U_exact - U), U_exact - U))/np.sqrt(np.dot(self.M @ U_exact, U_exact))
    
    def compute_H1_error(self, U_exact: np.ndarray) -> float:
        if self.U is None:
            self.solve()
        U = self.U
        return np.sqrt(np.dot(self.K @ (U_exact - U), U_exact - U))/np.sqrt(np.dot(self.K @ U_exact, U_exact))
    
    def display(self, save_name: str = None):
        if self.U is None:
            self.solve()
        display_field_on_mesh(mesh=self.mesh, field=self.U, label='$u_h$', save_name=save_name)
    
    def display_exact_solution(self, save_name: str = None):
        if not (self.exact_function is None):
            self.U_exact = self.exact_function(self.mesh.node_coords)
        display_field_on_mesh(mesh=self.mesh, field=self.U_exact, label='$u$', save_name=save_name)
    
    def display_error(self, save_name: str = None):
        if self.U is None:
            self.solve()
        display_field_on_mesh(mesh=self.mesh, field=self.U_exact - self.U, label='$u - u_h$', save_name=save_name)
    
    def display_3d(self, save_name: str = None):
        display_3d(mesh=self.mesh, field=self.U, label='$u_h$', save_name=save_name)
if __name__ == '__main__':
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.025
    create_rectangle_mesh(h = h, L_x = 2, L_y = 1, save_name = mesh_fname)
    
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    # Problem parameters --------------------------------------------
    def diffusion_tensor(x, y):
        return np.eye(2)
    
    def u(x, y):
        return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

    def f(x, y):
        return (1 + 5 * np.pi ** 2) * u(x, y)
    
    
    
    # Problem itself ------------------------------------------------
    neumann_pb = NeumannProblem(
        mesh = mesh, 
        diffusion_tensor = diffusion_tensor, 
        rhs = f, 
        exact_solution = u
        )
    # neumann_pb._construct_A()
    # neumann_pb.display_exact_solution()
    neumann_pb.solve()
    neumann_pb.display_error()
    neumann_pb.display_3d()
    # triangle = mesh.tri_nodes[0]
    # neumann_pb._construct_elementary_rigidity_matrix(triangle=triangle)