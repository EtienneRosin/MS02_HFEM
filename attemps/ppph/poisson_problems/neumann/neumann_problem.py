r"""
Module providing the `PoissonNeumannProblem` class to solve Poisson problems with Neumann boundary conditions.

The problem is formulated as:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
   & \boldsymbol{A} \nabla u \cdot \boldsymbol{n} = 0, \quad \text{on } \partial\Omega.
   \end{aligned}

This corresponds to a variational formulation:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   \int_\Omega u v \, dx 
   + \int_\Omega \boldsymbol{A} \nabla u \cdot \nabla v \, dx 
   &= \int_\Omega f v \, dx, \quad \forall v \in H^1(\Omega),
   \end{aligned}

References
----------
.. [1] [Finite Element Analysis](https://perso.ensta-paris.fr/~fliss/teaching-an201.html)
.. [2] [Project](https://perso.ensta-paris.fr/~fliss/ressources/Homogeneisation/TP.pdf)
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
# from ppph.utils.quadratures.gauss_lobatto_4_points import quadrature_weights, quadrature_points
from ppph.utils.quadratures.gauss_legendre_6_points import quadrature_weights, quadrature_points
from ppph.utils import ReferenceElementBarycentricCoordinates, DiffusionTensor, TwoDimensionFunction
from ppph.utils.graphics import display_field_on_mesh, display_3d


bc = ReferenceElementBarycentricCoordinates()

class PoissonNeumannProblem:
    r"""
    Class for solving Poisson problems with Neumann boundary conditions using P1 Lagrange finite elements.

    Attributes
    ----------
    mesh : CustomTwoDimensionMesh
        The finite element mesh representing the domain :math:`\Omega`.
    diffusion_tensor : callable
        A function defining the diffusion tensor :math:`\boldsymbol{A}`.
    rhs : callable
        The right-hand side function :math:`f`.
    M : sp.sparse.csr_matrix or None
        The mass matrix :math:`\mathbb{M}`, initialized as None.
    K : sp.sparse.csr_matrix or None
        The rigidity matrix :math:`\mathbb{K}`, initialized as None.
    U : np.ndarray or None
        The computed solution, initialized as None.
    exact_solution : callable or None
        An optional exact solution :math:`u` for validation and error analysis.
    U_exact : np.ndarray or None
        The exact solution evaluated at the mesh nodes, if provided.
    relative_L2_error : float
        The relative :math:`L^2(\Omega)` error between the computed and exact solutions, if available.
    relative_H1_error : float
        The relative :math:`H^1(\Omega)` error between the computed and exact solutions, if available.

    Methods
    -------
    _construct_elementary_mass_matrix(triangle)
        Computes the elementary mass matrix for a given triangle.
    _construct_elementary_rigidity_matrix(triangle)
        Computes the elementary rigidity matrix for a given triangle using numerical quadrature.
    _construct_A()
        Constructs the global mass and rigidity matrices.
    _construct_L()
        Constructs the global right-hand side vector.
    solve()
        Solves the variational formulation of the problem and computes the solution.
    compute_L2_error(U_exact)
        Computes the :math:`L^2(\Omega)` norm of the error.
    compute_H1_error(U_exact)
        Computes the :math:`H^1(\Omega)` norm of the error.
    display(save_name=None)
        Visualizes the computed solution on the mesh.
    display_exact_solution(save_name=None)
        Visualizes the exact solution if available.
    display_error(save_name=None)
        Visualizes the error between the computed and exact solutions.
    display_3d(save_name=None, view_init=(10, -7, 0))
        Displays the solution as a 3D surface plot.

    Notes
    -----
    The `solve` method performs all necessary steps, including constructing the system matrices and solving the resulting linear system.

    Examples
    --------
    Here's an example of how to use this class to solve a Poisson problem:

    .. code-block:: python

        from mesh_manager import create_rectangle_mesh, CustomTwoDimensionMesh
        import numpy as np
        from ppph.poisson_problems import PoissonNeumannProblem

        # Define the mesh and problem parameters
        def diffusion_tensor(x, y):
            return np.eye(2)

        def u(x, y):
            return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

        def f(x, y):
            return (1 + 5 * np.pi ** 2) * u(x, y)

        mesh_file = "rectangle_mesh.msh"
        create_rectangle_mesh(h=0.1, L_x=1, L_y=1, save_name=mesh_file)
        mesh = CustomTwoDimensionMesh(mesh_file)

        # Initialize and solve the problem
        problem = PoissonNeumannProblem(mesh, diffusion_tensor, f, exact_solution=u)
        problem.solve()
        problem.display_3d()
        problem.display_error()

    """
    def __init__(self, mesh: CustomTwoDimensionMesh, diffusion_tensor: callable, rhs: callable, exact_solution: callable = None) -> None:
        r"""
        Initialize a Poisson problem with Neumann boundary conditions.

        Parameters
        ----------
        mesh : CustomTwoDimensionMesh
            The mesh representing the domain.
        diffusion_tensor : callable
            Function defining the diffusion tensor.
        rhs : callable
            Function defining the right-hand side.
        exact_solution : callable, optional
            Exact solution for validation purposes.
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
    
    def display_3d(self, save_name: str = None, view_init: tuple = (10, -7, 0)):
        display_3d(mesh=self.mesh, field=self.U, label='$u_h$', save_name=save_name, view_init=view_init)
if __name__ == '__main__':
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    # h = 0.05
    h = 0.015
    create_rectangle_mesh(h = h, L_x = 1, L_y = 1, save_name = mesh_fname)
    
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    # Problem parameters --------------------------------------------
    a = 2
    sigma = 10
    def v(x, y):
        return np.sin(a * np.pi * x) * np.sin(a * np.pi * y) + 2

    # def diffusion_tensor(x, y):
    #     return v(x, y) * np.eye(2)
    
    # def u(x, y):
    #     return np.cos(np.pi * x) * np.cos(2 * np.pi * y)
    # def f(x, y):
    #     return sigma*y
    
    
    def diffusion_tensor(x, y):
        return np.eye(2)*v(x,y)

    def exact_solution(x, y):
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y)

    def right_hand_side(x, y):
        # return np.pi**2 * np.cos(np.pi*x) * np.cos(2*np.pi*y)
        return (1 + (16*np.pi**2)*(v(x,y) -1))*exact_solution(x,y)

    
    
    
    # Problem itself ------------------------------------------------
    neumann_pb = PoissonNeumannProblem(
        mesh = mesh, 
        diffusion_tensor = diffusion_tensor, 
        rhs = right_hand_side, 
        exact_solution = exact_solution
        )
    # neumann_pb._construct_A()
    # neumann_pb.display_exact_solution()
    neumann_pb.solve()
    # neumann_pb.display_error()
    # neumann_pb.display()
    # neumann_pb.display_3d(view_init=(8, -7, 0))
    neumann_pb.display_3d(save_name=f"neuman_{a}", view_init=(8, -7, 0))
    # field = neumann_pb.U - neumann_pb.rhs(mesh.node_coords)
    # field = neumann_pb.U - neumann_pb.U.mean()
    # display_3d(mesh=mesh, field=field, label='$u_h$')
    # triangle = mesh.tri_nodes[0]
    # neumann_pb._construct_elementary_rigidity_matrix(triangle=triangle)