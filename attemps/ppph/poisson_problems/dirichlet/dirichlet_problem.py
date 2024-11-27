r"""
This module defines a Poisson problem with Dirichlet boundary conditions. 

The problem is formulated as:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
   & u = 0, \quad \text{on } \partial\Omega.
   \end{aligned}

Which is equivalent to the variational formulation:

.. math::
   \text{Find } u \in H^1_0(\Omega) \text{ such that: } \\
   \begin{aligned}
   \int_\Omega u v \, dx 
   + \int_\Omega \boldsymbol{A} \nabla u \cdot \nabla v \, dx 
   &= \int_\Omega f v \, dx, \quad \forall v \in H^1_0(\Omega),
   \end{aligned}

References
----------
.. [1] [Finite Element Analysis](https://perso.ensta-paris.fr/~fliss/teaching-an201.html)
.. [2] [Project](https://perso.ensta-paris.fr/~fliss/ressources/Homogeneisation/TP.pdf)
"""
import numpy as np
import scipy as sp

from ppph.poisson_problems import PoissonNeumannProblem
from mesh_manager import CustomTwoDimensionMesh
from mesh_manager.geometries import rectangular_mesh

class PoissonDirichletProblem(PoissonNeumannProblem):
    r"""
    A class to represent a Poisson problem with Dirichlet boundary conditions.

    The considered problem is:

    .. math::
       \text{Find } u \in H^1(\Omega) \text{ such that: } \\
       \begin{aligned}
       & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
       & u = 0, \quad \text{on } \partial\Omega.
       \end{aligned}

    This class inherits from `PoissonNeumannProblem` and specializes it by imposing 
    Dirichlet boundary conditions through the construction of a projection matrix \( P \). 
    It also provides functionalities for solving the problem and analyzing the solution.

    Attributes
    ----------
    mesh : CustomTwoDimensionMesh
        The finite element mesh representing the domain :math:`\Omega`.
    diffusion_tensor : callable
        A function that returns the diffusion tensor :math:`\boldsymbol{A}`.
    rhs : callable
        The right-hand side function :math:`f`.
    exact_solution : callable, optional
        An optional exact solution :math:`u` for error analysis.
    P : sp.sparse.csr_matrix
        The projection matrix that selects interior nodes of the mesh.

    Methods
    -------
    _construct_P(mesh)
        Constructs the projection matrix :math:`\mathbb{P}` to enforce Dirichlet boundary conditions.
    solve()
        Solves the linear system corresponding to the discrete variational formulation.
    display()
        Displays the computed solution on the mesh.
    display_3d()
        Visualizes the computed solution as a 3D surface.
    display_error()
        Shows the error distribution, if the exact solution is provided.
    compute_L2_error(u_exact)
        Computes the :math:`L^2` norm of the error between the computed and exact solutions.
    compute_H1_error(u_exact)
        Computes the :math:`H^1` norm of the error between the computed and exact solutions.

    Notes
    -----
    The `solve` method handles the construction of the linear system and applies the 
    projection matrix :math:`\mathbb{P}` to enforce the Dirichlet boundary conditions on the solution.

    Examples
    --------
    Here's an example of how to use this class to solve a Poisson problem:

    .. code-block:: python

        # Define a mesh and problem parameters
        from mesh_manager import CustomTwoDimensionMesh, create_rectangle_mesh
        import numpy as np

        def diffusion_tensor(x, y):
            return np.eye(2)

        def u(x, y):
            return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

        def f(x, y):
            return (1 + 16 * (np.pi ** 2)) * u(x, y)

        # Create the mesh
        mesh_fname = "mesh_manager/geometries/rectangle.msh"
        h = 0.025
        create_rectangle_mesh(h=h, L_x=1, L_y=1, save_name=mesh_fname)
        mesh = CustomTwoDimensionMesh(mesh_fname, reordering=True)

        # Define and solve the problem
        dirichlet_pb = PoissonDirichletProblem(mesh=mesh, diffusion_tensor=diffusion_tensor, rhs=f, exact_solution=u)
        dirichlet_pb.solve()
        dirichlet_pb.display()
        dirichlet_pb.display_3d()
    """
    def __init__(self, mesh: CustomTwoDimensionMesh, diffusion_tensor: callable, rhs: callable, exact_solution: callable = None) -> None:
        super().__init__(mesh, diffusion_tensor, rhs, exact_solution)
        self.P = self._construct_P(mesh)
    
    def _construct_P(self, mesh: CustomTwoDimensionMesh) -> sp.sparse.csr_matrix:
        r"""
        Create the P matrix that selects the interior nodes of the mesh.

        Parameters
        ----------
        mesh : CustomTwoDimensionMesh
            Considered mesh.

        Returns
        -------
        P : sp.sparse.csr_matrix
            The P matrix.
        """
        on_border_ref = mesh.labels["$\\partial\\Omega$"]
        interior_indices = np.where(mesh.node_refs != on_border_ref)[0]
        # print(interior_indices)
        N_0 = len(interior_indices)
        N = mesh.num_nodes

        P = sp.sparse.lil_matrix((N_0, N), dtype=float)
        for i, j in enumerate(interior_indices):
            P[i, j] = 1

        return P.tocsr()
    
    def solve(self):
        r"""
        Solve the linear system of the discrete variational formulation.

        This method constructs the reduced system of equations by applying the 
        projection matrix :math:`\mathbb{P}` to enforce Dirichlet boundary conditions. 
        It then solves the resulting sparse linear system and then project back the 
        solution in the correct space.

        Returns
        -------
        U : np.ndarray
            The approximate solution at the mesh nodes.
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
            self.U_exact = self.exact_function(self.mesh.node_coords)
            self.relative_L2_error = self.compute_L2_error(self.U_exact)
            print(f"L^2 relative error : {self.relative_L2_error}")
            self.relative_H1_error = self.compute_H1_error(self.U_exact)
            print(f"H^1 relative error : {self.relative_H1_error}")
        return(U)
        

if __name__ == "__main__":
    # Mesh ----------------------------------------------------------
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.025
    rectangular_mesh(h = h, L_x = 1, L_y = 1, save_name = mesh_fname)
    
    mesh = CustomTwoDimensionMesh(mesh_fname, reordering= True)
    # Problem parameters --------------------------------------------
    # def diffusion_tensor(x, y):
    #     return np.eye(2)
    
    # def u(x, y):
    #     return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    # def f(x, y):
    #     return (1 + 5 * np.pi ** 2) * u(x, y)
    
    def v(x, y):
        return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 2

    def diffusion_tensor(x, y):
        return v(x, y) * np.eye(2)

    def u(x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def f(x, y): 
        return (1 + 16*(np.pi**2)*(v(x,y) - 1))*u(x,y)
    
    # Problem itself ------------------------------------------------
    dirichlet_pb = PoissonDirichletProblem(mesh=mesh, diffusion_tensor=diffusion_tensor, rhs=f, exact_solution=u)
    # neumann_pb._construct_A()
    dirichlet_pb.solve()
    dirichlet_pb.display()
    dirichlet_pb.display_3d()
    dirichlet_pb.display_error()
    # triangle = mesh.tri_nodes[0]
    # neumann_pb._construct_elementary_rigidity_matrix(triangle=triangle)