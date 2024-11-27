import scipy as sp
import numpy as np


from hfem.core import BasePoissonProblem, BasePoissonConfig
from mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh


class HomogenousDirichletPoissonProblem(BasePoissonProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, config: BasePoissonConfig):
        super().__init__(mesh, config)
        
    def _construct_P_0(self):
        """Construct the projection matrix P for Dirichlet boundary conditions."""
        on_border_ref = self.mesh.labels['$\\partial\\Omega$']
        interior_indices = np.where(self.mesh.node_refs != on_border_ref)[0]
        N_0 = len(interior_indices)
        N = self.mesh.num_nodes

        P = sp.sparse.lil_matrix((N_0, N), dtype=float)
        for i, j in enumerate(interior_indices):
            P[i, j] = 1

        # print(P.toarray())
        return P.tocsr()
    
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
        # A_0 = P_0 @ (self.M + self.K) @ P_0.T
        A_0 = P_0 @ A @ P_0.T
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
    
    

def test_homogeneous_dirichlet_poisson_problem():
    """Test the Poisson solver with a simple case."""
    import matplotlib.pyplot as plt
    from hfem.viz import FEMVisualizer, solution_config, error_config, ErrorType, VisualizationConfig
    
    # Create a test mesh
    mesh_file = "meshes/test_mesh.msh"
    from mesh_manager.geometries import rectangular_mesh
    rectangular_mesh(h=0.025, L_x=1, L_y=1, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    
    def v(x, y):
        return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 2

    def diffusion_tensor(x, y):
        return v(x, y) * np.eye(2)

    def exact_solution(x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def right_hand_side(x, y): 
        return (1 + 16*(np.pi**2)*(v(x,y) - 1))*exact_solution(x,y)
    
    
    # Create configuration
    config = BasePoissonConfig(
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side,
        exact_solution=exact_solution
    )
    
    # Create and solve
    solver = HomogenousDirichletPoissonProblem(mesh, config)
    solution = solver.solve()
    
    solver.display_solution(
        solution_config(
            kind='trisurface',
            # title='Numerical Solution',
            # cmap='viridis',
            cbar=True,
            save_name = 'hihi.pdf'
        )
    )
    plt.show()
    # Visualize error
    solver.display_error(
        error_config(
            kind='contourf',
            error_type=ErrorType.ABSOLUTE,
            cbar = True,
            save_name = 'czibzic.pdf'
            # title='Absolute Error',
            # cmap='RdBu_r'
        )
    )
    plt.show()
    
    solver.display_error(
        error_config(
            kind='trisurface',
            error_type=ErrorType.ABSOLUTE,
            # title='Absolute Error',
            # cmap='RdBu_r',
            cbar = False
        )
    )
    plt.show()
    return solver

if __name__ == "__main__":
    solver = test_homogeneous_dirichlet_poisson_problem()