import scipy as sp
import numpy as np


from hfem.problems import BasePoissonProblem, BasePoissonConfig
from hfem.mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh
from hfem.core.related_matrices import assemble_P_per

class PeriodicPoissonProblem(BasePoissonProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, config: BasePoissonConfig):
        super().__init__(mesh, config)
        
    # def _construct_P_per(self, boundary_labels: str | list[str] = '$\\partial\\Omega$', tolerance: float = 1e-10):
    #     """Construct the projection matrix P_per for periodic boundary conditions."""
    #     if isinstance(boundary_labels, str):
    #         boundary_labels = [boundary_labels]

    #     corner_indices, pairs_same_x, pairs_same_y, inner_indices = self.mesh.get_corner_and_boundary_pairs(boundary_labels, tolerance)
        
    #     # Initialize the projector matrix P_per
    #     N_inner = len(inner_indices)
    #     N_corner = 1  # we keep only one corner
    #     N_pairs_x = len(pairs_same_x)
    #     N_pairs_y = len(pairs_same_y)
    #     N_per = N_inner + N_corner + N_pairs_x + N_pairs_y
    #     P_per = sp.sparse.lil_matrix((N_per, self.mesh.num_nodes))

    #     # Assign values to P_per based on inner indices, corner indices, and pairs of indices
    #     for n, i in enumerate(inner_indices):
    #         P_per[n, i] = 1

    #     for i in corner_indices:
    #         P_per[N_inner, i] = 1

    #     for n, (i, j) in enumerate(pairs_same_x):
    #         P_per[n + N_inner + N_corner, i] = P_per[n + N_inner + N_corner, j] = 1

    #     for n, (i, j) in enumerate(pairs_same_y):
    #         P_per[n + N_inner + N_corner + N_pairs_x, i] = P_per[n + N_inner + N_corner + N_pairs_x, j] = 1

    #     return P_per.tocsr()
    
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
        # P_per = self._construct_P_per()
        P_per = assemble_P_per(self.mesh)
        # Project onto V_h^0
        # A_0 = P_per @ (self.M + self.K) @ P_per.T
        A_0 = P_per @ A @ P_per.T
        L_0 = P_per @ L
        
        # Solve system
        U_0 = sp.sparse.linalg.spsolve(A_0, L_0)
        
        # Get back to V_h
        self.solution = P_per.T @ U_0
        # Compute errors if exact solution available
        if self.config.exact_solution is not None:
            # print(f"sol calculée, on calcule l'erreur")
            self._compute_errors()
        
        return self.solution
    
    

def test_periodic_poisson_problem():
    """Test the Poisson solver with a simple case."""
    import matplotlib.pyplot as plt
    from hfem.viz import FEMVisualizer, solution_config, error_config, ErrorType, VisualizationConfig
    
    # Create a test mesh
    mesh_file = "meshes/test_mesh.msh"
    from hfem.mesh_manager.geometries import rectangular_mesh
    rectangular_mesh(h=0.01, L_x=1, L_y=1, save_name=mesh_file)
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
    solver = PeriodicPoissonProblem(mesh, config)
    solution = solver.solve()
    
    solver.display_solution(
        solution_config(
            kind='trisurface',
            # title='Numerical Solution',
            # cmap='viridis',
            cbar=True,
            # save_name = 'hihi.pdf'
        )
    )
    plt.show()
    # Visualize error
    solver.display_error(
        error_config(
            kind='contourf',
            error_type=ErrorType.ABSOLUTE,
            cbar = True,
            # save_name = 'czibzic.pdf'
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
    solver = test_periodic_poisson_problem()