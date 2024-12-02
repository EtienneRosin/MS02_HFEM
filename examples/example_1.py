from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
from hfem.poisson_problems.configs.standard.periodic import PeriodicConfig
from hfem.poisson_problems.solvers.standard.periodic import PeriodicProblem
import numpy as np

def main():
    # 1. Paramètres du problème
    h = 0.01     # taille du maillage
    L_x = L_y = 1  # dimensions du rectangle
    
    # 2. Définition des fonctions du problème
    def diffusion_tensor(x, y):
        """Tenseur de diffusion anisotrope et périodique."""
        return np.array([
            [2 + np.sin(2*np.pi*x)*np.sin(2*np.pi*y), 0],
            [0, 2 + np.sin(2*np.pi*x)*np.sin(2*np.pi*y)]
        ])
    
    def right_hand_side(x, y):
        """Second membre."""
        return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    
    # 3. Création du maillage périodique
    mesh_file = "meshes/periodic_square.msh"
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # 4. Configuration du problème
    config = PeriodicConfig(
        mesh=mesh,
        mesh_size=h,
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side
    )
    
    # 5. Création et résolution du problème
    problem = PeriodicProblem(config=config)
    problem.solve_and_save(save_name="results/periodic_test")
    
    # 6. Chargement et visualisation des résultats
    from hfem.core.io import FEMDataManager
    
    manager = FEMDataManager()
    solution, mesh, matrices = manager.load(
        f"simulation_data/{str(config.problem_type)}/results/periodic_test.h5"
    )
    
    # Affichage de la solution
    mesh.display_field(solution.data)

if __name__ == "__main__":
    main()