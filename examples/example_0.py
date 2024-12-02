from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh

from hfem.poisson_problems.solvers.standard import NeumannHomogeneousProblem
from hfem.poisson_problems.configs.standard import NeumannConfig
import numpy as np

# 1. Paramètres du problème
h = 0.01     # taille du maillage
L_x = L_y = 1  # dimensions du rectangle
a = 32
# 2. Définition des fonctions du problème
def v(x,y):
        return np.cos(a*np.pi*x)*np.cos(a*np.pi*y) + 2
def diffusion_tensor(x, y):
    """Tenseur de diffusion anisotrope et périodique."""
    return v(x,y)*np.eye(2)

def right_hand_side(x, y):
    """Second membre."""
    return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

def main():
    # 3. Création du maillage périodique
    mesh_file = "meshes/periodic_square.msh"
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # 4. Configuration du problème
    config = NeumannConfig(
        mesh=mesh,
        mesh_size=h,
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side
    )
    
    # 5. Création et résolution du problème
    problem = NeumannHomogeneousProblem(config=config)
    problem.solve_and_save(save_name=f"test_a_{a}")
    
    # 6. Chargement et visualisation des résultats
    from hfem.core.io import FEMDataManager
    
    manager = FEMDataManager()
    solution, mesh, matrices = manager.load(
        f"simulation_data/neumann/test_a_{a}.h5"
    )
    
    # Affichage de la solution
    mesh.display_field(
        field=solution.data,
        field_label=r"$u_h$",
        save_name=f"neumann_a_{a}"
        )

if __name__ == "__main__":
    main()