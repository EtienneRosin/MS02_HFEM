from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
from hfem.poisson_problems.configs.homogenization.cell import CellProblemConfig
from hfem.poisson_problems.solvers.homogenization.cell import CellProblem
import numpy as np


def diffusion_tensor(x, y):
    """Tenseur de diffusion anisotrope et périodique."""
    return np.array([
        [2 + np.sin(2*np.pi*x), 0],
        [0, 4 + np.sin(2*np.pi*y)]
    ])

def main():
    # 1. Paramètres du problème 
    h = 0.0075     # taille du maillage
    L_x = L_y = 1  # dimensions du rectangle
    
    # 2. Définition des fonctions du problème
    
    # def right_hand_side(x, y):
    #     """Second membre."""
    #     return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    
    # 3. Création du maillage périodique
    mesh_file = "meshes/periodic_square.msh"
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # 4. Configuration du problème
    config = CellProblemConfig(
        mesh=mesh,
        mesh_size=h,
        eta = 1e-5,
        diffusion_tensor=diffusion_tensor,
        # right_hand_side=right_hand_side
    )
    
    from datetime import datetime
    
    # 5. Création et résolution du problème
    problem = CellProblem(config=config)
    # problem.solve_and_save(save_name="cell_test")
    
    # 6. Chargement et visualisation des résultats
    from hfem.core.io import FEMDataManager
    
    manager = FEMDataManager()
    solution, mesh, matrices = manager.load(
        f"simulation_data/{str(config.problem_type)}/cell_test.h5"
    )
    
    import cmasher as cmr
    
    print(f"{solution.data['corrector_x']}")
    # Affichage de la solution
    # mesh.display_field(
    #     field=solution.data['corrector_x'], 
    #     field_label=r'$\omega_1^\eta(\boldsymbol{y})$',
    #     cmap=cmr.lavender, 
    #     kind = 'tricontour',
    #     # save_name= "dizbcduz"
    #     # cbar_props = {'label': r'$\omega_i$'}
    #     )
    # mesh.display_field(
    #     field=solution.data['corrector_y'], 
    #     field_label=r'$\omega_2^\eta(\boldsymbol{y})$',
    #     cmap=cmr.lavender, 
    #     # kind = 'trisurface'
    #     )
    
    mesh.display_correctors(
        correctors={
            'corrector_x': solution.data['corrector_x'],
            'corrector_y': solution.data['corrector_y']
        },
        field_label=r'$\omega_i^\eta(\boldsymbol{y})$',
        cmap=cmr.lavender,
        save_name="correctors_comparison"
    )

if __name__ == "__main__":
    main()