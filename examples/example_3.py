from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
from hfem.poisson_problems.solvers.homogenization.homogenized import HomogenizedConfig, HomogenizedProblem
import numpy as np

def exact_solution(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)
    
def right_hand_side(x, y):
    return -2*(np.pi**2)*(-np.sqrt(15) - 2*np.sqrt(3))*exact_solution(x,y)
    
effective_tensor = np.array([
    [4*np.sqrt(3), 0],
    [0, 2*np.sqrt(15)]
])

def exact_solution_derivatives(x, y):
    # ∂x(sin(πx)sin(πy)) = πcos(πx)sin(πy)
    dx = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
    # ∂y(sin(πx)sin(πy)) = sin(πx)πcos(πy)
    dy = np.sin(np.pi*x) * np.pi * np.cos(np.pi*y)
    return dx, dy

def main():
    # 1. Paramètres du problème 
    h = 0.0125     # taille du maillage
    # h = 0.1     # taille du maillage
    L_x = L_y = 2  # dimensions du rectangle
    
    # 2. Définition des fonctions du problème
    
    # def right_hand_side(x, y):
    #     """Second membre."""
    #     return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    
    # 3. Création du maillage périodique
    mesh_file = "meshes/periodic_square.msh"
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # 4. Configuration du problème
    config = HomogenizedConfig(
        mesh=mesh,
        mesh_size=h,
        effective_tensor=effective_tensor,
        right_hand_side=right_hand_side
    )
    
    
    # 5. Création et résolution du problème
    problem = HomogenizedProblem(config=config)
    # problem.solve_and_save(save_name="homogenized_test")
    
    # 6. Chargement et visualisation des résultats
    from hfem.core.io import FEMDataManager
    
    manager = FEMDataManager()
    solution, mesh, matrices = manager.load(
        f"simulation_data/{str(config.problem_type)}/homogenized_test.h5"
    )
    print(f"{matrices}")
    
    import cmasher as cmr
    # Affichage de la solution
    
    mesh.display_field(
        field=solution.data['solution'], 
        field_label=r'$u_0(\boldsymbol{x})$',
        cmap=cmr.lavender, 
        kind = 'tricontour',
        # save_name= "dizbcduz"
        # cbar_props = {'label': r'$\omega_i$'}
        )
    
    # mesh.display_field(
    #     field=solution.data['x_derivative'], 
    #     field_label=r'$\partial_x u_0(\boldsymbol{x})$',
    #     cmap=cmr.lavender, 
    #     kind = 'tricontour',
    #     # save_name= "dizbcduz"
    #     # cbar_props = {'label': r'$\omega_i$'}
    #     )
    
    # mesh.display_field(
    #     field=solution.data['y_derivative'], 
    #     field_label=r'$\partial_y u_0(\boldsymbol{x})$',
    #     cmap=cmr.lavender, 
    #     kind = 'trisurface',
    #     # save_name= "dizbcduz"
    #     # cbar_props = {'label': r'$\omega_i$'}
    #     )
    
    
    dx_exact, dy_exact = exact_solution_derivatives(*mesh.nodes.T)
    
    # Visualisation des erreurs
    mesh.display_field(
        field=np.abs(solution.data['x_derivative'] - dx_exact),
        field_label=r'$|\partial_x u_{0,h} - \partial_x u|$',
        cmap=cmr.lavender
    )
    
    mesh.display_field(
        field=np.abs(solution.data['y_derivative'] - dy_exact),
        field_label=r'$|\partial_y u_{0,h} - \partial_y u|$',
        cmap=cmr.lavender
    )
    
    
    # mesh.display_field(
    #     field=solution.data['corrector_y'], 
    #     field_label=r'$\omega_2^\eta(\boldsymbol{y})$',
    #     cmap=cmr.lavender, 
    #     # kind = 'trisurface'
    #     )
    
    # mesh.display_correctors(
    #     correctors={
    #         'corrector_x': solution.data['x_derivative'],
    #         'corrector_y': solution.data['y_derivative']
    #     },
    #     field_label=r'$\omega_i^\eta(\boldsymbol{y})$',
    #     cmap=cmr.lavender,
    #     # save_name="correctors_comparison"
    # )

if __name__ == "__main__":
    main()