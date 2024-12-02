from hfem.mesh_manager import CustomTwoDimensionMesh, rectangular_mesh
from hfem.poisson_problems.solvers.homogenization.homogenized import HomogenizedConfig, HomogenizedProblem
from hfem.poisson_problems.solvers.homogenization.diffusion import DiffusionProblem, DiffusionProblemConfig
from hfem.poisson_problems.solvers.homogenization.cell import CellProblem, CellProblemConfig
import numpy as np
from hfem.core.io import FEMDataManager
from multiprocessing import freeze_support
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator


# Parameters --------------------------------------------------------------------
    # general parameters --------------------------------------------
mesh_file = "meshes/rectangle.msh"
cell_problems_mesh_file = "meshes/periodicity_cell.msh"

save_file = "full"
# homogenized_save_file = "full"

    # simulation parameters -----------------------------------------
epsilon = 1/10  # periodicity cell size
# epsilon = 1/100  # periodicity cell size
eta = 1e-5      # penalization factor

    # geometric parameters ------------------------------------------
h = 0.01       # mesh size

if h > epsilon/4:
    raise ValueError("Relation not satisfied : h << ε (here h > ε/4)")

L_x = 2     # rectangle width
L_y = 2     # rectangle height


    # problem parameters --------------------------------------------
def A(x, y):
    """ Y-periodic diffusion tensor used in the cell problems """
    return (2 + np.sin(2 * np.pi * x))*(4 + np.sin(2 * np.pi * y))*np.eye(2)

def A_epsilon(x, y):
    """ εY-periodic diffusion tensor used in the diffusion problem """
    return A(x/epsilon, y/epsilon)

def exact_solution(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)

def right_hand_side(x, y):
    return -(2*np.pi**2/epsilon)*(-epsilon*(np.sin(2*np.pi*x/epsilon) + 2)*(np.sin(2*np.pi*y/epsilon) + 4)*exact_solution(x,y) \
        + (np.sin(2*np.pi*x/epsilon) + 2)*np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(2*np.pi*y/epsilon) \
        + (np.sin(2*np.pi*y/epsilon) + 4)*np.sin(np.pi*y)*np.cos(np.pi*x)*np.cos(2*np.pi*x/epsilon))

# def right_hand_side(x, y):
#     return -(2*np.pi**2)*(-(np.sin(2*np.pi*x) + 2)*(np.sin(2*np.pi*y) + 4)*exact_solution(x,y) \
#         + (np.sin(2*np.pi*x) + 2)*np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(2*np.pi*y) \
#         + (np.sin(2*np.pi*y) + 4)*np.sin(np.pi*y)*np.cos(np.pi*x)*np.cos(2*np.pi*x))

effective_tensor = np.array([
    [4*np.sqrt(3), 0],
    [0, 2*np.sqrt(15)]
])


def main():
    manager = FEMDataManager()
    
    try:
        rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
        problem_mesh = CustomTwoDimensionMesh(filename=mesh_file)
        # 1 - Cell problems ---------------------------------------------------
        
        # Check if the problem has already been solved
        cell_problem_save_file = f"simulation_data/cell/{save_file}.h5"
        compute_cell_problems = True
        if Path(cell_problem_save_file).is_file():
            cell_problems_solution, cell_problems_mesh, _ = manager.load(cell_problem_save_file)
            if cell_problems_solution.metadata['problem_params']['eta'] == eta and cell_problems_solution.metadata['problem_params']['mesh_size'] == h:
                compute_new_cell_problems = input('Cell problems already computed. Compute again [y/n]:')
                compute_cell_problems = True if compute_new_cell_problems == 'y' else False
        
        if compute_cell_problems:
            print("Computing cell problems --------------------------")
            
            rectangular_mesh(h=h, L_x=1, L_y=1, save_name=cell_problems_mesh_file)
            cell_problem_mesh = CustomTwoDimensionMesh(filename=cell_problems_mesh_file)
            
            cell_problem_config = CellProblemConfig(
                mesh=cell_problem_mesh,
                mesh_size=h,
                diffusion_tensor=A,
                eta=eta
            )
            cell_problem = CellProblem(config=cell_problem_config)
            cell_problem.solve_and_save(save_name=save_file)
            cell_problems_solution, cell_problems_mesh, _ = manager.load(cell_problem_save_file)
            print("Computed cell problems ---------------------------")
            

        # 2 - Homogenized problem ---------------------------------------------
        homogenized_problem_save_file = f"simulation_data/homogenized/{save_file}.h5"
        compute_homogenized_problem = True
        if Path(homogenized_problem_save_file).is_file():
            homogenized_solution, _, homogenized_matrices = manager.load(homogenized_problem_save_file)  # Définir la variable ici
            if homogenized_solution.metadata['problem_params']['mesh_size'] == h:
                compute_homogenized_problem = input('Homogenized problem saved file exists. Be aware that the diffusion tensor might not be the same. Compute again [y/n]:')
                compute_homogenized_problem = True if compute_homogenized_problem == 'y' else False

        if compute_homogenized_problem:
            print("Computing Homogenized problem --------------------")
            
            homogenized_config = HomogenizedConfig(
                mesh=problem_mesh,
                mesh_size=h,
                effective_tensor=cell_problems_solution.data['homogenized_tensor'],
                right_hand_side=right_hand_side
            )
            homogenized_problem = HomogenizedProblem(config=homogenized_config)
            homogenized_problem.solve_and_save(save_name=save_file)
            
            homogenized_solution, _, homogenized_matrices = manager.load(homogenized_problem_save_file)
            print("Computed Homogenized problem ---------------------")
        
        
        # 3 - Diffusion problem -----------------------------------------------
        diffusion_problem_save_file = f"simulation_data/diffusion/{save_file}.h5"
        diffusion_config = DiffusionProblemConfig(
            mesh=problem_mesh,
            mesh_size=h,
            epsilon=epsilon,
            diffusion_tensor=A_epsilon,
            right_hand_side=right_hand_side
        )
        diffusion_problem = DiffusionProblem(config=diffusion_config)
        diffusion_problem.solve_and_save(save_file)
        
        # homogenized_data = homogenized_solution.data.copy() 
        diffusion_solution, diffusion_mesh, diffusion_matrices = manager.load(diffusion_problem_save_file)
        
        
        # 4 - Interpolation of the correctors ---------------------------------
        # print(homogenized_solution.data['corrector_x'])
        # print(cell_problems_solution.data['corrector_x'])
        
        
        interpolators = [
            LinearNDInterpolator(cell_problems_mesh.nodes,
                                 cell_problems_solution.data['corrector_x'],
                                 fill_value=0),
            LinearNDInterpolator(cell_problems_mesh.nodes,
                                 cell_problems_solution.data['corrector_y'],
                                 fill_value=0),
        ]
        # print("dzcubcdbzobcuobuybèyo")
        def get_cell_coordinates(x1: np.ndarray, x2: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
            """Retourne les coordonnées dans la cellule unité ]0,1[²."""
            y1 = (x1/epsilon) % 1
            y2 = (x2/epsilon) % 1
            return y1, y2
        
        X, Y = get_cell_coordinates(*diffusion_mesh.nodes.T, epsilon)
        points = np.column_stack([X.ravel(), Y.ravel()])
        interpolated_correctors = [interpolator(points).reshape(X.shape) for interpolator in interpolators]        
        
        # 5 - Display and error computation -----------------------------------
        u_1 = homogenized_solution.data['x_derivative']*interpolated_correctors[0] + homogenized_solution.data['y_derivative']*interpolated_correctors[1]
        u_0 = homogenized_solution.data['solution']
        u_epsilon = diffusion_solution.data
        
        M = homogenized_matrices.mass_matrix
        K = homogenized_matrices.stiffness_matrix
        
        l2_error = np.sqrt((u_epsilon - u_0).T @ M @ (u_epsilon - u_0))
        h1_error = np.sqrt((u_epsilon - u_0).T @ K @ (u_epsilon - u_0))
        h1_error_bis = np.sqrt((u_epsilon - u_0 - epsilon*u_1).T @ K @ (u_epsilon - u_0 - epsilon*u_1))
        print(f"{l2_error = }, {h1_error = }, {h1_error_bis = }")
        
        # print(diffusion_solution.data)
        diffusion_mesh.display_field(
            field=u_epsilon,
            field_label=r"$u_\varepsilon(\boldsymbol{x})$"
        )
        diffusion_mesh.display_field(
            field=u_0,
            field_label=r"$u_0(\boldsymbol{x})$"
        )
        
        
        # diffusion_mesh.display_field(
        #     field=epsilon*u_1,
        #     field_label=r"$\varepsilon u_1(\boldsymbol{x}/\varepsilon)$"
        # )
        
        # print(f"{(u_epsilon - u_0).min() = }, {(u_epsilon - u_0).max() = }")
        diffusion_mesh.display_field(
            field=u_epsilon - u_0,
            field_label=r"$u_\varepsilon(\boldsymbol{x}) - u_0(\boldsymbol{x})$",
            # kind='trisurface'
        )
        diffusion_mesh.display_field(
            field=u_epsilon - u_0,
            field_label=r"$u_\varepsilon(\boldsymbol{x}) - u_0(\boldsymbol{x})$",
            kind='trisurface'
        )
        
        diffusion_mesh.display_field(
            field=u_epsilon - u_0 - epsilon*u_1,
            field_label=r"$u_\varepsilon(\boldsymbol{x}) - u_0(\boldsymbol{x}) - \varepsilon u_1(\boldsymbol{x}/\varepsilon)$"
        )
        diffusion_mesh.display_field(
            field=u_epsilon - u_0 - epsilon*u_1,
            field_label=r"$u_\varepsilon(\boldsymbol{x}) - u_0(\boldsymbol{x}) - \varepsilon u_1(\boldsymbol{x}/\varepsilon)$",
            kind='trisurface'
        )
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == '__main__':
    freeze_support()
    main()