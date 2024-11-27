
from hfem.problems import HomogenousNeumannPoissonProblem
from hfem.core import BasePoissonConfig

from mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh
from mesh_manager.geometries import rectangular_mesh
from hfem.viz import solution_config, error_config, ErrorType

import numpy as np
import matplotlib.pyplot as plt

# Problem configuration -----------------------------------

    # Geometry ----------------------------------
mesh_file = "meshes/rectangle_mesh.msh"
# rectangular geometry
h = 0.01    # mesh size
L_x = 1     # rectangle width
L_y = 1     # rectangle height

    # Parameters --------------------------------
def v(x,y):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y) + 2

def diffusion_tensor(x, y):
            return np.eye(2)*v(x, y)
        
def exact_solution(x, y):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

def right_hand_side(x, y):
    return (1 + (16*np.pi**2)*(v(x, y)- 1))*exact_solution(x, y)

pb_config = BasePoissonConfig(
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side,
        exact_solution=exact_solution
    )

if __name__ == '__main__':
    # Create the mesh
    rectangular_mesh(h=h, L_x=L_x, L_y=L_y, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    
    # Instanciate the problem
    neumann_pb = HomogenousNeumannPoissonProblem(mesh, pb_config)

    # Solve the problem
    neumann_pb.solve()
    
    # Display solution (and error if exact solution is provided)
    neumann_pb.display_solution(
        solution_config(
            kind='trisurface',
            # save_name = f'study_cases/I_Neumann/neumann_solution.pdf'
        )
    )
    plt.show()
    
    neumann_pb.display_error(
        error_config(
            kind='contourf',
            error_type=ErrorType.ABSOLUTE,
            cbar = True,
            # save_name = f'study_cases/I_Neumann/neumann_absolute_error.pdf'
        )
    )
    plt.show()
    
    neumann_pb.display_error(
        error_config(
            kind='trisurface',
            error_type=ErrorType.ABSOLUTE,
            cbar = False
        )
    )
    plt.show()