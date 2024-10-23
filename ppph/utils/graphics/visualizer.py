from mesh_manager import CustomTwoDimensionMesh
from ppph.utils import TwoDimensionFunction
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import cmasher as cmr

style_path = "ppph/utils/graphics/custom_science.mplstyle"

class Visualizer:
    def __init__(self, mesh: CustomTwoDimensionMesh, field: np.ndarray, field_name: str = None) -> None:
        self.field_name = self._validate_field_name(field_name)
        self.mesh = mesh
        self.field = self._validate_field(field)
        
    def _validate_field_name(self, field_name: str):
        field_name = "" if field_name is None else field_name
        if not isinstance(field_name, str):
            raise ValueError("Field name should be a string.")
        return field_name
    
    def _validate_field(self, field: np.ndarray):
        if field.shape != self.mesh.node_refs.shape:
            raise ValueError(f"Field shape {field.shape} is not compatible with mesh shape {self.mesh.node_refs.shape}.")
        return field
    
    
    
if __name__ == "__main__":
    mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
    h = 0.05
    # create_rectangle_mesh(h = h, L_x = 2, L_y = 2, save_name = mesh_fname)
    mesh = CustomTwoDimensionMesh(filename=mesh_fname, reordering=True)
    
    def u_expr(x, y):
        return np.cos(np.pi * x) * np.cos(np.pi * y)

    u = TwoDimensionFunction(u_expr)
    field = u(mesh.node_coords)
    
    # print(f"{field.shape = }")
    # print(f"{mesh.node_refs.shape = }")
    
    visu = Visualizer(mesh = mesh, field = field, field_name="1")