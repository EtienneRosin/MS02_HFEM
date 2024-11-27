

from hfem.core import BasePoissonProblem, BasePoissonConfig
from mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh


class HomogenousNeumannPoissonProblem(BasePoissonProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, config: BasePoissonConfig):
        super().__init__(mesh, config)