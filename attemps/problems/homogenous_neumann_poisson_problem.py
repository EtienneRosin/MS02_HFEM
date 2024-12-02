

# from hfem.core import BasePoissonProblem, BasePoissonConfig
from hfem.mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh
from hfem.problems import BasePoissonConfig, BasePoissonProblem

class HomogenousNeumannPoissonProblem(BasePoissonProblem):
    def __init__(self, mesh: CustomTwoDimensionMesh, config: BasePoissonConfig):
        super().__init__(mesh, config)