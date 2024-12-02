from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Type, Dict, Any
import numpy as np
from abc import ABC

from hfem.core.aliases import ScalarField, TensorField
from hfem.mesh_manager import CustomTwoDimensionMesh
from hfem.core import QuadratureRule, QuadratureFactory

class ProblemType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    CELL = "cell"
    HOMOGENIZED = "homogenized"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_str(cls, value: str) -> 'ProblemType':
        """Convert string to ProblemType for loading from files."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unknown problem type: {value}")

def _validate_diffusion_tensor(diffusion_tensor: TensorField|np.ndarray):
    test_point = (0.0, 0.0)
    test_vector = np.array([1.0, 1.0])
    
    try:
        if isinstance(diffusion_tensor, np.ndarray):
            A = diffusion_tensor
            if A.shape != (2, 2):
                raise ValueError("homogeneous diffusion_tensor should be a 2x2 matrix")
        else:
            A = diffusion_tensor(*test_point)
            if not isinstance(A, np.ndarray) or A.shape != (2, 2):
                raise ValueError("diffusion_tensor must return a 2x2 matrix")
            
        if not np.allclose(A, A.T):
            raise ValueError("diffusion_tensor must be symmetric")
            
        if np.dot(test_vector, A @ test_vector) <= 0:
            raise ValueError("diffusion_tensor might not be coercive")
            
    except Exception as e:
        raise ValueError(f"Invalid diffusion_tensor: {str(e)}")

@dataclass(frozen=True)
class CorePoissonProblemsConfig(ABC):
    """Base configuration for all Poisson problems."""
    mesh: CustomTwoDimensionMesh
    mesh_size: float
    problem_type: ProblemType = field(init=False)
    
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mesh_size': self.mesh_size,
            'problem_type': str(self.problem_type)
        }

@dataclass(frozen=True)
class StandardPoissonConfig(CorePoissonProblemsConfig):
    """Configuration for standard Poisson problems (Dirichlet, Neumann, Periodic)."""
    diffusion_tensor: TensorField
    right_hand_side: ScalarField
    quadrature_rule: Type[QuadratureRule] = field(
        default_factory=lambda: QuadratureFactory.get_quadrature("gauss_legendre_6").get_rule()
    )
    def __post_init__(self):
        _validate_diffusion_tensor(self.diffusion_tensor)

@dataclass(frozen=True)
class DirichletConfig(StandardPoissonConfig):
    """Configuration for homogeneous Dirichlet problem."""
    boundary_condition: ScalarField = field(default=lambda x, y: np.zeros_like(x))

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'problem_type', ProblemType.DIRICHLET)

@dataclass(frozen=True)
class NeumannConfig(StandardPoissonConfig):
    """Configuration for homogeneous Neumann problem."""
    boundary_condition: ScalarField = field(default=lambda x, y: np.zeros_like(x))
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'problem_type', ProblemType.NEUMANN)

@dataclass(frozen=True)
class PeriodicConfig(StandardPoissonConfig):
    """Configuration for periodic boundary conditions."""
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'problem_type', ProblemType.PERIODIC)
        if not self.mesh.is_periodic_compatible():
            raise ValueError("Mesh must be compatible with periodic boundary conditions")

@dataclass(frozen=True)
class CellProblemConfig(CorePoissonProblemsConfig):
    """Configuration for cell problems in homogenization."""
    diffusion_tensor: TensorField
    eta: float  # penalization factor
    
    def __post_init__(self):
        object.__setattr__(self, 'problem_type', ProblemType.CELL)
        if not isinstance(self.eta, (int, float)):
            raise ValueError("eta must be real value")
        if not self.eta > 0:
            raise ValueError("eta should be strictly positive")
        
        _validate_diffusion_tensor(self.diffusion_tensor)

@dataclass(frozen=True)
class HomogenizedConfig(CorePoissonProblemsConfig):
    """Configuration for the homogenized problem."""
    effective_tensor: np.ndarray
    right_hand_side: ScalarField
    
    def __post_init__(self):
        object.__setattr__(self, 'problem_type', ProblemType.HOMOGENIZED)
        _validate_diffusion_tensor(self.effective_tensor)

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'effective_tensor': self.effective_tensor.tolist()
        })
        return base_dict
    
if __name__ == '__main__':
    pass
    # config = DirichletConfig(
    # mesh=mesh,
    # mesh_size=h,
    # diffusion_tensor=diffusion_tensor,
    # right_hand_side=f
    # )

    # # Le type est automatiquement initialisé
    # print(config.problem_type)  # ProblemType.DIRICHLET
    # print(str(config.problem_type))  # "dirichlet"

    # # Pour charger depuis un fichier
    # problem_type = ProblemType.from_str("dirichlet")