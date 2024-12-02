from dataclasses import dataclass, field
from typing import Type, Dict, Any
import numpy as np
from abc import ABC

from hfem.core.aliases import ScalarField, TensorField
from hfem.mesh_manager import CustomTwoDimensionMesh
from hfem.core import QuadratureRule, QuadratureFactory
from .problem_type import ProblemType

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
    diffusion_tensor: TensorField
    right_hand_side: ScalarField
    quadrature_rule: Type[QuadratureRule] = field(
        default_factory=lambda: QuadratureFactory.get_quadrature("gauss_legendre_6").get_rule()
    )
    
    def __post_init__(self):
        _validate_diffusion_tensor(self.diffusion_tensor)