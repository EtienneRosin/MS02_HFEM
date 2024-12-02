from dataclasses import dataclass, field
from ..problem_type import ProblemType
from ..base import CorePoissonProblemsConfig, _validate_diffusion_tensor
from hfem.core.aliases import TensorField
from typing import Dict, Any
import numpy as np

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
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'eta': self.eta
        })
        return base_dict