from dataclasses import dataclass, field
from ..problem_type import ProblemType
from ..base import StandardPoissonConfig
from hfem.core.aliases import ScalarField
import numpy as np


from dataclasses import dataclass, field
from ..problem_type import ProblemType
from ..base import CorePoissonProblemsConfig, _validate_diffusion_tensor
from hfem.core.aliases import TensorField
import numpy as np
from typing import Dict, Any

@dataclass(frozen=True)
class FullDiffusionProblemConfig(CorePoissonProblemsConfig):
    """Configuration for diffusion problem in a material with periodic microstructure."""
    diffusion_tensor: TensorField
    diffusion_tensor_epsilon: TensorField
    right_hand_side: ScalarField
    epsilon: float  # penalization factor
    eta: float
    
    def __post_init__(self):
        object.__setattr__(self, 'problem_type', ProblemType.FULL_DIFFUSION)
        if not isinstance(self.epsilon, (int, float)):
            raise ValueError("epsilon must be real value")
        if not self.epsilon > 0:
            raise ValueError("epsilon should be strictly positive")
        
        if not isinstance(self.eta, (int, float)):
            raise ValueError("eta must be real value")
        if not self.epsilon > 0:
            raise ValueError("eta should be strictly positive")
        
        _validate_diffusion_tensor(self.diffusion_tensor)
        
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'epsilon': self.epsilon,
            'eta': self.eta
        })
        return base_dict