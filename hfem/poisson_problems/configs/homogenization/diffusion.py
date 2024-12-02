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

# @dataclass(frozen=True)
# class DiffusionProblemConfig(CorePoissonProblemsConfig):
#     """Configuration for diffusion problem in a material with periodic microstructure."""
#     diffusion_tensor: TensorField
#     right_hand_side: ScalarField
#     epsilon: float  # penalization factor
    
#     def __post_init__(self):
#         object.__setattr__(self, 'problem_type', ProblemType.DIFFUSION)
#         if not isinstance(self.epsilon, (int, float)):
#             raise ValueError("epsilon must be real value")
#         if not self.epsilon > 0:
#             raise ValueError("epsilon should be strictly positive")
        
#         _validate_diffusion_tensor(self.diffusion_tensor)
#         original_tensor = self.diffusion_tensor
#         object.__setattr__(self, 'diffusion_tensor', 
#                           lambda x,y : original_tensor(x/self.epsilon, y/self.epsilon))
        
    
#     def to_dict(self) -> Dict[str, Any]:
#         base_dict = super().to_dict()
#         base_dict.update({
#             'epsilon': self.epsilon
#         })
#         return base_dict

@dataclass(frozen=True)
class DiffusionProblemConfig(CorePoissonProblemsConfig):
    """Configuration for diffusion problem in a material with periodic microstructure."""
    diffusion_tensor: TensorField
    right_hand_side: ScalarField
    epsilon: float  # penalization factor
    
    def __post_init__(self):
        object.__setattr__(self, 'problem_type', ProblemType.DIFFUSION)
        if not isinstance(self.epsilon, (int, float)):
            raise ValueError("epsilon must be real value")
        if not self.epsilon > 0:
            raise ValueError("epsilon should be strictly positive")
        
        _validate_diffusion_tensor(self.diffusion_tensor)
        
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'epsilon': self.epsilon
        })
        return base_dict