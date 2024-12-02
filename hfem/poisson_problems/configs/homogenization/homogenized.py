from dataclasses import dataclass, field
from ..problem_type import ProblemType
from ..base import CorePoissonProblemsConfig, _validate_diffusion_tensor
from hfem.core.aliases import ScalarField
from typing import Dict, Any
import numpy as np

@dataclass(frozen=True)
class HomogenizedConfig(CorePoissonProblemsConfig):
    """Configuration for the homogenized problem."""
    effective_tensor: np.ndarray
    right_hand_side: ScalarField
    diffusion_tensor: np.ndarray = field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, 'problem_type', ProblemType.HOMOGENIZED)
        _validate_diffusion_tensor(self.effective_tensor)
        # self.diffusion_tensor = self.effective_tensor
        object.__setattr__(self, 'diffusion_tensor', self.effective_tensor)

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'effective_tensor': self.effective_tensor.tolist()
        })
        return base_dict