from dataclasses import dataclass, field
from ..problem_type import ProblemType
from ..base import StandardPoissonConfig
from hfem.core.aliases import ScalarField
import numpy as np

@dataclass(frozen=True)
class NeumannConfig(StandardPoissonConfig):
    """Configuration for homogeneous Neumann problem."""
    boundary_condition: ScalarField = field(default=lambda x, y: np.zeros_like(x))
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'problem_type', ProblemType.NEUMANN)