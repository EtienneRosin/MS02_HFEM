from dataclasses import dataclass, field
from ..problem_type import ProblemType
from ..base import StandardPoissonConfig
from hfem.core.aliases import ScalarField
import numpy as np

@dataclass(frozen=True)
class PeriodicConfig(StandardPoissonConfig):
    """Configuration for periodic boundary conditions."""
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'problem_type', ProblemType.PERIODIC)
        # if not self.mesh.is_periodic_compatible():
        #     raise ValueError("Mesh must be compatible with periodic boundary conditions")