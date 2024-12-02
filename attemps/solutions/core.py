from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseSolution


class CorePoissonProblemSolution(BaseSolution):
    """Solution data for core Poisson problems."""
    solution: np.ndarray
    
    @property
    def problem_type(self) -> str:
        return "homogenized"
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'solution': self.solution,
            'metadata': self.metadata
        }
        if self.gradient is not None:
            data['gradient'] = self.gradient
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorePoissonProblemSolution':
        return cls(
            solution=data['solution'],
            gradient=data.get('gradient'),  # None if not present
            metadata=data.get('metadata', {})
        )