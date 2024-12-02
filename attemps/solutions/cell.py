from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any
from .base import BaseSolution

@dataclass
class CellProblemSolution(BaseSolution):
    """Solution data for cell problems."""
    correctors: List[np.ndarray]
    homogenized_tensor: np.ndarray
    
    @property
    def problem_type(self) -> str:
        return "cell"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'correctors': np.array(self.correctors),  # Convert list to array for HDF5
            'homogenized_tensor': self.homogenized_tensor,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CellProblemSolution':
        return cls(
            correctors=list(data['correctors']),  # Convert back to list
            homogenized_tensor=data['homogenized_tensor'],
            metadata=data.get('metadata', {})  # Use empty dict if no metadata
        )