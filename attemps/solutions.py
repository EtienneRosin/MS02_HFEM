from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from hfem.core.io.data_structures import MeshData

@dataclass
class StandardPoissonSolution:
    solution: np.ndarray
    boundary_type: str  # 'neumann', 'dirichlet', 'periodic'
    
    @property
    def problem_type(self) -> str:
        return self.boundary_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'solution': self.solution,
            'boundary_type': self.boundary_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardPoissonSolution':
        """Crée une solution à partir d'un dictionnaire."""
        return cls(
            solution=data['solution'],
            boundary_type=data['boundary_type']
        )
    
class CellProblemSolution:
    """Solution pour problème de cellule."""
    correctors: List[np.ndarray]
    homogenized_tensor: np.ndarray
    eta: float  # paramètre de pénalisation
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'creation_date': datetime.now().isoformat()
    })
    
    @property
    def problem_type(self) -> str:
        return "cell"
    
    @classmethod
    def to_dict(self) -> Dict[str, Any]:
        return {
            'correctors': np.array(self.correctors),  # Convert list to array for HDF5
            'homogenized_tensor': self.homogenized_tensor,
            'metadata': self.metadata
        }


class HomogenizedSolution:
    """Solution pour problème homogénéisé."""
    solution: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'creation_date': datetime.now().isoformat()
    })
    @property 
    def problem_type(self) -> str:
        return "homogenized"
    
    @classmethod
    def to_dict(self) -> Dict[str, Any]:
        return {
            'solution': self.solution,
            'metadata': self.metadata
        }