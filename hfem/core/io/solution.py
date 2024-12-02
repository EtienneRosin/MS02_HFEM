from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import numpy as np
from datetime import datetime

@dataclass
class Solution:
    """Unified solution class with special handling for cell problems."""
    data: Union[np.ndarray, Dict[str, np.ndarray]]  # Solution data or list of correctors for cell problems
    problem_type: str  # 'neumann', 'dirichlet', 'periodic', 'cell', 'homogenized', 'base_microstructured'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                'creation_date': datetime.now().isoformat()
            }
            
        # Special handling for cell problems
        if self.problem_type == 'cell':
            if not isinstance(self.data, dict):
                raise ValueError("Cell problem requires dict containing correctors and homogenized tensor")
            required_keys = {'corrector_x', 'corrector_y', 'homogenized_tensor'}
            if not all(key in self.data for key in required_keys):
                raise ValueError(f"Cell problem data must contain keys: {required_keys}")
            if not isinstance(self.data['homogenized_tensor'], np.ndarray) or self.data['homogenized_tensor'].shape != (2, 2):
                raise ValueError("homogenized_tensor must be a 2x2 numpy array")

    def get_correctors(self) -> Optional[Dict[str, np.ndarray]]:
        """Get correctors for cell problems."""
        if self.problem_type == 'cell':
            return {
                'x': self.data['corrector_x'],
                'y': self.data['corrector_y']
            }
        return None
    
    def get_homogenized_tensor(self) -> Optional[np.ndarray]:
        """Get homogenized tensor for cell problems."""
        if self.problem_type == 'cell':
            return self.data['homogenized_tensor']
        return None
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        if self.problem_type == 'cell':
            # Handles the case where data is a list of correctors
            return {
                'correctors': np.array(self.data),  # Convert list to array for HDF5 storage
                'homogenized_tensor': self.metadata['homogenized_tensor'],
                'problem_type': self.problem_type,
                'metadata': {k: v for k, v in self.metadata.items() 
                           if k != 'homogenized_tensor'}  # Avoid duplication
            }
        return {
            'data': self.data,
            'problem_type': self.problem_type,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Solution':
        """Create from dictionary format."""
        if 'correctors' in data:  # Cell problem case
            return cls(
                data=list(data['correctors']),  # Convert back to list
                problem_type='cell',
                metadata={
                    'homogenized_tensor': data['homogenized_tensor'],
                    **(data.get('metadata', {}))
                }
            )
        return cls(
            data=data['data'],
            problem_type=data['problem_type'],
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def create_cell_solution(cls, 
                           correctors: List[np.ndarray],
                           homogenized_tensor: np.ndarray,
                           extra_metadata: Optional[Dict[str, Any]] = None) -> 'Solution':
        """Convenience method to create a cell problem solution."""
        metadata = {'homogenized_tensor': homogenized_tensor}
        if extra_metadata:
            metadata.update(extra_metadata)
        return cls(data=correctors, problem_type='cell', metadata=metadata)