from typing import Protocol, Dict, Any, ClassVar, Type

class SolutionProtocol(Protocol):
    """Protocol defining what a solution object should implement."""
    @property
    def problem_type(self) -> str:
        """Type of the problem (e.g. 'cell', 'homogenized')."""
        ...
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary format for storage."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolutionProtocol':
        """Create solution from dictionary format."""
        ...