from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
from hfem.core.io.protocols import SolutionProtocol

@dataclass
class BaseSolution:
    """Base class for all FEM solutions."""
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'creation_date': datetime.now().isoformat()
    })

    @property
    def problem_type(self) -> str:
        """Must be implemented by concrete solutions."""
        raise NotImplementedError
    


