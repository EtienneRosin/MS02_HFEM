from enum import Enum

class ProblemType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    CELL = "cell"
    HOMOGENIZED = "homogenized"
    DIFFUSION = "diffusion"
    FULL_DIFFUSION = "full_diffusion"
    HOMOGENIZATION_ANALYSIS = 'homogenization_analysis'
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_str(cls, value: str) -> 'ProblemType':
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unknown problem type: {value}")