"""
Configuration module for Homogenization and Poisson Problems.

This module provides configuration classes for different types of Poisson problems:
- Homogeneous Neumann boundary conditions
- Homogeneous Dirichlet boundary conditions 
- Periodic boundary conditions
- Homogenization with microstructure
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Type
import numpy as np
from numpy.typing import NDArray

from hfem.core.quadratures import (
    QuadratureStrategy,
    QuadratureFactory
)

# Type aliases
ScalarField = Callable[[float, float], float]
TensorField = Callable[[float, float], NDArray[np.float64]]

@dataclass(frozen=True)
class BasePoissonConfig:
    """
    Base configuration for all Poisson problems.

    Attributes
    ----------
    diffusion_tensor : TensorField
        Function A(x,y) that must satisfy:
        1. Uniform boundedness: ∃C>0, ∀(x,y)∈Ω, ∀i,j, |Aij(x,y)| ≤ C
        2. Uniform coercivity: ∃c>0, ∀(x,y)∈Ω, ∀ξ∈R², A(x,y)ξ·ξ ≥ c|ξ|²
    
    right_hand_side : ScalarField
        Source term f ∈ L²(Ω)
    
    exact_solution : Optional[ScalarField]
        Exact solution if known, for validation
    
    quadrature_strategy : Type[QuadratureStrategy]
        Numerical integration strategy
    """
    
    diffusion_tensor: TensorField
    right_hand_side: ScalarField
    exact_solution: Optional[ScalarField] = None
    quadrature_strategy: Type[QuadratureStrategy] = field(
        default_factory=lambda: QuadratureFactory.get_quadrature("gauss_legendre_6")
    )

    def __post_init__(self):
        """Validate tensor properties."""
        test_point = (0.0, 0.0)
        test_vector = np.array([1.0, 1.0])
        
        # Test diffusion tensor
        try:
            A = self.diffusion_tensor(*test_point)
            if not isinstance(A, np.ndarray) or A.shape != (2, 2):
                raise ValueError("diffusion_tensor must return a 2x2 matrix")
            
            # Test symmetry
            if not np.allclose(A, A.T):
                raise ValueError("diffusion_tensor must be symmetric")
            
            # Test coercivity (simplified check)
            if np.dot(test_vector, A @ test_vector) <= 0:
                raise ValueError("diffusion_tensor might not be coercive")
                
        except Exception as e:
            raise ValueError(f"Invalid diffusion_tensor: {str(e)}")

@dataclass(frozen=True)
class PoissonDirichletConfig(BasePoissonConfig):
    """
    Configuration for homogeneous Dirichlet problem:
    - u - ∇·(A∇u) = f in Ω
    - u = 0 on ∂Ω
    """
    pass

@dataclass(frozen=True)
class PoissonNeumannConfig(BasePoissonConfig):
    """
    Configuration for Neumann problem:
    - u - ∇·(A∇u) = f in Ω
    - A∇u·n = 0 on ∂Ω
    """
    pass

@dataclass(frozen=True)
class PoissonPeriodicConfig(BasePoissonConfig):
    """
    Configuration for periodic problem on Ω=[0,L]²:
    - u - ∇·(A∇u) = f in Ω
    - u|x=0 = u|x=L and u|y=0 = u|y=L
    - A∇u·n periodic on boundaries
    """
    L: float = 1.0  # Domain size

@dataclass(frozen=True)
class HomogenizationConfig(BasePoissonConfig):
    """
    Configuration for homogenization problem:
    - Find uε ∈ H¹₀(Ω) such that
    - ∫Ω Aε∇uε·∇v = ∫Ω fv ∀v ∈ H¹₀(Ω)
    - Where Aε(x) = A(x/ε)
    
    Attributes
    ----------
    epsilon : float
        Period of the microstructure
    """
    epsilon: float = 0.1

def create_config(
    problem_type: str,
    diffusion_tensor: TensorField,
    right_hand_side: ScalarField,
    **kwargs
) -> BasePoissonConfig:
    """
    Factory function for configuration objects.
    
    Parameters
    ----------
    problem_type : str
        One of: 'dirichlet', 'neumann', 'periodic', 'homogenization'
    diffusion_tensor : TensorField
        Diffusion coefficient A(x,y)
    right_hand_side : ScalarField
        Source term f(x,y)
    **kwargs
        Additional parameters specific to each problem type
    """
    config_map = {
        'dirichlet': PoissonDirichletConfig,
        'neumann': PoissonNeumannConfig,
        'periodic': PoissonPeriodicConfig,
        'homogenization': HomogenizationConfig
    }
    
    if problem_type not in config_map:
        raise ValueError(f"Unknown problem type: {problem_type}")
        
    return config_map[problem_type](
        diffusion_tensor=diffusion_tensor,
        right_hand_side=right_hand_side,
        **kwargs
    )