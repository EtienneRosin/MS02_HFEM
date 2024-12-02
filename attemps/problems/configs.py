"""
Configuration module for Homogenization and Poisson Problems.

This module provides configuration classes for different types of Poisson problems:
- Homogeneous Neumann boundary conditions
- Homogeneous Dirichlet boundary conditions 
- Periodic boundary conditions
- Homogenization with microstructure
"""

from dataclasses import dataclass, field
from typing import Optional, Type, List
import numpy as np
from numpy.typing import NDArray

from hfem.core.quadratures import (
    QuadratureStrategy,
    QuadratureFactory
)
from hfem.core.aliases import TensorField, ScalarField
# Type aliases


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

# @dataclass(frozen=True)
# class PoissonDirichletConfig(BasePoissonConfig):
#     """
#     Configuration for homogeneous Dirichlet problem:
#     - u - ∇·(A∇u) = f in Ω
#     - u = 0 on ∂Ω
#     """
#     pass

# @dataclass(frozen=True)
# class PoissonNeumannConfig(BasePoissonConfig):
#     """
#     Configuration for Neumann problem:
#     - u - ∇·(A∇u) = f in Ω
#     - A∇u·n = 0 on ∂Ω
#     """
#     pass

# @dataclass(frozen=True)
# class PoissonPeriodicConfig(BasePoissonConfig):
#     """
#     Configuration for periodic problem on Ω=[0,L]²:
#     - u - ∇·(A∇u) = f in Ω
#     - u|x=0 = u|x=L and u|y=0 = u|y=L
#     - A∇u·n periodic on boundaries
#     """
#     L: float = 1.0  # Domain size

# @dataclass(frozen=True)
# class HomogenizationConfig(BasePoissonConfig):
#     """
#     Configuration for homogenization problem:
#     - Find uε ∈ H¹₀(Ω) such that
#     - ∫Ω Aε∇uε·∇v = ∫Ω fv ∀v ∈ H¹₀(Ω)
#     - Where Aε(x) = A(x/ε)
    
#     Attributes
#     ----------
#     epsilon : float
#         Period of the microstructure
#     """
#     epsilon: float = 0.1
@dataclass(frozen=True)
class MicrostructuredPoissonConfig:
    """
    Configuration for a Poisson problem with periodic microstructure.
    
    Attributes
    ----------
    epsilon : float
        Period of the microstructure
    base_diffusion_tensor : TensorField
        The 1-periodic tensor A(y) used to define A_eps(x) = A(x/epsilon)
    right_hand_side : ScalarField
        Source term f ∈ L²(Ω)
    exact_solution : Optional[ScalarField]
        Exact solution if known, for validation
    quadrature_strategy : Type[QuadratureStrategy]
        Numerical integration strategy
    """
    # Required parameters
    epsilon: float
    base_diffusion_tensor: TensorField
    right_hand_side: ScalarField
    
    # Optional parameters
    exact_solution: Optional[ScalarField] = None
    quadrature_strategy: Type[QuadratureStrategy] = field(
        default_factory=lambda: QuadratureFactory.get_quadrature("gauss_legendre_6")
    )
    
    @property
    def diffusion_tensor(self) -> TensorField:
        """Get the scaled diffusion tensor A_eps(x) = A(x/epsilon)."""
        def scaled_tensor(x: float, y: float) -> NDArray[np.float64]:
            return self.base_diffusion_tensor(x/self.epsilon, y/self.epsilon)
        return scaled_tensor
    
    def validate(self) -> None:
        """Validate tensor properties."""
        test_point = (0.0, 0.0)
        test_vector = np.array([1.0, 1.0])
        
        try:
            A = self.diffusion_tensor(*test_point)
            if not isinstance(A, np.ndarray) or A.shape != (2, 2):
                raise ValueError("diffusion_tensor must return a 2x2 matrix")
            
            if not np.allclose(A, A.T):
                raise ValueError("diffusion_tensor must be symmetric")
            
            if np.dot(test_vector, A @ test_vector) <= 0:
                raise ValueError("diffusion_tensor might not be coercive")
                
        except Exception as e:
            raise ValueError(f"Invalid diffusion_tensor: {str(e)}")
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        self.validate()
        

@dataclass(frozen=True)
class PenalizedCellProblemConfig:
    """
    Configuration for the penalized cell problems.
    
    Attributes
    ----------
    eta : float
        Penalization factor
    diffusion_tensor : TensorField
        Function A(x,y) that must satisfy:
        1. Uniform boundedness: ∃C>0, ∀(x,y)∈Ω, ∀i,j, |Aij(x,y)| ≤ C
        2. Uniform coercivity: ∃c>0, ∀(x,y)∈Ω, ∀ξ∈R², A(x,y)ξ·ξ ≥ c|ξ|²
    exact_correctors : Optional[List[ScalarField]]
        List of exact correctors if known, for validation
    exact_homogenized_tensor : Optional[NDArray[np.float64]]
        Exact homogenized tensor if known, for validation
    quadrature_strategy : Type[QuadratureStrategy]
        Numerical integration strategy
    periodic_tolerance : float
        Tolerance for identifying periodic nodes (default: 1e-10)
    """
    # Required parameters
    eta: float
    diffusion_tensor: TensorField

    # Optional parameters
    exact_correctors: Optional[List[ScalarField]] = None
    exact_homogenized_tensor: Optional[NDArray[np.float64]] = None
    quadrature_strategy: Type[QuadratureStrategy] = field(
        default_factory=lambda: QuadratureFactory.get_quadrature("gauss_legendre_6")
    )
    periodic_tolerance: float = field(default=1e-10)

    def validate(self) -> None:
        """Validate configuration properties."""
        self._validate_eta()
        self._validate_diffusion_tensor()
        self._validate_exact_solutions()
        
    def _validate_eta(self) -> None:
        """Validate penalization parameter."""
        if self.eta <= 0:
            raise ValueError("eta should be greater than 0")
        if self.eta > 1e6:
            raise Warning("Large eta values may lead to ill-conditioning")
            
    def _validate_diffusion_tensor(self) -> None:
        """Validate properties of the diffusion tensor."""
        test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        test_vectors = [np.array([1., 0.]), np.array([0., 1.]), np.array([1., 1.])]
        
        for point in test_points:
            try:
                A = self.diffusion_tensor(*point)
                if not isinstance(A, np.ndarray) or A.shape != (2, 2):
                    raise ValueError("diffusion_tensor must return a 2x2 matrix")
                
                if not np.allclose(A, A.T):
                    raise ValueError("diffusion_tensor must be symmetric")
                
                # Test coercivity on multiple vectors
                for v in test_vectors:
                    if np.dot(v, A @ v) <= 0:
                        raise ValueError("diffusion_tensor might not be coercive")
                        
            except Exception as e:
                raise ValueError(f"Invalid diffusion_tensor at point {point}: {str(e)}")
    
    def _validate_exact_solutions(self) -> None:
        """Validate exact solutions if provided."""
        if self.exact_correctors is not None:
            if len(self.exact_correctors) != 2:
                raise ValueError("Should provide exactly 2 correctors")
            # Test correctors at some points
            test_point = (0.5, 0.5)
            for i, corr in enumerate(self.exact_correctors):
                try:
                    _ = corr(*test_point)
                except Exception as e:
                    raise ValueError(f"Invalid corrector {i} at {test_point}: {str(e)}")
                    
        if self.exact_homogenized_tensor is not None:
            if not isinstance(self.exact_homogenized_tensor, np.ndarray):
                raise ValueError("exact_homogenized_tensor must be a numpy array")
            if self.exact_homogenized_tensor.shape != (2, 2):
                raise ValueError("exact_homogenized_tensor must be 2x2")
            if not np.allclose(self.exact_homogenized_tensor, 
                             self.exact_homogenized_tensor.T):
                raise ValueError("exact_homogenized_tensor must be symmetric")
            
    def __post_init__(self):
        """Validate configuration on initialization."""
        self.validate()

# def create_config(
#     problem_type: str,
#     diffusion_tensor: TensorField,
#     right_hand_side: ScalarField,
#     **kwargs
# ) -> BasePoissonConfig:
#     """
#     Factory function for configuration objects.
    
#     Parameters
#     ----------
#     problem_type : str
#         One of: 'dirichlet', 'neumann', 'periodic', 'homogenization'
#     diffusion_tensor : TensorField
#         Diffusion coefficient A(x,y)
#     right_hand_side : ScalarField
#         Source term f(x,y)
#     **kwargs
#         Additional parameters specific to each problem type
#     """
#     config_map = {
#         'dirichlet': PoissonDirichletConfig,
#         'neumann': PoissonNeumannConfig,
#         'periodic': PoissonPeriodicConfig,
#         'homogenization': HomogenizationConfig
#     }
    
#     if problem_type not in config_map:
#         raise ValueError(f"Unknown problem type: {problem_type}")
        
#     return config_map[problem_type](
#         diffusion_tensor=diffusion_tensor,
#         right_hand_side=right_hand_side,
#         **kwargs
#     )