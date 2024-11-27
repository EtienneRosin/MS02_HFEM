"""
Quadrature rules for numerical integration on triangular elements.

This module provides various quadrature rules for finite element integration,
including Gauss-Legendre and Gauss-Lobatto points and weights, with specific
focus on triangular elements.

The quadrature rules are designed to exactly integrate polynomials up to a 
certain order on the reference triangle T = {(x,y) | x ≥ 0, y ≥ 0, x + y ≤ 1}.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, ClassVar
import numpy as np
from numpy.typing import NDArray

@dataclass
class QuadratureRule:
    """
    Container for quadrature points and weights.
    
    Attributes
    ----------
    points : NDArray[np.float64]
        Array of quadrature points, shape (n_points, 2)
    weights : NDArray[np.float64]
        Array of weights, shape (n_points,)
    order : int
        Maximum order of polynomials exactly integrated
    """
    points: NDArray[np.float64]
    weights: NDArray[np.float64]
    order: int

class QuadratureStrategy(ABC):
    """
    Abstract base class for quadrature strategies.
    
    Each strategy provides points and weights for numerical integration
    on the reference triangle.
    """
    
    name: ClassVar[str]
    order: ClassVar[int]
    
    @classmethod
    @abstractmethod
    def get_rule(cls) -> QuadratureRule:
        """Return the quadrature rule with points and weights."""
        pass

class GaussLobatto4(QuadratureStrategy):
    """
    4-point Gauss-Lobatto quadrature rule for triangles.
    
    This rule exactly integrates polynomials up to order 3 and is defined
    on the reference triangle with points:
    - (1/3, 1/3)  weight: -9/32
    - (1/5, 1/5)  weight: 25/96
    - (1/5, 3/5)  weight: 25/96
    - (3/5, 1/5)  weight: 25/96
    """
    
    name = "gauss_lobatto_4"
    order = 3
    
    @classmethod
    def get_rule(cls) -> QuadratureRule:
        points = np.array([
            [1/3, 1/3],
            [1/5, 1/5],
            [1/5, 3/5],
            [3/5, 1/5]
        ])
        
        weights = np.array([-9/32, 25/96, 25/96, 25/96])
        
        return QuadratureRule(points, weights, cls.order)

class GaussLegendre1(QuadratureStrategy):
    """
    1-point Gauss-Legendre quadrature rule.
    Exactly integrates polynomials of order 1.
    """
    
    name = "gauss_legendre_1"
    order = 1
    
    @classmethod
    def get_rule(cls) -> QuadratureRule:
        points = np.array([[1/3, 1/3]])
        weights = np.array([1.0])
        return QuadratureRule(points, weights, cls.order)

class GaussLegendre3(QuadratureStrategy):
    """
    3-point Gauss-Legendre quadrature rule.
    Exactly integrates polynomials of order 2.
    """
    
    name = "gauss_legendre_3"
    order = 2
    
    @classmethod
    def get_rule(cls) -> QuadratureRule:
        points = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])
        weights = np.array([1/3, 1/3, 1/3])
        return QuadratureRule(points, weights, cls.order)

class GaussLegendre6(QuadratureStrategy):
    """
    6-point Gauss-Legendre quadrature rule.
    Exactly integrates polynomials of order 4.
    """
    
    name = "gauss_legendre_6"
    order = 4
    
    @classmethod
    def get_rule(cls) -> QuadratureRule:
        # a1 = 0.091576213509771
        # b1 = 0.445948490915965
        # w1 = 0.109951743655322
        
        # a2 = 0.816847572980459
        # b2 = 0.091576213509771
        # w2 = 0.109951743655322
        
        # points = np.array([
        #     [a1, b1], [b1, a1], [b1, b1],
        #     [a2, b2], [b2, a2], [b2, b2]
        # ])
        # weights = np.array([w1, w1, w1, w2, w2, w2])
        points = np.array([
            [0.0915762135098, 0.0915762135098],
            [0.8168475729805, 0.0915762135098],
            [0.0915762135098, 0.8168475729805],
            [0.1081030181681, 0.4459484909160],
            [0.4459484909160, 0.1081030181681],
            [0.4459484909160, 0.4459484909160]
        ])
        
        weights = np.array([
            0.05497587183, 0.05497587183, 0.05497587183,
            0.1116907948, 0.1116907948, 0.1116907948
        ])
        return QuadratureRule(points, weights, cls.order)

class QuadratureFactory:
    """Factory class to create quadrature rules by name."""
    
    _strategies = {
        "gauss_lobatto_4": GaussLobatto4,
        "gauss_legendre_1": GaussLegendre1,
        "gauss_legendre_3": GaussLegendre3,
        "gauss_legendre_6": GaussLegendre6
    }
    
    @classmethod
    def get_quadrature(cls, name: str) -> QuadratureStrategy:
        """
        Get a quadrature strategy by name.
        
        Parameters
        ----------
        name : str
            Name of the quadrature rule
            
        Returns
        -------
        QuadratureStrategy
            The requested quadrature strategy
            
        Raises
        ------
        ValueError
            If the requested strategy doesn't exist
        """
        if name not in cls._strategies:
            raise ValueError(
                f"Unknown quadrature rule: {name}. "
                f"Available rules: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name]

    @classmethod
    def register_strategy(cls, strategy: QuadratureStrategy) -> None:
        """
        Register a new quadrature strategy.
        
        Parameters
        ----------
        strategy : QuadratureStrategy
            The strategy to register
        """
        cls._strategies[strategy.name] = strategy

def integrate_on_reference_triangle(
    f: callable,
    rule: QuadratureRule = None,
    strategy_name: str = "gauss_lobatto_4"
) -> float:
    """
    Integrate a function on the reference triangle.
    
    Parameters
    ----------
    f : callable
        Function to integrate, takes (x,y) arguments
    rule : QuadratureRule, optional
        Specific quadrature rule to use
    strategy_name : str, optional
        Name of the quadrature strategy if no rule provided
        
    Returns
    -------
    float
        The approximated integral
    """
    if rule is None:
        rule = QuadratureFactory.get_quadrature(strategy_name).get_rule()
        
    result = 0.0
    for point, weight in zip(rule.points, rule.weights):
        result += weight * f(*point)
    return result


if __name__ == '__main__':
    rule = QuadratureFactory.get_quadrature("gauss_lobatto_4").get_rule()

    # Intégrer une fonction
    def f(x, y):
        return x*y

    result = integrate_on_reference_triangle(f)
    print(f"{result = }")