import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class QuadratureRule:
    """
    Represents a quadrature rule for numerical integration.
    
    Attributes:
        points (np.ndarray): Quadrature points in reference triangle
        weights (np.ndarray): Corresponding quadrature weights
        order (int): Approximation order of the quadrature rule
        name (str): Name of the quadrature method
    """
    points: np.ndarray
    weights: np.ndarray
    order: int
    name: str


class QuadratureStrategy(ABC):
    """Abstract base class for quadrature strategies."""
    
    @abstractmethod
    def get_quadrature_rule(self) -> QuadratureRule:
        """Generate and return the specific quadrature rule."""
        pass


class GaussLegendre6PointsQuadrature(QuadratureStrategy):
    """6-points Gauss-Legendre quadrature for triangular elements."""
    
    def get_quadrature_rule(self) -> QuadratureRule:
        """
        Return 6-points Gauss-Legendre quadrature rule.
        
        Returns:
            QuadratureRule: Configured quadrature rule
        """
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
        
        return QuadratureRule(
            points=points, 
            weights=weights, 
            order=4, 
            name="Gauss-Legendre 6-points"
        )


class GaussLobatto4PointsQuadrature(QuadratureStrategy):
    """4-points Gauss-Lobatto quadrature for triangular elements."""
    
    def get_quadrature_rule(self) -> QuadratureRule:
        """
        Return 4-points Gauss-Lobatto quadrature rule.
        
        Returns:
            QuadratureRule: Configured quadrature rule
        """
        # Example implementation - replace with actual points and weights
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1/3, 1/3]
        ])
        
        weights = np.array([
            1/12, 1/12, 1/12, 2/3
        ])
        
        return QuadratureRule(
            points=points, 
            weights=weights, 
            order=3, 
            name="Gauss-Lobatto 4-points"
        )


class QuadratureFactory:
    """
    Factory for creating and managing quadrature rules.
    Supports easy extension with new quadrature strategies.
    """
    
    _strategies = {
        "gauss_legendre_6": GaussLegendre6PointsQuadrature,
        "gauss_lobatto_4": GaussLobatto4PointsQuadrature
    }
    
    @classmethod
    def get_quadrature(cls, name: str) -> QuadratureRule:
        """
        Retrieve a specific quadrature rule.
        
        Args:
            name (str): Name of the quadrature rule
        
        Returns:
            QuadratureRule: The requested quadrature rule
        
        Raises:
            ValueError: If quadrature rule is not found
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown quadrature: {name}")
        
        strategy = cls._strategies[name]()
        return strategy.get_quadrature_rule()
    
    @classmethod
    def register_quadrature(cls, name: str, strategy: QuadratureStrategy):
        """
        Register a new quadrature strategy.
        
        Args:
            name (str): Name to register the strategy under
            strategy (QuadratureStrategy): Quadrature strategy to register
        """
        cls._strategies[name] = strategy


def example_usage():
    """Demonstrate quadrature framework usage."""
    
    # Get pre-defined quadrature rules
    legendre_rule = QuadratureFactory.get_quadrature("gauss_legendre_6")
    print(f"Quadrature: {legendre_rule.name}")
    print(f"Order: {legendre_rule.order}")
    print(f"Points:\n{legendre_rule.points}")
    print(f"Weights:\n{legendre_rule.weights}")
    
    # Custom quadrature example
    class CustomQuadrature(QuadratureStrategy):
        def get_quadrature_rule(self) -> QuadratureRule:
            points = np.array([[0.25, 0.25]])
            weights = np.array([1.0])
            return QuadratureRule(points, weights, order=1, name="Custom Single Point")
    
    # Register custom quadrature
    QuadratureFactory.register_quadrature("custom", CustomQuadrature)
    custom_rule = QuadratureFactory.get_quadrature("custom")
    print(f"\nCustom Quadrature: {custom_rule.name}")


if __name__ == "__main__":
    example_usage()