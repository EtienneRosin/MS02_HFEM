"""
Utility module for barycentric coordinate transformations in finite element computations.

This module provides tools for handling coordinate transformations between the reference
triangle and physical triangles, including computation of gradients, Jacobian matrices,
and related geometric transformations.
"""

import numpy as np
from numpy.typing import NDArray

class BarycentricTransformation:
    """
    Utility class for barycentric coordinate transformations.
    
    This class provides static methods for:
    - Computing gradients of reference basis functions
    - Computing Jacobian matrices for triangular elements
    - Transforming coordinates between reference and physical triangles
    """
    
    @staticmethod
    def compute_reference_gradient(point_index: int) -> NDArray[np.float64]:
        """
        Compute gradient of reference basis function.
        
        Parameters
        ----------
        point_index : int
            Index of the vertex (0, 1, or 2)
            
        Returns
        -------
        NDArray[np.float64]
            Gradient vector [dx, dy] of the basis function
        
        Raises
        ------
        ValueError
            If point_index is not 0, 1, or 2
            
        Notes
        -----
        The reference triangle is defined by vertices:
        - (0,0) for point_index = 0
        - (1,0) for point_index = 1
        - (0,1) for point_index = 2
        """
        grad_ref = {
            0: np.array([-1.0, -1.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0])
        }
        
        if point_index not in grad_ref:
            raise ValueError(f"Invalid point_index: {point_index}. Must be 0, 1, or 2.")
            
        return grad_ref[point_index]

    @staticmethod
    def compute_jacobian(triangle_nodes: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the Jacobian matrix for transformation from reference to physical triangle.
        
        Parameters
        ----------
        triangle_nodes : NDArray[np.float64]
            Coordinates of triangle vertices, shape (3, 2)
            
        Returns
        -------
        NDArray[np.float64]
            Jacobian matrix [dx1/dxi dx2/dxi; dx1/deta dx2/deta]
            
        Raises
        ------
        ValueError
            If triangle_nodes has incorrect shape
            
        Notes
        -----
        The Jacobian matrix J is defined as:
        J = [ x₂-x₁  x₃-x₁ ]
            [ y₂-y₁  y₃-y₁ ]
        where (xᵢ,yᵢ) are the coordinates of vertex i.
        """
        if triangle_nodes.shape != (3, 2):
            raise ValueError(
                f"triangle_nodes must have shape (3, 2), got {triangle_nodes.shape}"
            )
            
        x1, x2, x3 = triangle_nodes[:, 0]
        y1, y2, y3 = triangle_nodes[:, 1]
        
        return np.array([
            [x2 - x1, x3 - x1],
            [y2 - y1, y3 - y1]
        ])
    
    @staticmethod
    def compute_barycentric_coordinates(
        point: NDArray[np.float64],
        triangle_nodes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute barycentric coordinates of a point in a triangle.
        
        Parameters
        ----------
        point : NDArray[np.float64]
            Point coordinates [x, y]
        triangle_nodes : NDArray[np.float64]
            Triangle vertices coordinates, shape (3, 2)
            
        Returns
        -------
        NDArray[np.float64]
            Barycentric coordinates [λ₁, λ₂, λ₃]
            
        Notes
        -----
        For a point (x,y), the barycentric coordinates are:
        λ₁(x,y) = ((y₂-y₃)(x-x₃) + (x₃-x₂)(y-y₃)) / D
        λ₂(x,y) = ((y₃-y₁)(x-x₃) + (x₁-x₃)(y-y₃)) / D
        λ₃(x,y) = 1 - λ₁ - λ₂
        where D = (y₂-y₃)(x₁-x₃) + (x₃-x₂)(y₁-y₃)
        """
        x, y = point
        x1, x2, x3 = triangle_nodes[:, 0]
        y1, y2, y3 = triangle_nodes[:, 1]
        
        # Compute differences
        x23, x31, x12 = x2 - x3, x3 - x1, x1 - x2
        y23, y31, y12 = y2 - y3, y3 - y1, y1 - y2
        
        # Compute denominator
        D = x23*y31 - x31*y23
        
        if abs(D) < 1e-10:
            raise ValueError("Triangle has zero area")
            
        # Compute barycentric coordinates
        lambda1 = ((y23*(x - x3) + x32*(y - y3)) / D)
        lambda2 = ((y31*(x - x3) + x13*(y - y3)) / D)
        lambda3 = 1.0 - lambda1 - lambda2
        
        return np.array([lambda1, lambda2, lambda3])

    @staticmethod
    def physical_to_reference(
        point: NDArray[np.float64],
        triangle_nodes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Transform point from physical to reference triangle.
        
        Parameters
        ----------
        point : NDArray[np.float64]
            Physical coordinates [x, y]
        triangle_nodes : NDArray[np.float64]
            Physical triangle vertices, shape (3, 2)
            
        Returns
        -------
        NDArray[np.float64]
            Reference coordinates [xi, eta]
        """
        jacobian = BarycentricTransformations.compute_jacobian(triangle_nodes)
        point_relative = point - triangle_nodes[0]
        return np.linalg.solve(jacobian, point_relative)