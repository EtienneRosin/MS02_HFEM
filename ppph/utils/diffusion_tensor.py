"""
Module for defining and working with 2D diffusion tensors.

Classes:
    DiffusionTensor

Functions:
    identity_tensor_expr(x, y)
    diffusion_tensor_2_expr(x, y)
"""

import numpy as np

class DiffusionTensor:
    r"""
    Define a 2D diffusion tensor of the form :math:`\boldsymbol{A}(x,y)`.

    Attributes
    ----------
    expr : callable
        Expression of the tensor.
    """

    def __init__(self, expr: callable) -> None:
        r"""
        Construct the DiffusionTensor object.

        Parameters
        ----------
        expr : callable
            Expression of the tensor.
        """
        self.expr = self._validate_expr(expr)

    def _validate_expr(self, expr: callable) -> callable:
        """
        Validate the diffusion tensor expression.

        Parameters
        ----------
        expr : callable
            Expression of the tensor.

        Returns
        -------
        expr : callable
            Validated expression of the tensor.

        Raises
        ------
        ValueError
            If the expression is not correct for a 2D diffusion tensor. It should return :
            - a (2x2) matrix if a point is given
            - Nx(2x2) matrices if a list/array of N points is given
        """
        O = np.zeros(2)
        test_points = np.arange(start=0, stop=10, step=1).reshape((5, 2))
        result_one_point = expr(*O)
        result_multiple_points = np.array([expr(x, y) for x, y in test_points])

        if not (isinstance(result_one_point, np.ndarray) and result_one_point.shape == (2, 2)):
            raise ValueError("Expression must return a 2x2 matrix for a point.")
        if not (isinstance(result_multiple_points, np.ndarray) and result_multiple_points.shape == (5, 2, 2)):
            raise ValueError("Expression must return a Nx2x2 matrix for N points.")
        return expr

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the tensor expression at given points.

        Parameters
        ----------
        points : np.ndarray
            Array of points where the tensor should be evaluated.

        Returns
        -------
        np.ndarray
            Array of evaluated tensors at the given points.
        """
        points = np.atleast_2d(points)
        tensors = np.array([self.expr(x, y) for x, y in points])
        if points.shape[0] == 1:  # If a single point is provided, squeeze the extra dimension
            return tensors[0]
        return tensors

def identity_tensor_expr(x, y):
    """
    Diffusion tensor that returns the identity matrix for all points.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.

    Returns
    -------
    np.ndarray
        The identity matrix.
    """
    return np.eye(2)

def diffusion_tensor_2_expr(x, y):
    """
    Diffusion tensor that scales components based on the point coordinates.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.

    Returns
    -------
    np.ndarray
        A tensor of the form [[x, 2*x], [y, 2*y]].
    """
    return np.array([[x, 2 * x], [y, 2 * y]])

if __name__ == "__main__":
    # Test points
    lst_points = np.array([[1, 2], [3, 4], [5, 6]])
    O = np.zeros(2)

    # Create instances with expressions
    dt1 = DiffusionTensor(identity_tensor_expr)
    dt2 = DiffusionTensor(diffusion_tensor_2_expr)

    # Apply the tensors
    print("Optimized Diffusion Tensor 1:")
    print(dt1(O).shape)          # Should print (2, 2)
    print(dt1(lst_points).shape) # Should print (3, 2, 2)

    print("\nOptimized Diffusion Tensor 2:")
    print(dt2(O).shape)          # Should print (2, 2)
    print(dt2(lst_points).shape) # Should print (3, 2, 2)
