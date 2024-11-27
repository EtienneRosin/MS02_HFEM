"""
Module for defining and working with 2D functions in 2D problems.

Classes:
    TwoDimensionFunction

Functions:
    f_expr(x, y)
    u(x, y)
"""

import numpy as np

class TwoDimensionFunction:
    """
    Define a 2D function.

    Attributes
    ----------
    expr : callable
        Expression of the 2D function of the form :math:`f(x,y)`.

    Methods
    -------
    __init__(expr: callable)
        Construct the TwoDimensionFunction object.
    _validate_expr(expr: callable) -> callable
        Validate the 2D function expression.
    __call__(points: np.ndarray) -> np.ndarray
        Evaluate the expression at given points.
    """

    def __init__(self, expr: callable) -> None:
        """
        Construct the TwoDimensionFunction object.

        Parameters
        ----------
        expr : callable
            Expression of the 2D function.
        """
        self.expr = self._validate_expr(expr)

    def _validate_expr(self, expr: callable) -> callable:
        """
        Validate the 2D function expression.

        Parameters
        ----------
        expr : callable
            Expression of the 2D function.

        Returns
        -------
        expr : callable
            Validated expression of the 2D function.

        Raises
        ------
        ValueError
            If the expression is not correct for a 2D function. It should return :
            - A scalar if a point is given
            - An array of scalars if a list/array of N points is given
        """
        O = np.zeros(2)
        test_points = np.arange(start=0, stop=10, step=1).reshape((5, 2))
        result_one_point = expr(*O)
        result_multiple_points = np.array([expr(x, y) for x, y in test_points])

        if not isinstance(result_one_point, (int, float, np.number)):
            raise ValueError("Expression must return a scalar for a point.")
        if not (isinstance(result_multiple_points, np.ndarray) and result_multiple_points.shape == (5,)):
            raise ValueError("Expression must return an array of scalars for N points.")
        return expr

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the 2D function expression at given points.

        Parameters
        ----------
        points : np.ndarray
            Array of points where the expression should be evaluated.

        Returns
        -------
        np.ndarray
            Array of evaluated expressions at the given points.
        """
        points = np.atleast_2d(points)
        values = np.array([self.expr(x, y) for x, y in points])
        if points.shape[0] == 1:  # If a single point is provided, squeeze the extra dimension
            return values[0]
        return values

def f_expr(x, y):
    """
    2D function expression.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.

    Returns
    -------
    float
        The value of the expression at the given point.
    """
    return (1 + 5 * np.pi ** 2) * u(x, y)

def u(x, y):
    """
    Example function u(x, y).

    Parameters
    ----------
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.

    Returns
    -------
    float
        The value of the function at the given point.
    """
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

if __name__ == "__main__":
    # Test points
    lst_points = np.array([[1, 2], [3, 4], [5, 6]])
    O = np.zeros(2)

    # Create instance with expression
    rhs = TwoDimensionFunction(f_expr)

    # Apply the expression
    print("2D function:")
    print(rhs(O))          # Should print scalar value
    print(rhs(lst_points)) # Should print array of values
