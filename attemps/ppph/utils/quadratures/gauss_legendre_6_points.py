"""Module providing triangle 6-points Gauss-Legendre quadrature info. 

Notes
-----
The 6-points Gauss-Legendre quadrature is of order 4 and defined on the
reference triangle (0,0)(1,0)(0,1).

References
----------
The quadrature was given by:

.. [1] https://perso.ensta-paris.fr/~fliss/teaching-an201.html, TP Stokes
"""

import numpy as np

quadrature_points = np.array([
    [0.0915762135098,0.0915762135098],
    [0.8168475729805,0.0915762135098],
    [0.0915762135098,0.8168475729805],
    [0.1081030181681,0.4459484909160],
    [0.4459484909160,0.1081030181681],
    [0.4459484909160,0.4459484909160]]).astype(float)
quadrature_weights = np.array([0.05497587183,0.05497587183,0.05497587183,0.1116907948,0.1116907948,0.1116907948]).astype(float)