import numpy as np
from poisson_equation.utils.reference_element_barycentric_coordinates import ReferenceElementBarycentricCoordinates
from poisson_equation.utils.quadratures.gauss_legendre_6_points import *

bc = ReferenceElementBarycentricCoordinates()

if __name__ == '__main__':
    S_1 = np.array([0, 0])
    S_2 = np.array([1, 0])
    S_3 = np.array([0, 1])
    vertices = np.array([S_1, S_2, S_3])
    D_l = np.abs(bc.D_l(*vertices))
    print(f"Surface of the triangle : {D_l/2}")
    
    # try to see if our quadrature is good enough
    value = np.sum(quadrature_weights)
    print(f"Approximate surface of the triangle : {D_l*value}")
    pass