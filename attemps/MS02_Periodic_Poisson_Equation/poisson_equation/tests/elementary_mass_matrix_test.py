import numpy as np

from poisson_equation.utils.reference_element_barycentric_coordinates import ReferenceElementBarycentricCoordinates
from poisson_equation.utils.quadratures.gauss_legendre_6_points import *

bc = ReferenceElementBarycentricCoordinates()



def rho(x,y):
    return 1


def expected_mass_matrix(vertices):
    M = np.zeros((3,3))
    # A_l = bc.A_l(*vertices)
    D_l = np.abs(bc.D_l(*vertices))
    for i in range(3):
        for j in range(3):
            for omega_q, S_q in zip(quadrature_weights, quadrature_points):
                M[i,j] += omega_q * bc.w_tilde(i+1, S_q) * bc.w_tilde(j+1, S_q)
    print(f"{D_l = }")
    return(D_l * M)
    
    
def constructed_mass_matrix(vertices):
    RHO = rho(*bc.F_l(quadrature_points, *vertices).T)
    W_i = np.array([bc.w_tilde(i + 1, quadrature_points) for i in range(3)])
    D_l = np.abs(bc.D_l(*vertices))
    
    M = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            M[i,j] = np.sum(quadrature_weights * RHO * W_i[i] * W_i[j])
            
    return D_l * M

if __name__ == '__main__':
    S_1 = np.array([0, 0])
    S_2 = np.array([1, 0])
    S_3 = np.array([0, 1])
    vertices = np.array([S_1, S_2, S_3])
    vertices_prime = np.array([[1, 0], [1, 1], [0, 1]])
    
    M_expected = expected_mass_matrix(vertices_prime)
    print(M_expected)
    
    M = constructed_mass_matrix(vertices_prime)
    print(M)
    
    print(f"{np.linalg.norm(M_expected - M) = }")