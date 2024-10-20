import numpy as np

from poisson_equation.utils.reference_element_barycentric_coordinates import ReferenceElementBarycentricCoordinates
from poisson_equation.utils.quadratures.gauss_legendre_6_points import *

bc = ReferenceElementBarycentricCoordinates()

def diffusion_tensor(x, y):
        return np.eye(2)

def rho(x,y):
    return 1


def expected_rigidity_matrix(vertices):
    K = np.zeros((3,3))
    D_l = np.abs(bc.D_l(*vertices))
    A_l = bc.A_l(*vertices)
    
    for i in range(3):
        for j in range(3):
            for omega_q, S_q in zip(quadrature_weights, quadrature_points):
                K[i,j] += omega_q * np.dot(np.eye(2) @ A_l @ bc.grad_w_tilde(i+1, S_q), A_l @ bc.grad_w_tilde(j+1, S_q)) 
                # K[i,j] += omega_q *  * bc.grad_w_tilde(j+1, S_q)
    # print(f"{D_l = }")
    return(K / D_l)
    
    
def constructed_rigidity_matrix(vertices):
    D_l = np.abs(bc.D_l(*vertices))
    A_l = bc.A_l(*vertices)
    
    grad_W_i = np.array([bc.grad_w_tilde(i + 1, quadrature_points) for i in range(3)])
    mat_a = np.array([diffusion_tensor(*bc.F_l(S_q, *vertices).T) for S_q in quadrature_points])
    K = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for q, omega_q in enumerate(quadrature_weights):
                K[i,j] += omega_q * np.dot(mat_a[q] @ (A_l @ grad_W_i[i, q]), A_l @ grad_W_i[j, q])
                
    K /= D_l
    return(K)



if __name__ == '__main__':
    S_1 = np.array([0, 0])
    S_2 = np.array([1, 0])
    S_3 = np.array([0, 1])
    vertices = np.array([S_1, S_2, S_3])
    
    vertices_prime = np.array([[1, 0], [1, 1], [0, 1]])
    
    K_expected = expected_rigidity_matrix(vertices_prime)
    print(K_expected)
    
    K = constructed_rigidity_matrix(vertices_prime)
    print(K)
    
    print(f"{np.linalg.norm(K_expected - K) = }")