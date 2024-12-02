import numpy as np
from hfem.core import BarycentricTransformation

# def assemble_elementary_derivatives_matrices(
#     triangle_nodes: np.ndarray
#     ):
    
#     jacobian = BarycentricTransformation.compute_jacobian(triangle_nodes)
#     inv_jac = np.linalg.inv(jacobian).T
#     det_j = np.abs(np.linalg.det(jacobian))
    
    
#     # vec_1 = jacobian_inv[0, 0] * reference_grad_x + jacobian_inv[0, 1] * reference_grad_y
# # vec_2 = jacobian_inv[1, 0] * reference_grad_x + jacobian_inv[1, 1] * reference_grad_y
    
#     vec_1 = np.zeros(3)
#     vec_2 = np.zeros(3)
#     for j in range(3):
#         vec_1[j] = BarycentricTransformation.compute_reference_gradient(j)[0]
#         vec_2[j] = BarycentricTransformation.compute_reference_gradient(j)[1]
        
        
    
#     G_1_elem = (det_j/6)*np.repeat(vec_1[:, np.newaxis], 3, axis=1)
#     G_2_elem = (det_j/6)*np.repeat(vec_2[:, np.newaxis], 3, axis=1)
    
#     return G_1_elem, G_2_elem

# if __name__ == '__main__':
#     triangle_nodes = np.array([[0, 0], [0, 1], [1, 0]])
    
#     print(assemble_elementary_derivatives_matrices(triangle_nodes))
    
#     jacobian_inv = np.linalg.inv(jacobian)

# def assemble_elementary_derivatives_matrices(triangle_nodes: np.ndarray):
#     # on a G_i_IJ = \int_\Omega = \partial_{x_i} w_I w_J
    
#     # Calcul de la jacobienne et son déterminant
#     jacobian = BarycentricTransformation.compute_jacobian(triangle_nodes)
#     jacobian_inv = np.linalg.inv(jacobian).T
#     det_j = np.abs(np.linalg.det(jacobian))
    
#     # Initialisation des matrices pour stocker les gradients transformés
#     G_1_elem = np.zeros((3, 3))
#     G_2_elem = np.zeros((3, 3))
    
#     # Pour chaque fonction de base
#     for i in range(3):
#         # Récupérer le gradient de référence
#         ref_grad = BarycentricTransformation.compute_reference_gradient(i)
        
#         # Transformer le gradient via la jacobienne inverse
#         physical_grad = jacobian_inv @ ref_grad
#         # physical_grad = ref_grad
        
#         # Remplir les matrices avec les composantes transformées
#         for j in range(3):
#             G_1_elem[i, j] = physical_grad[0]  # composante x
#             G_2_elem[i, j] = physical_grad[1]  # composante y
    
#     # return G_1_elem, G_2_elem
#     return (det_j)*G_1_elem, (det_j)*G_2_elem
#     # return (det_j/6)*G_1_elem, (det_j/6)*G_2_elem

def assemble_elementary_derivatives_matrices(triangle_nodes: np.ndarray):
    """Calcule les matrices élémentaires G_i_IJ = ∂_{x_i} w_I w_J."""
    
    jacobian = BarycentricTransformation.compute_jacobian(triangle_nodes)
    # jacobian_inv = np.linalg.inv(jacobian).T
    jacobian_inv = np.linalg.inv(jacobian.T)
    det_j = np.abs(np.linalg.det(jacobian))
    
    G_1_elem = np.zeros((3, 3))
    G_2_elem = np.zeros((3, 3))
    
    for I in range(3):
        ref_grad_I = BarycentricTransformation.compute_reference_gradient(I)
        physical_grad_I = jacobian_inv @ ref_grad_I
        
        # La dérivée de la fonction de base est constante sur l'élément
        for J in range(3):
            # On intègre juste la fonction de base J sur l'élément
            G_1_elem[I, J] = physical_grad_I[0] * (det_j/6)  # 1/6 est l'intégrale de la fonction de base
            G_2_elem[I, J] = physical_grad_I[1] * (det_j/6)
    
    return G_1_elem, G_2_elem


if __name__ == '__main__':
    # Triangle test
    triangle_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    triangle_nodes = np.array([[1, 1], [2, 1], [1, 4]])
    triangle_nodes = np.array([[0, 0], [2, 0], [0, 2]])
    
    G1, G2 = assemble_elementary_derivatives_matrices(triangle_nodes)
    print("G1 (dérivées en x):")
    print(G1)
    print("\nG2 (dérivées en y):")
    print(G2)
    