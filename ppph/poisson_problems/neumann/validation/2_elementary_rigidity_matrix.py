from validation_parameters import *
from ppph.utils import ReferenceElementBarycentricCoordinates, DiffusionTensor, TwoDimensionFunction
from ppph.utils.quadratures.gauss_lobatto_4_points import quadrature_weights, quadrature_points
bc = ReferenceElementBarycentricCoordinates()

diffusion_tensor = DiffusionTensor(diffusion_tensor_expr)

def _construct_elementary_rigidity_matrix(triangle_nodes: np.ndarray) -> np.ndarray:
    r"""Construct the P1 lagrange elementary rigidity matrix K^l on the considered triangle by a quadrature.
    
    Parameters
    ----------
    triangle: list
        considered triangle (list of 3 indices).

    Returns
    -------
    K_l: np.ndarray
        (3x3) elementary rigidity matrix K^l on the considered triangle.
    """
    D_l = np.abs(bc.D_l(*triangle_nodes))
    print(f"{D_l = }")
    K_l = np.zeros((3,3), dtype = float)
    
    # Inverse of the jacobian of the F_l transform
    A_l = bc.A_l(*triangle_nodes)
    
    # Compute the gradient of each barycentric function on the quadrature points
    lst_grad = np.array([[bc.grad_w_tilde(i, M_q) for M_q in quadrature_points] for i in range(1, 3 + 1)])
    # print(lst_grad.shape) # (3, 4, 2)
    
    # Apply A_l to each gradient of barycentric function
    
    lst_A_l_grad = np.array([[A_l @ grad[q] for q in range(len(quadrature_points))] for grad in lst_grad])
    # lst_A_l_grad = np.einsum('ij,klm->klij', A_l, grad)
    # print(lst_A_l_grad[0].shape) # (3, 4, 2, 2)
    
    # Compute the weighted diffusion tensor on each quadrature points
    mat_a = np.array([omega_q*diffusion_tensor(bc.F_l(M_q, *triangle_nodes)) 
                    for omega_q, M_q in zip(quadrature_weights, quadrature_points)])
    # print(mat_a.shape) # (4, 2, 2)
    
    # Apply the weighted diffusion tensor on each "gradient"
    # a_applied = np.einsum('qij,qkl->qikl', mat_a, lst_A_l_grad)  # Shape (4, 3, 4, 2)
    
    a_applied = np.array([[mat_a[q] @ A_l_grad[q] for q in range(len(quadrature_points))] for A_l_grad in lst_A_l_grad])
    # print(a_applied.shape) # (3, 4, 2, 2)

    # Compute the elementary rigidity matrix elements: A(M^q) A_l grad w_i \cdot A_l grad w_j
    for i in range(3):
        for j in range(3):
            for q in range(len(quadrature_points)):
                K_l[i,j] += np.dot(a_applied[i, q], lst_grad[j, q])
    
    # # Normalize the rigidity matrix
    K_l /= D_l
    # print(K_l)
    return K_l
if __name__ == '__main__':
    triangle_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    print(_construct_elementary_rigidity_matrix(triangle_nodes))