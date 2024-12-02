import numpy as np
from hfem.core import BarycentricTransformation
from hfem.core.quadratures import QuadratureStrategy, GaussLegendre6, QuadratureFactory, QuadratureRule
from hfem.core.aliases import TensorField
from typing import Optional, Callable
from numpy.typing import NDArray

def assemble_elementary_stiffness_matrix(
    triangle_nodes: np.ndarray,
    diffusion_tensor: TensorField | NDArray[np.float64],
    quadrature: Optional[QuadratureRule] = QuadratureFactory.get_quadrature("gauss_legendre_6").get_rule()
    ) -> np.ndarray:
    """
    Assemble elementary stiffness matrix using quadrature.
    
    Parameters
    ----------
    triangle : np.ndarray
        Triangle vertex indices
        
    Returns
    -------
    np.ndarray
        3x3 elementary stiffness matrix
    """
    
    jacobian = BarycentricTransformation.compute_jacobian(triangle_nodes)
    inv_jac = np.linalg.inv(jacobian).T
    det_j = np.abs(np.linalg.det(jacobian))

    stiffness_matrix = np.zeros((3, 3))
    
    for w_q, x_q in zip(quadrature.weights, quadrature.points):
        point = np.dot(jacobian, x_q) + triangle_nodes[0]
        if isinstance(diffusion_tensor, Callable):
            A_local = diffusion_tensor(*point)
        else:
            A_local = diffusion_tensor
        
        for i in range(3):
            for j in range(3):
                grad_ref_i = BarycentricTransformation.compute_reference_gradient(i)
                grad_ref_j = BarycentricTransformation.compute_reference_gradient(j)
                
                grad_i = inv_jac @ grad_ref_i
                grad_j = inv_jac @ grad_ref_j
                
                stiffness_matrix[i, j] += w_q * np.dot(A_local @ grad_i, grad_j)
    
    return stiffness_matrix * det_j


if __name__ == '__main__':
    pass
    # def f(x):
    #     return 1


    # # print(type(f) == Callable[[float, float], NDArray[np.float64]])
    # print(type(f) == Callable[[float], float])
    # # print(type(f))
    # print(isinstance(, Callable))
    # print(isinstance(f, ))