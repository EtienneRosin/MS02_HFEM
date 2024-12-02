import numpy as np
from hfem.core import BarycentricTransformation

def assemble_elementary_mass_matrix(triangle_nodes: np.ndarray) -> np.ndarray:
    """
    Assemble elementary mass matrix for a triangle.
    
    Parameters
    ----------
    triangle : np.ndarray
        Triangle vertex indices
        
    Returns
    -------
    np.ndarray
        3x3 elementary mass matrix
    """
    det_j = np.abs(np.linalg.det(BarycentricTransformation.compute_jacobian(triangle_nodes)))
    mass_matrix = np.ones((3, 3)) * det_j / 24
    np.fill_diagonal(mass_matrix, mass_matrix[0,0] * 2)
    return mass_matrix