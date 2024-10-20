from validation_parameters import *
from ppph.utils import ReferenceElementBarycentricCoordinates, DiffusionTensor, TwoDimensionFunction
bc = ReferenceElementBarycentricCoordinates()

def _construct_elementary_mass_matrix(triangle_nodes: np.ndarray) -> np.ndarray:
    r"""Construct the P1 lagrange elementary mass matrix M^l on the considered triangle.
    
    Parameters
    ----------
    triangle: list
        considered triangle (list of 3 indices).

    Returns
    -------
    M_l: np.ndarray
        (3x3) elementary mass matrix M^l on the considered triangle.
    """
    D_l = np.abs(bc.D_l(*triangle_nodes))
    M_l = np.ones((3,3), dtype = float)
    for i in range(3):
        M_l[i,i] += 1
    M_l *= D_l / 24
    return M_l

if __name__ == '__main__':
    triangle_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    print(_construct_elementary_mass_matrix(triangle_nodes))