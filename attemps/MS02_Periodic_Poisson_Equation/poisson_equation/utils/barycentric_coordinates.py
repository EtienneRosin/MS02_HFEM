"""
@file barycentric_coordinates.py
@brief Module providing barycentric coordinates functions.
@authors Etienne Rosin
@version 0.1
@date 2024
"""
import numpy as np

# gradient of the barycentric coordinates on the triangle {(0, 0), (1, 0), (0, 1)}
Δw_1 = np.array([-1, -1])
Δw_2 = np.array([1, 0])
Δw_3 = np.array([0, 1])
gradients = np.array([Δw_1, Δw_2, Δw_3])

def w_tilde_1(M: np.ndarray) -> float:
    """
    @brief Calculate the first barycentric coordinate on the triangle of reference
    @param M: Considered point.
    @return: w_tilde_1(M)
    """
    M = np.atleast_2d(M)
    return - M[:, 0] - M[:, 1] + 1
    # return - M[0] - M[1] + 1

def w_tilde_2(M: np.ndarray) -> float:
    """
    @brief Calculate the second barycentric coordinate on the triangle of reference
    @param M: Considered point.
    @return: w_tilde_2(M)
    """
    M = np.atleast_2d(M)
    return M[:, 0]
    # return M[0]

def w_tilde_3(M: np.ndarray) -> float:
    """
    @brief Calculate the third barycentric coordinate on the triangle of reference
    @param M: Considered point.
    @return: w_tilde_3(M)
    """
    # return M[1]
    M = np.atleast_2d(M)
    return M[:, 1]
    

def D_l(S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> float:
    """
    @brief Calculate (to a factor 2 near) the area of the triangle T_l defined by S_1, S_2 and S_3, i.e. area(T_l) = |D_l|/2.
    @param S_1: vertex of the triangle.
    @param S_2: vertex of the triangle.
    @param S_3: vertex of the triangle.
    @return: surface of the triangle.
    """
    return((S_2[0] - S_1[0])*(S_3[1] - S_1[1]) - (S_2[1] - S_1[1])*(S_3[0] - S_1[0]))


def A_l(S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray:
    """
    @brief Calculate the matrix det(B)*(B.T)^-1, where B is the matrix formed by triangle vertices.
    @param S_1: vertex of the triangle.
    @param S_2: vertex of the triangle.
    @param S_3: vertex of the triangle.
    @return: the considered matrix det(B)*(B.T)^-1.
    """
    A = np.stack([S_3 - S_1, S_2 - S_1], axis=-1)
    A[0, 1] *= -1 
    A[1, 0] *= -1
    return(A)


def F_l(M: np.ndarray, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray:
    """
    @brief Map a point M in the reference element to the corresponding point in the real triangle T_l defined by S_1, S_2 and S_3.
    @param M: Point in the reference element (e.g., S_1, S_2, S_3).
    @param S_1: vertex of the triangle.
    @param S_2: vertex of the triangle.
    @param S_3: vertex of the triangle.
    @return: F_l(M): The corresponding point in the real triangle.
    """
    # Create the transformation matrix B_l
    B_l = np.stack([S_2 - S_1, S_3 - S_1], axis=-1)
    
    # print(f"{M = }")
    # Applicate the affine transformation
    return M @ B_l.T + S_1
    # return B_l @ M + S_1
    
if __name__ == '__main__':
    S_1 = np.array([0, 0])
    S_2 = np.array([1, 0])
    S_3 = np.array([0, 1])

    # Calculate area (should be 0.5 for a right triangle)
    area = D_l(S_1, S_2, S_3) / 2
    print(f"Area of the triangle: {area}")

    # Calculate the matrix A_l
    matrix_A = A_l(S_1=S_1, S_2=S_2, S_3=S_3)
    print(f"Matrix A_l: \n{matrix_A}")

    # Test transformation F_l
    M_ref = np.array([0.5, 0.5])  # Point in reference element (inside the unit triangle)
    M_real = F_l(M_ref, S_1, S_2, S_3)
    print(f"Point in the real triangle: {M_real}")
    
    M_ref_batch = np.array([
        [0.5, 0.5],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.1, 0.8]
    ])

    # Apply the batch transformation
    M_real_batch = F_l(M_ref_batch, S_1, S_2, S_3)
    print(f"Points in the real triangle:\n{M_real_batch}")
    
    M_ref_single = np.array([0.5, 0.5])
    # Multiple points in the reference element
    M_ref_multiple = np.array([[0.5, 0.5], [0.2, 0.3], [0.7, 0.1]])

    # For a single point
    w1_single = w_tilde_1(M_ref_single)
    print(f"w_tilde_1 for a single point: {w1_single}")

    # For multiple points
    w1_multiple = w_tilde_1(M_ref_multiple)
    print(f"w_tilde_1 for multiple points: {w1_multiple}")
    
    # lst_A_grad = np.array([matrix_A @ grad for grad in gradients])
    
    # print(f"{lst_A_grad = }")