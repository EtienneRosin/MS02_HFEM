import numpy as np


class ReferenceElementBarycentricCoordinates:
    r"""
        @class BarycentricCoordinates
        @brief Contains definition of functions related to the barycentric coordinates
    """
    def __init__(self):
        """
        Initialize the class with barycentric coordinate functions.
        """
        # Dictionary mapping integers i to the corresponding barycentric functions
        self.w_tilde_functions__ = {
            1: self.w_tilde_1__,
            2: self.w_tilde_2__,
            3: self.w_tilde_3__
        }
        
        self.w_functions__ = {
            1: self.w_1__,
            2: self.w_2__,
            3: self.w_3__
        }

        # Gradient functions for barycentric coordinates
        self.grad_w_tilde_functions__ = {
            1: self.grad_w_tilde_1__,
            2: self.grad_w_tilde_2__,
            3: self.grad_w_tilde_3__
        }

    # Existing functions for barycentric coordinates
    # def w_tilde_1__(self, M: np.ndarray) -> np.ndarray:
    #     M = np.atleast_2d(M)
    #     return - M[:, 0] - M[:, 1] + 1

    # def w_tilde_2__(self, M: np.ndarray) -> np.ndarray:
    #     M = np.atleast_2d(M)
    #     return M[:, 0]

    # def w_tilde_3__(self, M: np.ndarray) -> np.ndarray:
    #     M = np.atleast_2d(M)
    #     return M[:, 1]

    # def w_tilde(self, i: int, M: np.ndarray) -> np.ndarray:
    #     if i in self.w_tilde_functions__:
    #         return self.w_tilde_functions__[i](M)
    #     else:
    #         raise ValueError(f"Invalid index {i}. Valid indices are 1, 2, or 3.")
    def w_1__(self, M: np.ndarray, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray :
        D_l = self.D_l(S_1, S_2, S_3)
        y_23 = S_2[1] - S_3[1]
        x_23 = S_2[0] - S_3[0]
        return (y_23 * (M[0] - S_3[0]) - x_23 * (M[1] - S_3[1])) / D_l
    
    def w_2__(self, M: np.ndarray, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray :
        D_l = self.D_l(S_1, S_2, S_3)
        y_31 = S_3[1] - S_1[1]
        x_31 = S_3[0] - S_1[0]
        return (y_31 * (M[0] - S_1[0]) - x_31 * (M[1] - S_1[1])) / D_l
    
    def w_3__(self, M: np.ndarray, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray :
        D_l = self.D_l(S_1, S_2, S_3)
        y_12 = S_1[1] - S_2[1]
        x_12 = S_1[0] - S_2[0]
        return (y_12 * (M[0] - S_2[0]) - x_12 * (M[1] - S_2[1])) / D_l
    
    def w(self, i: int, M: np.ndarray) -> np.ndarray:
        if i in self.w_tilde_functions__:
            return self.w_functions__[i](M)
        else:
            raise ValueError(f"Invalid index {i}. Valid indices are 1, 2, or 3.")
    
    
    
    def w_tilde_1__(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)  # Assure qu'il est au moins 2D
        result = - M[:, 0] - M[:, 1] + 1
        return result if M.shape[0] > 1 else result[0]  # Renvoie un scalaire si M contient un seul point

    def w_tilde_2__(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)
        result = M[:, 0]
        return result if M.shape[0] > 1 else result[0]

    def w_tilde_3__(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)
        result = M[:, 1]
        return result if M.shape[0] > 1 else result[0]

    def w_tilde(self, i: int, M: np.ndarray) -> np.ndarray:
        if i in self.w_tilde_functions__:
            return self.w_tilde_functions__[i](M)
        else:
            raise ValueError(f"Invalid index {i}. Valid indices are 1, 2, or 3.")

    # New functions for gradients
    def grad_w_tilde_1__(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)
        return np.full((M.shape[0], 2), [-1, -1])  # Gradient is constant

    def grad_w_tilde_2__(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)
        return np.full((M.shape[0], 2), [1, 0])  # Gradient is constant

    def grad_w_tilde_3__(self, M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)
        return np.full((M.shape[0], 2), [0, 1])  # Gradient is constant

    def grad_w_tilde(self, i: int, M: np.ndarray) -> np.ndarray:
        if i in self.grad_w_tilde_functions__:
            grad = self.grad_w_tilde_functions__[i](M)
            # If the input M was 1D (i.e., a single point), return a 1D gradient (shape: (2,))
            # print(f"M = {M}, {M.size = }")
            if M.size == 2:
                return grad[0]  # Return the first (and only) gradient vector
            else:
                return grad 
            # return self.grad_w_tilde_functions__[i](M)
        else:
            raise ValueError(f"Invalid index {i}. Valid indices are 1, 2, or 3.")

    def F_l(self, M: np.ndarray, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M)  # Handle single and multiple points
        B_l = np.stack([S_2 - S_1, S_3 - S_1], axis=-1)
        return M @ B_l.T + S_1

    def D_l(self, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> float:
        return (S_2[0] - S_1[0]) * (S_3[1] - S_1[1]) - (S_2[1] - S_1[1]) * (S_3[0] - S_1[0])

    def A_l(self, S_1: np.ndarray, S_2: np.ndarray, S_3: np.ndarray) -> np.ndarray:
        return np.array([[S_3[1] - S_1[1], - (S_2[1] - S_1[1])], [-(S_3[0] - S_1[0]), S_2[0] - S_1[0]]])


# Test the extended class
if __name__ == '__main__':
    # Initialize the class
    bc = ReferenceElementBarycentricCoordinates()
    
    S_1 = np.array([0, 0])
    S_2 = np.array([1, 0])
    S_3 = np.array([0, 1])
    vertices = np.array([S_1, S_2, S_3])
    vertices_prime = np.array([[1, 0], [1, 1], [0, 1]])
    
    # # A = np.stack([S_3 - S_1, S_2 - S_1], axis=-1)
    # A = np.stack([S_2 - S_1, S_3 - S_1], axis=-1)
    # print(A)
    # # A = np.stack([S_3 - S_1, S_2 - S_1], axis=-1)
    # # print(A)
    # B = np.array([[S_3[1] - S_1[1], - (S_2[1] - S_1[1])], [-(S_3[0] - S_1[0]), S_2[0] - S_1[0]]])
    # print(B)
    
    
    print(bc.A_l(*vertices))
    print(bc.A_l(*vertices_prime))

    # # Test w_tilde functions for batch of points
    # M_ref_batch = np.array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.1, 0.8]])
    # print(f"{barycentric_coords.grad_w_tilde(1, S_1) = }")
    # for i in range(1, 4):
    #     result_batch = barycentric_coords.w_tilde(i, M_ref_batch)
    #     print(f"w_tilde_{i} for batch of points:\n{result_batch}")

    # # Test gradients
    # for i in range(1, 4):
    #     grad_batch = barycentric_coords.grad_w_tilde(i, M_ref_batch)
    #     print(f"grad_w_tilde_{i} for batch of points:\n{grad_batch}")