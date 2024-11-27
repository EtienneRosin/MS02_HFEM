"""Module providing some matrix information functions"""
import numpy as np
import scipy as sp


def is_invertible(matrix):
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def is_symmetric(matrix):
    return sp.linalg.issymmetric(matrix)

def is_hermitian(matrix):
    return sp.linalg.ishermitian(matrix)

def display_matrix_info(matrix, name: str = None):
    if isinstance(matrix, sp.sparse._csr.csr_array) or isinstance(matrix, sp.sparse._lil.lil_matrix):
        new_matrix = matrix.toarray()
    else:
        new_matrix = matrix
    name = "matrix" if name is None else name
    
    print(f"{name} is :")
    print(f"| symmetric : {is_symmetric(new_matrix)}")
    print(f"| hermitian : {is_hermitian(new_matrix)}")
    print(f"| invertible : {is_invertible(new_matrix)}")
    # print(f"{}")


        # # mat_M = M.toarray()
        # #     mat_K = K.toarray()
        #     print(f"M est :")
        #     print(f"| symetrique : {sp.linalg.issymmetric(mat_M)}")
        #     print(f"| hermitienne : {sp.linalg.ishermitian(mat_M)}")
        #     print(f"| inversible : {is_invertible(mat_M)}")
        #     print(f"K est :")
        #     print(f"| symetrique : {sp.linalg.issymmetric(mat_K)}")
        #     print(f"| hermitienne : {sp.linalg.ishermitian(mat_K)}")
        #     print(f"| inversible : {is_invertible(mat_K)}")
            
        #     print("K + M est :")
        #     print(f"| symetrique : {sp.linalg.issymmetric(mat_M + mat_K)}")
        #     print(f"| hermitienne : {sp.linalg.ishermitian(mat_M + mat_K)}")
        #     print(f"| inversible : {is_invertible(mat_M + mat_K)}")
        
if __name__ == '__main__':
    # print(f"{None}")
    
    A = np.eye((3))
    # display_matrix_info(A)
    row  = np.array([0, 0, 1, 3, 1, 0, 0])
    col  = np.array([0, 2, 1, 3, 1, 0, 0])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    # A = sp.sparse.coo_array((data, (row, col)), shape=(4, 4)).tocsr()
    # print(type(A))
    
    # A = sp.sparse.lil_matrix((3,3), dtype = float)
    # print(A.toarray())
    display_matrix_info(A)
    # print(isinstance(A, sp.sparse._csr.csr_array) or isinstance(A, sp.sparse._lil.lil_matrix))
    # print(A.toarray())