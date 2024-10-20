import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from poisson_equation.mesh import CustomTwoDimensionMesh
# from poisson_equation.utils import *
# import poisson_equation.utils.barycentric_coordinates as bc
from poisson_equation.utils.reference_element_barycentric_coordinates import ReferenceElementBarycentricCoordinates
from poisson_equation.utils.quadratures.gauss_legendre_6_points import quadrature_points, quadrature_weights


# | -\Delta u + u= f,   dans \Omega
# % |         u = 0,   sur le bord
bc = ReferenceElementBarycentricCoordinates()

"""
Class that represents the following Poisson problem : 
| Find u \in H_0^1(\Omega) such that :
| \rho u - \nabla (A \Delta u) = f, in \Omega
| A \nabla u \dot n = 0, over \partial\Omega
"""

def is_invertible(matrix):
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def matrix_info(matrix):
    info = []
    

class PoissonProblem:
    def __init__(
        self,
        mesh: CustomTwoDimensionMesh,
        f: callable,
        diffusion_tensor: callable,
        rho: callable) -> None:
        self.mesh = mesh
        
        self.f = f
        self.diffusion_tensor = diffusion_tensor
        self.rho = rho
        
       
    def construct_elementary_rigidity_matrix(self, triangle: np.ndarray) -> np.ndarray:
        """ 
        @brief Construct the elementary rigidity matrix (K) on the triangle.
        @param triangle: considered triangle (list of 3 indices).
        @note sigma is supposed here to be homogeneous on the triangle.
        @return: the elementary rigidity matrix on the triangle. 
        """
        # Calcul de la mesure du triangle (aire en 2D, volume en 3D)
        print(f"{bc.D_l(*self.mesh.node_coords[triangle]) = }")
        D = np.abs(bc.D_l(*self.mesh.node_coords[triangle]))
        
        # Initialisation de la matrice de masse élémentaire
        M = np.zeros((3, 3), dtype=float)
        
        # Inverse de la transposée du jacobien du mapping F_l
        A_l = bc.A_l(*self.mesh.node_coords[triangle])
        
        # Calcul des gradients transformés (A_l @ grad) pour chaque fonction test
        lst_A_l_grad = np.array([A_l @ grad for grad in bc.grad_w_tilde.values()])  # shape (3, 2)
        
        # Calcul du tenseur de diffusion pour chaque point de quadrature, déjà pondéré par les poids
        mat_a = np.array([omega_q*self.diffusion_tensor(*bc.F_l(M_q, *self.mesh.node_coords[triangle]).T) 
                        for omega_q, M_q in zip(quadrature_weights, quadrature_points)])  # shape (num_quad_points, 2, 2)
        
        # Application des matrices mat_a à chaque gradient
        a_applied = np.einsum('ijk,lk->ilj', mat_a, lst_A_l_grad)  # shape (num_quad_points, 3, 2)
        
        # Calcul vectorisé de la matrice élémentaire
        # M = np.einsum('ilj,ikj->lk', a_applied, lst_A_l_grad)
        M = np.einsum('ilj,lj->il', a_applied, lst_A_l_grad)
        
        # Normalisation par la mesure du triangle
        M /= D
        # print(M)
        return M

    def construct_elementary_mass_matrix(self, triangle: np.ndarray) -> np.ndarray:
        """ 
        @brief Construct the elementary mass matrix (M) on the triangle.
        @param triangle: considered triangle (list of 3 indices).
        @return: the elementary mass matrix on the triangle. 
        """
        triangle_nodes = pb.mesh.node_coords[triangle]

        RHO = self.rho(*bc.F_l(quadrature_points, *triangle_nodes).T)
        W_i = np.array([bc.w_tilde(i + 1, bc.F_l(quadrature_points, *triangle_nodes)) for i in range(3)])
        D_l = bc.D_l(*triangle_nodes)
        # M = np.zeros((3,3))
        
        # for i in range(3):
        #     for j in range(3):
        #         M[i,j] = np.sum(quadrature_weights * RHO * W_i[i] * W_i[j])
        
        # M *= np.abs(D_l)
        
        # print(f"{M = }")
        
        # triangle = pb.mesh.tri_nodes[0]
        # triangle_nodes = pb.mesh.node_coords[triangle]
        # RHO = rho(*bc.F_l(quadrature_points, *triangle_nodes).T)
        # W_i = np.array([bc.w_tilde(i + 1, bc.F_l(quadrature_points, *triangle_nodes)) for i in range(3)])
        # D_l = np.abs(bc.D_l(*triangle_nodes))

        # Using np.einsum for the matrix multiplication
        M = np.einsum('k, ik, jk -> ij', quadrature_weights * RHO, W_i, W_i)
        M *= D_l
        print(f"{M = }")
        return(M)
    
    
    def construct_A_0(self, show_info: bool = False) -> np.ndarray:
        """ 
        @brief Construct the matrix of the problem before the pseudo-elimination.
        @param show_info: If True show the matrices properties (sysmetric, hermitian, inversible).
        @return: the considered matrix.
        """
        M = sp.sparse.lil_matrix((self.mesh.num_nodes, self.mesh.num_nodes), dtype = float)
        K = sp.sparse.lil_matrix((self.mesh.num_nodes, self.mesh.num_nodes), dtype = float)
        
        
        for triangle in self.mesh.tri_nodes:
            M_elem = self.construct_elementary_mass_matrix(triangle)
            K_elem = self.construct_elementary_rigidity_matrix(triangle)
            
            for i in range(0, 3):
                I = triangle[i]
                for j in range(0, 3):
                    J = triangle[j]
                    M[I, J] += M_elem[i, j]
                    K[I, J] += K_elem[i, j]
                    
        # Conversion to CSR format for better performance
        M = M.tocsr()
        K = K.tocsr()
        if show_info :
            mat_M = M.toarray()
            mat_K = K.toarray()
            print(f"M est :")
            print(f"| symetrique : {sp.linalg.issymmetric(mat_M)}")
            print(f"| hermitienne : {sp.linalg.ishermitian(mat_M)}")
            print(f"| inversible : {is_invertible(mat_M)}")
            print(f"K est :")
            print(f"| symetrique : {sp.linalg.issymmetric(mat_K)}")
            print(f"| hermitienne : {sp.linalg.ishermitian(mat_K)}")
            print(f"| inversible : {is_invertible(mat_K)}")
            
            print("K + M est :")
            print(f"| symetrique : {sp.linalg.issymmetric(mat_M + mat_K)}")
            print(f"| hermitienne : {sp.linalg.ishermitian(mat_M + mat_K)}")
            print(f"| inversible : {is_invertible(mat_M + mat_K)}")
        return(M, K)
    
    def construct_L_0(self) -> np.ndarray:
        """ 
        @brief Construct the right-hand side member of the discrete variationnal formulation.
        @return: L
        """
        L = self.f(*self.mesh.node_coords.T)
        return(self.M @ L)
    
    def boundary_condition_elimination(self):
        """ 
        @brief Eliminate the boundary terms in A by applying Dirichlet boundary conditions..
        @exception ValueError: if the border reference is not the largest.
        """
        border_ref = self.mesh.labels['$\\partial\\Omega$'] 

        if border_ref != max(self.mesh.labels.values()):
            raise ValueError("The border ∂Ω does not have the largest physical reference. Please modify the mesh to ensure the border has the highest reference tag by adding the physical reference on it at least.")
        
        # Obtenir les indices des nœuds de bord
        node_refs = self.mesh.node_refs
        boundary_nodes = np.where(node_refs == border_ref)[0]
        
        # Appliquer les conditions de Dirichlet (mettre 1 sur la diagonale et 0 ailleurs)
        self.A = self.A.tolil()
        for node in boundary_nodes:
            # Remettre à zéro toutes les entrées de la ligne et de la colonne, sauf la diagonale
            self.A[node, :] = 0.
            self.A[:, node] = 0.
            self.A[node, node] = 1.
            
            # Fixer la valeur de la condition de Dirichlet dans le second membre (0 ici)
            self.L[node] = 0.  # On suppose que les conditions de Dirichlet sont homogènes
        
        
        self.A = self.A.tocsr()
            # print(N_0)
    

    def solve(self):
        U = sp.sparse.linalg.spsolve(self.A, self.L)
        return(U)
    
    def display_solution(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        contour = ax.tricontourf(self.mesh.node_coords[:, 0], self.mesh.node_coords[:, 1], self.mesh.tri_nodes, self.solution)
        fig.colorbar(contour, ax=ax, orientation='vertical', label=r'$u_h$')

        ax.set(xlabel = "$x$", ylabel = "$y$", aspect = "equal")
        plt.show()
        
        
if __name__ == '__main__':
    mesh = CustomTwoDimensionMesh("poisson_equation/geometries/two_domains_rectangle.msh")
    # mesh.display()
    def diffusion_tensor(x, y):
        return(np.eye(2))
        # return(np.array([[1, 1], [1, 1]]))
    
    def f(x, y):
        return(2*np.ones_like(x))
    
    def rho(x, y):
        return(2*np.ones_like(x))
    
    pb = PoissonProblem(mesh=mesh, f = f, diffusion_tensor=diffusion_tensor, rho = rho)
    
    # triangle = pb.mesh.tri_nodes[0]
    
    pb.construct_elementary_mass_matrix(pb.mesh.tri_nodes[0])
    pb.construct_elementary_rigidity_matrix(pb.mesh.tri_nodes[0])
    # print(f"{triangle = }")
    

    
    
    # F_l_points = bc.F_l(quadrature_points, *triangle_nodes)
    # RHO = rho(*F_l_points.T)  # RHO est un tableau de la forme (N,) où N est le nombre de points de quadrature
    # RHO = RHO[:, np.newaxis]
    # print(f"{RHO.shape = }")
    # # Calcul des W_i (barycentriques)
    # W_i = np.array([bc.w_tilde(i + 1, F_l_points) for i in range(3)])  # (3, N)
    # print(f"{W_i.shape = }")
    # quadrature_weights = quadrature_weights[:, np.newaxis]
    # print(f"{(quadrature_weights * RHO).shape = }")
    
    # # Calcul de M en utilisant la vectorisation
    # # M = np.einsum('i,j,k->ij', W_i, RHO * quadrature_weights)
    # # M = np.einsum('ij,j->ij', W_i, RHO * quadrature_weights)
    # # M = np.einsum('ij,jk->ik', W_i, RHO * quadrature_weights)
    # # M = np.einsum('i,jk->ij', W_i, RHO * quadrature_weights)
    # M = np.einsum('k, ik, jk -> ij', quadrature_weights * RHO, W_i, W_i)
    # # M_temp = np.einsum('i,jk->ij', W_i, RHO * quadrature_weights) 
    # M *= np.abs(D_l)
    # print(f"{M.shape = }")
    
    
    # pb.boundary_condition_elimination()
    # A = pb.construct_A_0(show_info = False)
    # A = A.toarray()
    # print(A.toarray())
    # K_l = pb.K_l(mesh.tri_nodes[10])
    # M_l = pb.M_l(mesh.tri_nodes[10])
    # print(M_l)
    # mesh.display()
    
    # pb = StationnaryProblem
    pass
    