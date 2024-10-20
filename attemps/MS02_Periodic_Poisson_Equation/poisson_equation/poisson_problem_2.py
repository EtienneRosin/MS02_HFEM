import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm


from poisson_equation.mesh import CustomTwoDimensionMesh
from poisson_equation.utils.reference_element_barycentric_coordinates import ReferenceElementBarycentricCoordinates
# from poisson_equation.utils.quadratures.gauss_legendre_6_points import quadrature_points, quadrature_weights
from poisson_equation.utils.quadratures.gauss_lobatto_4_points import quadrature_points, quadrature_weights
from poisson_equation.geometries.rectangle_mesh import create_rectangle_mesh


bc = ReferenceElementBarycentricCoordinates()

class PoissonProblem:
    def __init__(
        self, 
        mesh: CustomTwoDimensionMesh, 
        f: callable, 
        diffusion_tensor: callable, 
        rho: callable,
        exact_solution: callable = None) -> None:
        self.mesh = mesh
        self.f = f
        self.diffusion_tensor = diffusion_tensor
        self.rho = rho
        
        self.M, self.K = self.construct_A_0()
        self.A = self.M + self.K
        
        self.L = self.construct_L_0()
        
        self.solution = self.solve()
        
        self.relative_L2_norm = None
        self.relative_H1_norm = None
        
        if not exact_solution is None:
            
            U = exact_solution(*self.mesh.node_coords.T)
            self.exact_solution = U
            U_h = self.solution
            self.relative_L2_norm = np.sqrt(np.dot(self.M @ (U - U_h), U - U_h))/ np.sqrt(np.dot(self.M @ U, U))
            print(f"L^2 relative error : {self.relative_L2_norm}")
            self.relative_H1_norm = np.sqrt(np.dot(self.K @ (U - U_h), U - U_h))/ np.sqrt(np.dot(self.K @ U, U))
            print(f"H^1 relative error : {self.relative_H1_norm}")

    def construct_elementary_rigidity_matrix(self, triangle: np.ndarray) -> np.ndarray:
        triangle_nodes = self.mesh.node_coords[triangle]
        D_l = np.abs(bc.D_l(*triangle_nodes))
        A_l = bc.A_l(*triangle_nodes)
        
        # grad_W_i = np.array([bc.grad_w_tilde(i + 1, quadrature_points) for i in range(3)])
        # mat_a = np.array([self.diffusion_tensor(*bc.F_l(S_q, *triangle_nodes).T) for S_q in quadrature_points])
        K = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                for omega_q, S_q in zip(quadrature_weights, quadrature_points):
                    # K[i,j] += omega_q * np.dot(mat_a[q] @ (A_l @ grad_W_i[i, q]), A_l @ grad_W_i[j, q])
                    K[i,j] += omega_q * np.dot(np.eye(2) @ A_l @ bc.grad_w_tilde(i+1, S_q), A_l @ bc.grad_w_tilde(j+1, S_q)) 
        return K / D_l

    # def construct_elementary_mass_matrix(self, triangle: np.ndarray) ->  np.ndarray:
    #     triangle_nodes = self.mesh.node_coords[triangle]
    #     RHO = self.rho(*bc.F_l(quadrature_points, *triangle_nodes).T)
    #     W_i = np.array([bc.w_tilde(i + 1, quadrature_points) for i in range(3)])
    #     D_l = np.abs(bc.D_l(*triangle_nodes))
        
    #     M = np.zeros((3,3))
    #     for i in range(3):
    #         for j in range(3):
    #             M[i,j] = np.sum(quadrature_weights * RHO * W_i[i] * W_i[j])

    #     return D_l * M
    def construct_elementary_mass_matrix(self, triangle: np.ndarray) ->  np.ndarray:
        triangle_nodes = self.mesh.node_coords[triangle]
        D_l = np.abs(bc.D_l(*triangle_nodes))
        M = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                for node in triangle_nodes :
                    # M[i,j] += self.rho(*node) * bc.w_tilde(i + 1, node) * bc.w_tilde(j + 1, node)
                    M[i,j] += self.rho(*node) * bc.w_tilde(i + 1, node) * bc.w_tilde(j + 1, node)
                # M[i,j] += (D_l/6) * np.sum(self.rho(*triangle_nodes.T) * bc.w_tilde(i + 1, triangle_nodes.T) * bc.w_tilde(j + 1, triangle_nodes.T))
                # for S_T in triangle_nodes:
                #     M[i,j] = np.sum(quadrature_weights * RHO * W_i[i] * W_i[j])
                # M[i,j] = np.sum(quadrature_weights * RHO * W_i[i] * W_i[j])

        return (D_l/6) * M

    def construct_A_0(self, show_info: bool = False) -> np.ndarray:
        M = sp.sparse.lil_matrix((self.mesh.num_nodes, self.mesh.num_nodes), dtype=float)
        K = sp.sparse.lil_matrix((self.mesh.num_nodes, self.mesh.num_nodes), dtype=float)
        for triangle in tqdm(self.mesh.tri_nodes):
            M_elem = self.construct_elementary_mass_matrix(triangle)
            K_elem = self.construct_elementary_rigidity_matrix(triangle)
            for i in range(3):
                I = triangle[i]
                for j in range(3):
                    J = triangle[j]
                    M[I, J] += M_elem[i, j]
                    K[I, J] += K_elem[i, j]
        M = M.tocsr()
        K = K.tocsr()
        
        return M, K

    def construct_L_0(self) -> np.ndarray:
        return self.M @ self.f(*self.mesh.node_coords.T)


    def solve(self):
        U = sp.sparse.linalg.spsolve(self.M + self.K, self.L)
        residual = np.linalg.norm((self.M + self.K).toarray() @ U - self.L)
        print(f"Residual norm: {residual}")
        return U

    def display_solution(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        contour = ax.tricontourf(self.mesh.node_coords[:, 0], self.mesh.node_coords[:, 1], self.mesh.tri_nodes, self.solution)
        fig.colorbar(contour, ax=ax, orientation='vertical', label=r'$u_h$')
        ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
        plt.show()
        
    def display_error(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        contour = ax.tricontourf(self.mesh.node_coords[:, 0], self.mesh.node_coords[:, 1], self.mesh.tri_nodes, self.solution - self.exact_solution)
        fig.colorbar(contour, ax=ax, orientation='vertical', label=r'$u_h$')
        ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
        plt.show()



if __name__ == '__main__':
    h = 0.05
    create_rectangle_mesh(h = 0.1, L_x = 2, L_y = 1, save_name = "attemps/MS02_Periodic_Poisson_Equation/poisson_equation/geometries/rectangle.msh")
    
    mesh = CustomTwoDimensionMesh("attemps/MS02_Periodic_Poisson_Equation/poisson_equation/geometries/rectangle.msh")
    mesh.display()
    def diffusion_tensor(x, y):
        return np.eye(2)

    def u(x,y):
        return np.cos(np.pi * x) * np.cos(2 * np.pi * y)
    
    def f(x, y):
        return (1 + 5 * np.pi ** 2) * u(x, y)

    def rho(x, y):
        return 1

    pb = PoissonProblem(mesh=mesh, f=f, diffusion_tensor=diffusion_tensor, rho=rho, exact_solution=u)
    pb.display_solution()
    fig = plt.figure()
    ax = fig.add_subplot()
    solution = u(*mesh.node_coords.T)
    contour = ax.tricontourf(mesh.node_coords[:, 0], mesh.node_coords[:, 1], mesh.tri_nodes, solution)
    fig.colorbar(contour, ax=ax, orientation='vertical', label=r'$u_h$')
    ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    plt.show()
    
    
    
    # S_1 = np.array([0, 0])
    # S_2 = np.array([1, 0])
    # S_3 = np.array([0, 1])
    # vertices = np.array([S_1, S_2, S_3])
    # D_l = np.abs(bc.D_l(*vertices))
    # A_l = bc.A_l(*vertices)
    # # print(f"{A_l = }, {bc.grad_w_tilde(0 + 1, quadrature_points[0]) = }")
    
    # K_exact = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         for q, omega_q in enumerate(quadrature_weights):
    #             # print(omega_q * bc.w_tilde(i + 1, quadrature_points[q]) * bc.w_tilde(j + 1, quadrature_points[q]))
    #             # K_exact[i,j] += omega_q * bc.w_tilde(i + 1, quadrature_points[q])[0] * bc.w_tilde(j + 1, quadrature_points[q])[0]
    #             K_exact[i,j] += omega_q * np.dot(diffusion_tensor(*quadrature_points[q]) @ (A_l @ bc.grad_w_tilde(i + 1, quadrature_points[q])), A_l @ bc.grad_w_tilde(j + 1, quadrature_points[q]))
    # K_exact /= D_l
    
    # print(K_exact, "\n")
    
    
    # grad_W_i = np.array([bc.grad_w_tilde(i + 1, quadrature_points) for i in range(3)])
    # # get the diffusion tensor to each quadrature points
    # mat_a = np.array([diffusion_tensor(*bc.F_l(S_q, *vertices).T) for S_q in quadrature_points])
    # # print(diffusion_tensor(*bc.F_l(quadrature_points, *vertices).T))
    # K = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         # K[i,j] = np.sum(quadrature_weights * np.dot(mat_a @ (A_l @ grad_W_i[i]), A_l @ grad_W_i[j]))
    #         for q, omega_q in enumerate(quadrature_weights):
    #             K[i,j] += omega_q * np.dot(mat_a[q] @ (A_l @ grad_W_i[i, q]), A_l @ grad_W_i[j, q])
                
    # K /= D_l
    # print(K)
    
    
    # RHO = rho(*bc.F_l(quadrature_points, *vertices).T)
    # # print(RHO)
    # W_i = np.array([bc.w_tilde(i + 1, quadrature_points) for i in range(3)])
    # # print(W_i)
    # D_l = np.abs(bc.D_l(*vertices))
    
    # # print(quadrature_weights * RHO * W_i[1] * W_i[1])
    # M = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         M[i,j] = np.sum(quadrature_weights * RHO * W_i[i] * W_i[j])
    
    # # M = np.einsum('k, ik, jk -> ij', quadrature_weights * RHO, W_i, W_i)
    # M *= D_l
    # print(M)
    
    
    
    # solution = u(*mesh.node_coords.T)
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # contour = ax.tricontourf(mesh.node_coords[:, 0], mesh.node_coords[:, 1], mesh.tri_nodes, solution)
    # fig.colorbar(contour, ax=ax, orientation='vertical', label=r'$u_h$')
    # ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    # plt.show()
    
    
    
    
    # triangle = pb.mesh.tri_nodes[0]
    # triangle_nodes = pb.mesh.node_coords[triangle]
    # D_l = np.abs(bc.D_l(*triangle_nodes))
    # A_l = bc.A_l(*triangle_nodes)
    
    # grad_W_i = np.array([bc.grad_w_tilde(i + 1, bc.F_l(quadrature_points, *triangle_nodes)) for i in range(3)])
    
    # # get the diffusion tensor to each quadrature points
    # mat_a = np.array([omega_q * diffusion_tensor(*bc.F_l(M_q, *triangle_nodes).T) 
    #                     for omega_q, M_q in zip(quadrature_weights, quadrature_points)])
    
    # K = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         for q in range(len(quadrature_points)):
    #             K[i,j] += np.dot(mat_a[q] @ grad_W_i[i, q], grad_W_i[j, q])
    # K /= D_l
    # print(K)
    
    
    # K = np.einsum('ilj,lj->il', a_applied, lst_A_l_grad)
    # print(f"{K.shape = }")
    # pb.display_solution()
    
    # pb.construct_elementary_mass_matrix(pb.mesh.tri_nodes[0])
    # pb.construct_elementary_rigidity_matrix(pb.mesh.tri_nodes[0])
