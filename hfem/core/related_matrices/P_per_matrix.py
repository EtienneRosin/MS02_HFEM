

from hfem.mesh_manager import CustomTwoDimensionMesh

import scipy as sp

def assemble_P_per(mesh: CustomTwoDimensionMesh, boundary_labels: str | list[str] = '$\\partial\\Omega$', tolerance: float = 1e-10):
        """Construct the projection matrix P_per for periodic boundary conditions."""
        if isinstance(boundary_labels, str):
            boundary_labels = [boundary_labels]

        corner_indices, pairs_same_x, pairs_same_y, inner_indices = mesh.get_corner_and_boundary_pairs(boundary_labels, tolerance)
        
        # Initialize the projector matrix P_per
        N_inner = len(inner_indices)
        N_corner = 1  # we keep only one corner
        N_pairs_x = len(pairs_same_x)
        N_pairs_y = len(pairs_same_y)
        N_per = N_inner + N_corner + N_pairs_x + N_pairs_y
        P_per = sp.sparse.lil_matrix((N_per, mesh.num_nodes))

        # Assign values to P_per based on inner indices, corner indices, and pairs of indices
        for n, i in enumerate(inner_indices):
            P_per[n, i] = 1

        for i in corner_indices:
            P_per[N_inner, i] = 1

        for n, (i, j) in enumerate(pairs_same_x):
            P_per[n + N_inner + N_corner, i] = P_per[n + N_inner + N_corner, j] = 1

        for n, (i, j) in enumerate(pairs_same_y):
            P_per[n + N_inner + N_corner + N_pairs_x, i] = P_per[n + N_inner + N_corner + N_pairs_x, j] = 1

        return P_per.tocsr()
    

if __name__ == '__main__':
    mesh_file = "meshes/test_mesh.msh"
    from hfem.mesh_manager.geometries import rectangular_mesh
    rectangular_mesh(h=0.5, L_x=1, L_y=1, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    # mesh.display()
    
    P_per = assemble_P_per(mesh)
    from pprint import pprint
    pprint(P_per.toarray())