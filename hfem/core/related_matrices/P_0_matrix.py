from hfem.mesh_manager import CustomTwoDimensionMesh

import scipy as sp
import numpy as np

def assemble_P_0(mesh: CustomTwoDimensionMesh, boundary_labels: str | list[str] = '$\\partial\\Omega$'):
        """Construct the projection matrix P for Dirichlet boundary conditions."""
        
        if isinstance(boundary_labels, str):
            boundary_labels = [boundary_labels]
        boundary_indices = np.hstack([np.where(mesh.node_refs == mesh.labels[label])[0] for label in boundary_labels])
        
        on_border_ref = mesh.labels['$\\partial\\Omega$']
        interior_indices = np.where(mesh.node_refs != on_border_ref)[0]
        N_0 = len(interior_indices)
        N = mesh.num_nodes

        P = sp.sparse.lil_matrix((N_0, N), dtype=float)
        for i, j in enumerate(interior_indices):
            P[i, j] = 1

        # print(P.toarray())
        return P.tocsr()
    
if __name__ == '__main__':
    mesh_file = "meshes/test_mesh.msh"
    from hfem.mesh_manager.geometries import rectangular_mesh
    rectangular_mesh(h=0.5, L_x=1, L_y=1, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    # mesh.display()
    
    P_0 = assemble_P_0(mesh)
    from pprint import pprint
    pprint(P_0.toarray())