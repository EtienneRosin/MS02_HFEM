from pathlib import Path
import h5py
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
from hfem.core.io import MeshData, Solution, FEMMatrices
import scipy.sparse as sparse
from hfem.poisson_problems.configs.problem_type import ProblemType


class FEMDataManager:
    def __init__(self, root_dir: Union[str, Path] = "simulation_data"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self,
         name: str,
         solution: 'Solution',
         mesh: 'MeshData',
         matrices: Optional['FEMMatrices'] = None,
         metadata: Optional[Dict[str, Any]] = None) -> Path:
    
        problem_type = (ProblemType.from_str(solution.problem_type) 
                    if isinstance(solution.problem_type, str) 
                    else solution.problem_type)
        
        filepath = self.root_dir / f"{solution.problem_type}" / f"{name}.h5"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['creation_date'] = datetime.now().isoformat()
            meta.attrs['problem_type'] = str(problem_type)
            if metadata:
                for key, value in metadata.items():
                    meta.attrs[key] = value
            
            # Solution
            solution_group = f.create_group('solution')
            if isinstance(solution.data, dict):
                # Cas des problèmes de cellule ou autres avec données multiples
                data_group = solution_group.create_group('data')
                for key, value in solution.data.items():
                    data_group.create_dataset(key, data=value)
            else:
                # Cas standard
                solution_group.create_dataset('data', data=solution.data)
            solution_group.create_dataset('metadata', data=str(solution.metadata))
            
            # Mesh
            mesh_dict = mesh.to_dict()
            for key, value in mesh_dict.items():
                f.create_dataset(f'mesh/{key}', data=value)
            
            # Matrices
            if matrices is not None:
                matrices_group = f.create_group('matrices')
                for matrix_name, matrix in matrices.__dict__.items():
                    if isinstance(matrix, sparse.spmatrix):
                        matrix = matrix.tocsr()
                        matrix_group = matrices_group.create_group(matrix_name)
                        matrix_group.create_dataset('data', data=matrix.data)
                        matrix_group.create_dataset('indices', data=matrix.indices)
                        matrix_group.create_dataset('indptr', data=matrix.indptr)
                        matrix_group.attrs['shape'] = matrix.shape
        
        return filepath

    def load(self, filepath: Path) -> Tuple['Solution', 'MeshData', Optional['FEMMatrices']]:
        with h5py.File(filepath, 'r') as f:
            # Solution
            if isinstance(f['solution/data'], h5py.Group):
                # Cas des problèmes de cellule ou autres avec données multiples
                solution_data = {
                    key: f['solution/data'][key][()] 
                    for key in f['solution/data'].keys()
                }
            else:
                # Cas standard
                solution_data = f['solution/data'][()]
            
            solution_metadata = eval(f['solution/metadata'][()])
            problem_type = f['metadata'].attrs['problem_type']
            
            solution = Solution(
                data=solution_data,
                problem_type=problem_type,
                metadata=solution_metadata
            )
            
            # Mesh
            mesh_data = {key: f['mesh'][key][()] for key in f['mesh'].keys()}
            mesh = MeshData.from_dict(mesh_data)
            
            # Matrices
            matrices = None
            if 'matrices' in f:
                matrix_dict = {}
                for matrix_name in f['matrices']:
                    matrix_group = f['matrices'][matrix_name]
                    matrix_dict[matrix_name] = sparse.csr_matrix(
                        (matrix_group['data'][()],
                        matrix_group['indices'][()],
                        matrix_group['indptr'][()]),
                        shape=matrix_group.attrs['shape']
                    )
                matrices = FEMMatrices(**matrix_dict)
            
        return solution, mesh, matrices