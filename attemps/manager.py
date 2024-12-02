from pathlib import Path
import h5py
import numpy as np
from typing import Optional, Dict, Any, Type, Tuple, Union
from datetime import datetime

from .data_structures import MeshData, FEMMatrices
from .protocols import SolutionProtocol
from .solutions import CellProblemSolution, HomogenizedSolution, StandardPoissonSolution

class FEMDataManager:
    """Manager for FEM data I/O operations."""
    
    SOLUTION_TYPES = {
        'neumann': StandardPoissonSolution,
        'dirichlet': StandardPoissonSolution,
        'periodic': StandardPoissonSolution,
        'standard': StandardPoissonSolution,
        'cell': CellProblemSolution,
        'homogenized': HomogenizedSolution
    }
    
    def __init__(self, root_dir: Union[str, Path] = "simulation_data/results"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_filepath(self, 
                       problem_type: str,
                       save_name: Union[str, Path]
                       ) -> Path:
        """Build standardized filepath based on parameters."""
        # Créer un nom de fichier basé sur les paramètres clés
        # param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        filename = f"{save_name}.h5"
        
        # Organiser dans des sous-dossiers par type de problème
        problem_dir = self.root_dir / problem_type
        problem_dir.mkdir(exist_ok=True)
        
        return problem_dir / filename
    
    def save_solution(self,
                      save_name: Union[str, Path],
                      solution: SolutionProtocol,
                      mesh: MeshData,
                      params: Optional[Dict[str, Any]] = {},
                      matrices: Optional[FEMMatrices] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save solution data to HDF5 file.
        
        Parameters
        ----------
        solution : SolutionProtocol
            Solution object implementing to_dict method
        mesh : MeshData
            Mesh data
        params : Dict[str, Any]
            Problem parameters
        matrices : Optional[FEMMatrices]
            FEM matrices if needed for error computations
        metadata : Optional[Dict[str, Any]]
            Additional metadata to store
            
        Returns
        -------
        Path
            Path to saved file
        """
        filepath = self._build_filepath(solution.problem_type, save_name)
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.attrs['creation_date'] = datetime.now().isoformat()
            meta.attrs['problem_type'] = solution.problem_type
            if metadata:
                for key, value in metadata.items():
                    meta.attrs[key] = value
            
            # Save parameters
            param_group = f.create_group('parameters')
            for key, value in params.items():
                param_group.attrs[key] = value
            
            # Save mesh data
            mesh_group = f.create_group('mesh')
            for key, value in mesh.to_dict().items():
                mesh_group.create_dataset(key, data=value)
            
            # Save solution
            sol_group = f.create_group('solution')
            for key, value in solution.to_dict().items():
                if isinstance(value, dict):
                    sub_group = sol_group.create_group(key)
                    for k, v in value.items():
                        sub_group.create_dataset(k, data=v)
                else:
                    sol_group.create_dataset(key, data=value)
            
            # Save matrices if provided
            if matrices is not None:
                mat_group = f.create_group('matrices')
                mat_dict = matrices.to_dict()
                for matrix_name, matrix_data in mat_dict.items():
                    matrix_group = mat_group.create_group(matrix_name)
                    for key, value in matrix_data.items():
                        matrix_group.create_dataset(key, data=value)
        
        return filepath
    
    def load_solution(self, 
                     filepath: Path) -> Tuple[SolutionProtocol, MeshData, Dict[str, Any], Optional[FEMMatrices]]:
        """
        Load solution from HDF5 file.
        
        Parameters
        ----------
        filepath : Path
            Path to HDF5 file
            
        Returns
        -------
        Tuple[SolutionProtocol, MeshData, Dict[str, Any], Optional[FEMMatrices]]
            Solution, mesh data, parameters, and optionally matrices
        """
        with h5py.File(filepath, 'r') as f:
            # Load problem type
            problem_type = f['metadata'].attrs['problem_type']
            solution_class = self.SOLUTION_TYPES[problem_type]
            
            # Load solution data
            sol_data = {}
            for key in f['solution'].keys():
                if isinstance(f['solution'][key], h5py.Group):
                    sol_data[key] = {k: f['solution'][key][k][()] 
                                   for k in f['solution'][key].keys()}
                else:
                    sol_data[key] = f['solution'][key][()]
            solution = solution_class.from_dict(sol_data)
            
            # Load mesh data
            mesh_data = {key: f['mesh'][key][()] for key in f['mesh'].keys()}
            # print(f"{mesh_data = }")
            mesh = MeshData.from_dict(mesh_data)
            
            # Load parameters
            params = dict(f['parameters'].attrs)
            
            # Load matrices if they exist
            matrices = None
            if 'matrices' in f:
                mat_data = {}
                for matrix_name in f['matrices']:
                    mat_data[matrix_name] = {
                        key: f['matrices'][matrix_name][key][()]
                        for key in f['matrices'][matrix_name]
                    }
                matrices = FEMMatrices.from_dict(mat_data)
            
        return solution, mesh, params, matrices
    
    def list_solutions(self, 
                      problem_type: Optional[str] = None) -> Dict[str, list[Path]]:
        """
        List all available solutions.
        
        Parameters
        ----------
        problem_type : Optional[str]
            If provided, only list solutions for this problem type
            
        Returns
        -------
        Dict[str, list[Path]]
            Dictionary mapping problem types to lists of solution files
        """
        if problem_type:
            problem_dir = self.root_dir / problem_type
            return {problem_type: list(problem_dir.glob("*.h5"))}
        
        solutions = {}
        for problem_dir in self.root_dir.iterdir():
            if problem_dir.is_dir():
                solutions[problem_dir.name] = list(problem_dir.glob("*.h5"))
        return solutions