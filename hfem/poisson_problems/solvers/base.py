"""
This module provides the base class for solving Poisson problems using the finite element method.
It includes both sequential and parallel assembly strategies with automatic fallback.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import scipy.sparse as sparse
from tqdm.auto import tqdm
from multiprocessing import Pool

from hfem.core.io import FEMDataManager, FEMMatrices, Solution, MeshData
from hfem.core.related_matrices import (
    assemble_elementary_mass_matrix, 
    assemble_elementary_stiffness_matrix
)
from hfem.poisson_problems.configs.base import CorePoissonProblemsConfig


def process_chunk(chunk_data: Tuple[np.ndarray, np.ndarray, Tuple[str, Union[np.ndarray, callable]]]):
    """Process a chunk of triangles for parallel assembly.
    
    Args:
        chunk_data: Tuple containing:
            - triangles_chunk: Array of triangle indices
            - node_coords: Array of node coordinates
            - diff_tensor_info: Tuple of (tensor_type, tensor_data) where tensor_type
              is either "constant" or "function" and tensor_data is either the flattened
              tensor array or the tensor function
    
    Returns:
        Tuple containing:
            - rows: Array of row indices
            - cols: Array of column indices
            - mass_data: Array of mass matrix values
            - stiff_data: Array of stiffness matrix values
    """
    triangles_chunk, node_coords, diff_tensor_info = chunk_data
    tensor_type, tensor_data = diff_tensor_info
    
    n_triangles = len(triangles_chunk)
    rows = np.zeros(9 * n_triangles, dtype=np.int32)
    cols = np.zeros(9 * n_triangles, dtype=np.int32)
    mass_data = np.zeros(9 * n_triangles)
    stiff_data = np.zeros(9 * n_triangles)
    
    for idx, triangle in enumerate(triangles_chunk):
        nodes_coords = node_coords[triangle]
        
        # Determine diffusion tensor based on type
        if tensor_type == "constant":
            diffusion_tensor = np.array(tensor_data).reshape(2, 2)
        else:  # tensor_type == "function"
            center_x = np.mean(nodes_coords[:, 0])
            center_y = np.mean(nodes_coords[:, 1])
            diffusion_tensor = tensor_data(center_x, center_y)
        
        # Compute elementary matrices
        mass_elem = assemble_elementary_mass_matrix(nodes_coords)
        stiff_elem = assemble_elementary_stiffness_matrix(
            triangle_nodes=nodes_coords,
            diffusion_tensor=diffusion_tensor
        )
        
        # Fill arrays
        start = 9 * idx
        for i in range(3):
            for j in range(3):
                pos = start + 3 * i + j
                rows[pos] = triangle[i]
                cols[pos] = triangle[j]
                mass_data[pos] = mass_elem[i, j]
                stiff_data[pos] = stiff_elem[i, j]
                
    return rows, cols, mass_data, stiff_data


class PoissonProblem(ABC):
    """Abstract base class for solving Poisson problems.
    
    This class provides the framework for assembling and solving Poisson problems
    using the finite element method. It includes both parallel and sequential
    assembly strategies with automatic fallback.
    
    Attributes:
        config: Problem configuration containing mesh and parameters
        mass_matrix: Assembled mass matrix (CSR format)
        stiffness_matrix: Assembled stiffness matrix (CSR format)
        solution: Solution vector
        rhs: Right-hand side vector
    """
    
    def __init__(self, config: CorePoissonProblemsConfig):
        """Initialize the Poisson problem.
        
        Args:
            config: Problem configuration
        """
        self.config = config
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.solution = None
        self.rhs = None
    
    def try_parallel_assembly(self) -> bool:
        """Attempt parallel assembly of system matrices.
        
        Returns:
            bool: True if parallel assembly succeeded, False otherwise
        """
        try:
            mesh = self.config.mesh
            n_nodes = mesh.num_nodes
            n_elements = len(mesh.tri_nodes)
            
            # Convert to numpy arrays for serialization
            node_coords = np.array(mesh.node_coords)
            
            # Prepare diffusion tensor information
            if callable(self.config.diffusion_tensor):
                diff_tensor_info = ("function", self.config.diffusion_tensor)
            else:
                diff_tensor_info = ("constant", self.config.diffusion_tensor.flatten().tolist())
            
            # Determine chunk size for parallelization
            n_processes = 4
            chunk_size = max(1, n_elements // n_processes)
            
            # Prepare data chunks
            chunks_data = []
            for i in range(0, n_elements, chunk_size):
                chunk_triangles = np.array(mesh.tri_nodes[i:i + chunk_size])
                chunks_data.append((chunk_triangles, node_coords, diff_tensor_info))
            
            # Parallel processing
            print(f"Attempting parallel assembly with {len(chunks_data)} chunks...")
            with Pool(processes=n_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_chunk, chunks_data),
                    total=len(chunks_data), leave=False
                ))
            
            # Concatenate results
            print("Assembling results...")
            rows = np.concatenate([r[0] for r in results])
            cols = np.concatenate([r[1] for r in results])
            mass_data = np.concatenate([r[2] for r in results])
            stiff_data = np.concatenate([r[3] for r in results])
            
            # Build final sparse matrices
            print("Building sparse matrices...")
            self.mass_matrix = sparse.csr_matrix(
                (mass_data, (rows, cols)),
                shape=(n_nodes, n_nodes)
            )
            self.stiffness_matrix = sparse.csr_matrix(
                (stiff_data, (rows, cols)),
                shape=(n_nodes, n_nodes)
            )
            return True
            
        except Exception as e:
            print(f"Parallel assembly failed: {str(e)}")
            return False

    def sequential_assembly(self) -> None:
        """Perform sequential assembly of system matrices."""
        print("Using sequential assembly...")
        n = self.config.mesh.num_nodes
        n_elements = len(self.config.mesh.tri_nodes)
        
        # Pre-allocate arrays
        rows = np.zeros(9 * n_elements, dtype=np.int32)
        cols = np.zeros(9 * n_elements, dtype=np.int32)
        mass_data = np.zeros(9 * n_elements)
        stiff_data = np.zeros(9 * n_elements)
        
        # Sequential assembly
        for idx, triangle in enumerate(tqdm(self.config.mesh.tri_nodes, leave=False)):
            nodes_coords = self.config.mesh.node_coords[triangle]
            
            # Determine diffusion tensor
            if callable(self.config.diffusion_tensor):
                center_x = np.mean(nodes_coords[:, 0])
                center_y = np.mean(nodes_coords[:, 1])
                diffusion_tensor = self.config.diffusion_tensor(center_x, center_y)
            else:
                diffusion_tensor = self.config.diffusion_tensor
            
            # Compute elementary matrices
            mass_elem = assemble_elementary_mass_matrix(nodes_coords)
            stiff_elem = assemble_elementary_stiffness_matrix(
                triangle_nodes=nodes_coords,
                diffusion_tensor=diffusion_tensor
            )
            
            # Fill arrays
            start = 9 * idx
            for i in range(3):
                for j in range(3):
                    pos = start + 3 * i + j
                    rows[pos] = triangle[i]
                    cols[pos] = triangle[j]
                    mass_data[pos] = mass_elem[i, j]
                    stiff_data[pos] = stiff_elem[i, j]
        
        # Build final sparse matrices
        self.mass_matrix = sparse.csr_matrix(
            (mass_data, (rows, cols)),
            shape=(n, n)
        )
        self.stiffness_matrix = sparse.csr_matrix(
            (stiff_data, (rows, cols)),
            shape=(n, n)
        )

    def assemble_system(self) -> None:
        """Assemble system matrices with automatic parallel/sequential fallback."""
        if not self.try_parallel_assembly():
            self.sequential_assembly()
    
    @abstractmethod
    def _compute_rhs(self) -> np.ndarray:
        """Compute the right-hand side vector.
        
        To be implemented by derived classes.
        
        Returns:
            np.ndarray: The right-hand side vector
        """
        pass

    @abstractmethod
    def _solve_system(self) -> np.ndarray:
        """Solve the linear system.
        
        To be implemented by derived classes.
        
        Returns:
            np.ndarray: The solution vector
        """
        pass

    def solve_and_save(self, save_name: Union[str, Path]) -> None:
        """Solve the problem and save results.
        
        Args:
            save_name: Name/path for saving the results
        """
        self.assemble_system()
        self.solution = self._solve_system()
        self._save_solution(save_name)
    
    def _save_solution(self, save_name: Union[str, Path]) -> None:
        """Save the solution and associated data.
        
        Args:
            save_name: Name/path for saving the results
        """
        solution = Solution(
            data=self.solution,
            problem_type=str(self.config.problem_type),
            metadata={
                'mesh_size': self.config.mesh_size,
                'problem_params': self.config.to_dict()
            }
        )
        
        manager = FEMDataManager()
        manager.save(
            name=save_name,
            solution=solution,
            mesh=MeshData.from_mesh(self.config.mesh),
            matrices=FEMMatrices(
                mass_matrix=self.mass_matrix,
                stiffness_matrix=self.stiffness_matrix
            )
        )