# hfem/mesh_manager/mesh_generator.py
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Union
from pathlib import Path

from hfem.mesh_manager.custom_2D_mesh import CustomTwoDimensionMesh
from hfem.mesh_manager.geometries.rectangular_mesh import rectangular_mesh  # et autres générateurs

@dataclass
class MeshGenerator:
    """Gère la création de maillages avec traçabilité des paramètres."""
    
    generator: Callable
    mesh_params: Dict[str, Any]
    output_dir: Union[str, Path]
    
    def __post_init__(self):
        """Convertit output_dir en Path et crée le dossier si nécessaire."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_mesh(self, name: str) -> CustomTwoDimensionMesh:
        """
        Crée un maillage et conserve ses paramètres de génération.
        
        Parameters
        ----------
        name : str
            Nom du fichier maillage
        
        Returns
        -------
        CustomTwoDimensionMesh
            Le maillage créé avec ses paramètres de génération
        """
        # Assure l'extension .msh
        if not name.endswith('.msh'):
            name += '.msh'
            
        # Génère le maillage
        mesh_file = self.generator(
            save_name=str(self.output_dir / name),
            **self.mesh_params
        )
        
        # Crée et configure le maillage
        mesh = CustomTwoDimensionMesh(mesh_file)
        # mesh.generation_params = self.mesh_params.copy()  # Copie pour éviter les références
        mesh.generation_params = {
            'type': self.generator.__name__,  # ou 'rectangular' si hardcodé
            **self.mesh_params
        }
        
        return mesh