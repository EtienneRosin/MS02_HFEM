import os
from mesh_manager import CustomTwoDimensionMesh
from mesh_manager.geometries import rectangular_mesh

def batch_create_meshes(
    mesh_configurations: list[dict], 
    mesh_geometry: callable,
    output_directory: str = 'meshes',
) -> list[str]:
    """
    Create multiple meshes with different configurations.

    Parameters
    ----------
    mesh_configurations : list[dict]
        List of mesh configuration dictionaries
    output_directory : str, optional
        Directory to save generated meshes

    Returns
    -------
    list[str]
        List of generated mesh file paths
    """
    os.makedirs(output_directory, exist_ok=True)
    
    mesh_paths = []
    for idx, config in enumerate(mesh_configurations, 1):
        save_name = config.get('save_name', f'mesh_{idx}')
        mesh_path = mesh_geometry(
            save_name=save_name,
            output_directory=output_directory,
            **{k: v for k, v in config.items() if k != 'save_name'}
        )
        mesh_paths.append(mesh_path)
    
    return mesh_paths

if __name__ == '__main__':
    
    batch_configurations = [
        {'save_name': 'fine_mesh', 'h': 0.05, 'L_x': 2, 'L_y': 1},
        {'save_name': 'coarse_mesh', 'h': 0.2, 'L_x': 2, 'L_y': 1},
        {'save_name': 'square_mesh', 'h': 0.1, 'L_x': 1, 'L_y': 1}
    ]
    print(f"{batch_configurations[0]['save_name'] = }")
    generated_meshes = batch_create_meshes(batch_configurations, rectangular_mesh)
    
    for mesh in batch_configurations:
        mesh_rectangle = CustomTwoDimensionMesh(filename=f"meshes/{mesh['save_name']}.msh")
        mesh_rectangle.display()
    