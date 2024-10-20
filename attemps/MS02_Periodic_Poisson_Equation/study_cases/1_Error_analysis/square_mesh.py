import meshio
import gmsh
import pygmsh
import numpy as np

def create_square_mesh(h = 0.1, L = 1, save_name = "./study_cases/1_Error_analysis/square.msh"):
    print(f"Creatig a square mesh of size {L:.1e} x {L:.1e} with h = {h:.1e}.")
    points_definition = [
        (0, 0, 0),  # 0
        (L, 0, 0),  # 1
        (L, L, 0),  # 2
        (0, L, 0)   # 3
    ]

    # Initialize empty geometry using the built-in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()

    # Add points for the full rectangular domain
    points = [
        model.add_point(point, mesh_size = h) for point in points_definition
    ]

    omega_segments = [
        model.add_line(points[0], points[1]),
        model.add_line(points[1], points[2]),
        model.add_line(points[2], points[3]),
        model.add_line(points[3], points[0]),
    ]

    # Create line loops and plane surfaces for both subdomains
    omega_loop = model.add_curve_loop(omega_segments)
    omega_surface = model.add_plane_surface(omega_loop)

    # Call gmsh kernel before adding physical entities
    model.synchronize()

    # Add physical groups for the subdomains and boundaries
    model.add_physical([omega_surface], r"\Omega")  # Left subdomain
    # model.add_physical([omega_2_surface], r"\Omega_2")  # Right subdomain

    # Boundaries
    model.add_physical([*omega_segments], r"\partial\Omega")

    # Generate the mesh and write it to a file
    geometry.generate_mesh(dim=2)
    gmsh.write(save_name)
    gmsh.clear()
    geometry.__exit__()
    print(f"Square mesh created.")
    
if __name__ == '__main__':
    create_square_mesh()