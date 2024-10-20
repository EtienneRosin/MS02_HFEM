import meshio
import gmsh
import pygmsh
import numpy as np

h = 0.035

# Channel parameters
L = 1
# H = 2

points_definition = [
    (0, 0, 0),      # 0
    (L, 0, 0),    # 1
    (L, L, 0),      # 2
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

# Add lines for the left and right subdomains
# omega_1_lines = [
#     model.add_line(points[0], points[1]),  # Bottom left
#     model.add_line(points[1], points[4]),  # Vertical middle
#     model.add_line(points[4], points[5]),  # Top left
#     model.add_line(points[5], points[0]),  # Left side
# ]

# omega_2_lines = [
#     model.add_line(points[1], points[2]),  # Bottom right
#     model.add_line(points[2], points[3]),  # Right side
#     model.add_line(points[3], points[4]),  # Top right
#     model.add_line(points[4], points[1]),  # Vertical middle
# ]

omega_segments = [
    model.add_line(points[0], points[1]),
    model.add_line(points[1], points[2]),
    model.add_line(points[2], points[3]),
    model.add_line(points[3], points[0]),
]

# Create line loops and plane surfaces for both subdomains
omega_loop = model.add_curve_loop(omega_segments)
omega_surface = model.add_plane_surface(omega_loop)

# omega_2_loop = model.add_curve_loop(omega_2_lines)
# omega_2_surface = model.add_plane_surface(omega_2_loop)

# Call gmsh kernel before adding physical entities
model.synchronize()

# Add physical groups for the subdomains and boundaries
model.add_physical([omega_surface], r"\Omega")  # Left subdomain
# model.add_physical([omega_2_surface], r"\Omega_2")  # Right subdomain

# Boundaries
# model.add_physical([left_subdomain_lines[1],], r"\Omega_1\cap\Omega_2")
model.add_physical([*omega_segments], r"\partial\Omega")



# model.add_physical([left_subdomain_lines[0]], r"\Gamma_1")  # Bottom boundary of the left subdomain
# model.add_physical([left_subdomain_lines[1], right_subdomain_lines[1]], r"\Gamma_2")  # Right boundary of the domain
# model.add_physical([left_subdomain_lines[3], right_subdomain_lines[2]], r"\partial\Omega")  # Top boundary of both subdomains

# Generate the mesh and write it to a file
geometry.generate_mesh(dim=2)
gmsh.write("./poisson_equation/geometries/square.msh")
gmsh.clear()
geometry.__exit__()