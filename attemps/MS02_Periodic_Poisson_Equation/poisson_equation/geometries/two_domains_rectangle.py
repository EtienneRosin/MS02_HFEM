import meshio
import gmsh
import pygmsh

h = 0.08

# Channel parameters
L = 9
H = 2

points_definition = [
    (0, 0, 0),      # 0
    (L/2, 0, 0),    # 1
    (L, 0, 0),      # 2
    (L, H, 0),      # 3
    (L/2, H, 0),    # 4
    (0, H, 0)       # 5
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
omega_1_lines = [
    model.add_line(points[0], points[1]),  # Bottom left
    model.add_line(points[1], points[4]),  # Vertical middle
    model.add_line(points[4], points[5]),  # Top left
    model.add_line(points[5], points[0]),  # Left side
]

omega_2_lines = [
    model.add_line(points[1], points[2]),  # Bottom right
    model.add_line(points[2], points[3]),  # Right side
    model.add_line(points[3], points[4]),  # Top right
    model.add_line(points[4], points[1]),  # Vertical middle
]

# Create line loops and plane surfaces for both subdomains
omega_1_loop = model.add_curve_loop(omega_1_lines)
omega_1_surface = model.add_plane_surface(omega_1_loop)

omega_2_loop = model.add_curve_loop(omega_2_lines)
omega_2_surface = model.add_plane_surface(omega_2_loop)

# Call gmsh kernel before adding physical entities
model.synchronize()

# Add physical groups for the subdomains and boundaries
model.add_physical([omega_1_surface], r"\Omega_1")  # Left subdomain
model.add_physical([omega_2_surface], r"\Omega_2")  # Right subdomain

# Boundaries
# model.add_physical([left_subdomain_lines[1],], r"\Omega_1\cap\Omega_2")
model.add_physical([
    omega_1_lines[0], 
    omega_2_lines[0], 
    omega_2_lines[1], 
    omega_2_lines[2], 
    omega_1_lines[2],
    omega_1_lines[-1]], r"\partial\Omega")



# model.add_physical([left_subdomain_lines[0]], r"\Gamma_1")  # Bottom boundary of the left subdomain
# model.add_physical([left_subdomain_lines[1], right_subdomain_lines[1]], r"\Gamma_2")  # Right boundary of the domain
# model.add_physical([left_subdomain_lines[3], right_subdomain_lines[2]], r"\partial\Omega")  # Top boundary of both subdomains

# Generate the mesh and write it to a file
geometry.generate_mesh(dim=2)
gmsh.write("./geometries/two_domains_rectangle.msh")
gmsh.clear()
geometry.__exit__()