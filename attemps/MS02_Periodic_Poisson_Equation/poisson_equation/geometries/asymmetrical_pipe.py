import meshio
import gmsh
import pygmsh


# Channel parameters
L = 2.2
resolution = 0.01
H = 1
h = 0.3


# Initialize empty geometry using the built-in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

# Add points for the full rectangular domain
points = [
    model.add_point((0, 0, 0), mesh_size=resolution),
    model.add_point((L, 0, 0), mesh_size=resolution),
    model.add_point((L, h, 0), mesh_size=resolution),
    model.add_point((L - h, h, 0), mesh_size=resolution),
    model.add_point((L - h, H, 0), mesh_size=resolution),
    model.add_point((0, H, 0), mesh_size=resolution),    
]

# Add lines for the left and right subdomains
domain_lines = [
    model.add_line(points[0], points[1]),   # bottom
    model.add_line(points[1], points[2]),   # right bottom
    model.add_line(points[2], points[3]),   # top right
    model.add_line(points[3], points[4]),   # right top
    model.add_line(points[4], points[5]),   # top left
    model.add_line(points[5], points[0]),   # left
      
]



# Create line loops and plane surfaces for both subdomains
loop = model.add_curve_loop(domain_lines)
surface = model.add_plane_surface(loop)

# Call gmsh kernel before adding physical entities
model.synchronize()

# Add physical groups for the subdomains and boundaries
model.add_physical([surface], r"\Omega")  
model.add_physical(domain_lines[:], r"\partial\Omega")  

# model.add_physical([left_subdomain_lines[0]], r"\Gamma_1")  # Bottom boundary of the left subdomain
# model.add_physical([left_subdomain_lines[1], right_subdomain_lines[1]], r"\Gamma_2")  # Right boundary of the domain
# model.add_physical([left_subdomain_lines[3], right_subdomain_lines[2]], r"\partial\Omega")  # Top boundary of both subdomains

# Generate the mesh and write it to a file
geometry.generate_mesh(dim=2)
gmsh.write("./geometries/asymmetrical_pipe.msh")
gmsh.clear()
geometry.__exit__()