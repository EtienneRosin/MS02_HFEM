"""Create a mesh of a square."""


from sys import argv
from typing import Optional

import gmsh

Factory = gmsh.model.geo


def mesh(h: float) -> None:
    """Mesh."""

    if h <= 0:
        raise ValueError(f"h={h} must be positive.")

    # définition des points (en 3D, raison pour laquelle il y a un 0 en z)
    points = [
        Factory.addPoint(0, 0, 0, h),
        Factory.addPoint(0, 2, 0, h),
        Factory.addPoint(2, 2, 0, h),
        Factory.addPoint(2, 0, 0, h),
    ]

    # définition des segments qui relient les points
    lines = [
        Factory.addLine(points[0], points[1]),
        Factory.addLine(points[1], points[2]),
        Factory.addLine(points[2], points[3]),
        Factory.addLine(points[3], points[0]),
    ]

    # définition des contours fermés
    loops = [Factory.addCurveLoop(lines)]

    # définition des surfaces à partir contours fermés
    surfaces = [Factory.addPlaneSurface(loops)]

    Factory.synchronize()

    # définition des éléments physiques : pour ces éléments, nous pourrons récupérer
    # les références
    gmsh.model.addPhysicalGroup(dim=0, tags=points, name="boundary_points")
    gmsh.model.addPhysicalGroup(dim=1, tags=lines, name="boundary")
    gmsh.model.addPhysicalGroup(dim=2, tags=surfaces, name="surface")


def square_mesh(
    h: float,
    *,
    filename: Optional[str] = None,
    gui: bool = False,
    terminal_output: bool = True,
) -> None:
    """Main."""

    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", int(terminal_output))
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.add("mesh")

    mesh(h)

    gmsh.model.mesh.generate(2)

    if gui:
        gmsh.fltk.run()

    if filename is not None:
        gmsh.write(f"{filename}.msh")

    gmsh.finalize()


if __name__ == "__main__":
    square_mesh(0.1)
