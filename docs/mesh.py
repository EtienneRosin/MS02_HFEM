"""Read msh file."""


from dataclasses import dataclass
from typing import Any, Optional

from matplotlib.pyplot import Axes, colorbar, subplots
from meshio import read
from numpy.typing import NDArray


@dataclass(slots=True)
class Mesh:
    """Mesh class.

    Attributes
    ----------
    nb_nodes : int
        number of nodes
    node_coordinates: array of size (nb_nodes, 2)
        coordinates of the nodes
    nb_triangles: int
        number of triangles
    triangles: array of size (nb_triangles, 3)
        array of triangles
    nb_edges: int
        number of edges
    edges: array of size (nb_edges, 2)
        array of edges
    """

    nb_nodes: int
    node_coordinates: NDArray  # of size (nb_nodes, 2)
    # node_tags: NDArray
    #
    nb_triangles: int
    triangles: NDArray  # of size (nb_triangles, 3)
    # triangle_tags: NDArray
    #
    nb_edges: int
    edges: NDArray  # of size (nb_edges, 2)
    # edge_tags: NDArray


def read_msh(filename: str) -> Mesh:
    """Read a .msh file containing a 2d triangular mesh.

    Parameters
    ----------
    filename : str
        name of the file

    Returns
    -------
    Mesh
        Mesh object
    """

    mesh = read(filename)

    node_coordinates = mesh.points[:, :2]
    nb_nodes = node_coordinates.shape[0]
    # node_tags = mesh.cell_data["gmsh:physical"][type_idx["vertex"]]

    type_idx = {cell.type: i for i, cell in enumerate(mesh.cells)}

    triangles = mesh.cells[type_idx["triangle"]].data
    nb_triangles = triangles.shape[0]
    # triangle_tags = mesh.cell_data["gmsh:physical"][type_idx["triangle"]]

    edges = mesh.cells[type_idx["line"]].data
    nb_edges = edges.shape[0]
    # edge_tags = mesh.cell_data["gmsh:physical"][type_idx["line"]]

    # mesh.field_data  # dict[domain_name: [tag, dim]]
    # mesh.cell_data  # "gmsh:physical": list[NDArray[tag]]

    return Mesh(
        nb_nodes=nb_nodes,
        node_coordinates=node_coordinates,
        # node_tags=node_tags,
        nb_triangles=nb_triangles,
        triangles=triangles,
        # triangle_tags=triangle_tags,
        nb_edges=nb_edges,
        edges=edges,
        # edge_tags=edge_tags,
    )


def plot_mesh(
    mesh: Mesh,
    *,
    ax: Optional[Axes] = None,
) -> None:
    """Plot a 2d mesh.

    Parameters
    ----------
    mesh : Mesh
        Mesh object
    ax : Optional[Axes], optional
        matplotlib axes if not given is creted, by default None
    """

    if ax is None:
        _, ax = subplots(layout="constrained")

    ax.triplot(mesh.node_coordinates[:, 0], mesh.node_coordinates[:, 1], mesh.triangles)


def plot_function(
    vector: NDArray,
    mesh: Mesh,
    *,
    ax: Optional[Axes] = None,
    kwa_plot: dict[str, Any] = None,
) -> None:
    """Plot of the solution."""

    if ax is None:
        _, ax = subplots(layout="constrained", subplot_kw={"projection": "3d"})

    im = ax.plot_trisurf(
        mesh.node_coordinates[:, 0],
        mesh.node_coordinates[:, 1],
        mesh.triangles,
        vector,
        **kwa_plot,
    )
    colorbar(im, ax=ax)
