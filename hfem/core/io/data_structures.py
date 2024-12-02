from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import scipy.sparse as sparse
import cmasher as cmr
from hfem.viz.conditional_style_context import conditional_style_context

@dataclass
class MeshData:
    """Structure for mesh data storage."""
    nodes: np.ndarray
    elements: np.ndarray
    h: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'h': self.h,
            'elements': self.elements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeshData':
        return cls(**data)
    
    @classmethod
    def from_mesh(cls, mesh) -> 'MeshData':
        """Create from a CustomTwoDimensionMesh instance."""
        return cls(
            nodes=mesh.node_coords,
            elements=mesh.tri_nodes,
            h=mesh.h,
            # domain_size=mesh.domain_size
        )
    
    # @conditional_style_context()
    # def display_field(self, 
    #              field: np.ndarray,
    #              field_label: str,
    #              ax: Optional[plt.Axes] = None,
    #              title: Optional[str] = None,
    #              show: bool = True,
    #              kind = 'tricontour',
    #              cmap = cmr.wildfire,
    #              num_levels: int = 30,
    #              show_cbar: bool = True,
    #              cbar_props: Optional[Dict] = {},
    #              save_name: Optional[str] = None, 
    #              **kwargs) -> plt.Axes:
    #     """Affiche un champ scalaire sur le maillage."""
    #     if ax is None:
    #         # fig, ax = plt.subplots()
    #         fig = plt.figure()
        
    #     # Créer la triangulation
    #     triangulation = Triangulation(*self.nodes.T, self.elements)
    #     vmin, vmax = field.min(), field.max()
    #     bound = max(np.abs(vmin), np.abs(vmax))
    #     levels = np.linspace(-bound, bound, num_levels)

    #     if kind == 'tricontour':
    #         if ax is None:
    #             ax = fig.add_subplot()
    #             ax.set(xlabel = r'$x$', ylabel = r'$y$')
    #         im = ax.tricontourf(
    #             triangulation, 
    #             field,
    #             levels=levels,
    #             vmax = bound,
    #             vmin = -bound,
    #             cmap = cmap,
    #             **kwargs
    #         )
            
    #         # Lignes de contour
    #         ax.tricontour(
    #             triangulation, 
    #             field,
    #             colors='k',
    #             alpha=0.2,
    #             levels=levels,
    #             vmax = bound,
    #             vmin = -bound,
    #             linewidths=0.25,
    #         )
    #         if show_cbar:
    #             plt.colorbar(im, ax=ax, label = field_label, **cbar_props)
    #         ax.set_aspect('equal')
    #     elif kind == 'trisurface':
    #         if ax is None:
    #             ax = fig.add_subplot(projection = '3d')
    #             ax.set(xlabel = r'$x$', ylabel = r'$y$', zlabel = field_label)
    #         im = ax.plot_trisurf(
    #             triangulation,
    #             field,
    #             cmap=cmap,
    #             # alpha=config.alpha,
    #             vmin=-bound,
    #             vmax=bound
    #         )
    #         ax.grid(True, alpha=0.3)
    #         for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    #             pane.fill = False
    #         ax.set_aspect('equalxy')
        
    #     # ax.view_init(config.view_elevation, config.view_azimuth)
        
        
    #     # vmin = config.vmin if config.vmin is not None else np.min(field)
    #     # vmax = config.vmax if config.vmax is not None else np.max(field)
    #     # return np.linspace(vmin, vmax, config.num_levels)
    #     if title:
    #         ax.set_title(title)
        
    #     plt.tight_layout()
    #     if save_name:
    #         plt.savefig(f"{save_name}.pdf")
    #     plt.show()
        
    #     if show:
    #         plt.show()
        
    #     return ax
    
    
    @conditional_style_context()
    def display_correctors(self,
                        correctors: Dict[str, np.ndarray],
                        titles: Optional[Dict[str, str]] = None,
                        show: bool = True,
                        field_label: str = r'$\omega^\eta(\boldsymbol{y})$',
                        figsize: Tuple[float, float] = (5, 3.75),
                        cmap=cmr.lavender,
                        num_levels: int = 30,
                        cbar_props: Optional[Dict] = None,
                        save_name: Optional[str] = None,
                        **kwargs) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Display two corrector fields side by side with a single colorbar.
        
        Parameters
        ----------
        correctors : Dict[str, np.ndarray]
            Dictionary containing the corrector data with keys 'corrector_x' and 'corrector_y'
        titles : Optional[Dict[str, str]], optional
            Dictionary of titles for each corrector plot, by default None
            Example: {'corrector_x': r'$\omega_1^\eta(\boldsymbol{y})$', 
                    'corrector_y': r'$\omega_2^\eta(\boldsymbol{y})$'}
        show : bool, optional
            Whether to display the plot, by default True
        field_label : str, optional
            Label for the colorbar, by default '$\omega^\eta(\boldsymbol{y})$'
        figsize : Tuple[float, float], optional
            Figure size (width, height), by default (5, 3.75)
        cmap : matplotlib.colors.Colormap, optional
            Colormap to use, by default cmr.lavender
        num_levels : int, optional
            Number of contour levels, by default 30
        cbar_props : Optional[Dict], optional
            Properties to pass to the colorbar, by default None
        save_name : Optional[str], optional
            If provided, save the figure to this path, by default None
        **kwargs
            Additional arguments passed to tricontourf
            
        Returns
        -------
        Tuple[plt.Figure, Dict[str, plt.Axes]]
            Figure and dictionary of axes objects
        """
        # Create figure and axes
        fig = plt.figure(figsize=figsize, layout = 'constrained')
        axes = fig.subplot_mosaic(
            mosaic=[["corrector_x", "corrector_y"]], 
            gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.0},
            sharex=True,
            sharey=True
        )
        
        # Compute global bounds and levels
        all_values = np.concatenate(list(correctors.values()))
        bound = max(abs(all_values.min()), abs(all_values.max()))
        levels = np.linspace(-bound, bound, num_levels)
        
        # Create triangulation once
        triangulation = Triangulation(*self.nodes.T, self.elements)
        
        # Set default titles if not provided
        if titles is None:
            titles = {
                'corrector_x': r'$\omega_1^\eta(\boldsymbol{y})$',
                'corrector_y': r'$\omega_2^\eta(\boldsymbol{y})$'
            }
        
        # Plot each corrector
        for idx, (key, field) in enumerate(correctors.items()):
            # Main contour plot
            im = axes[key].tricontourf(
                triangulation,
                field,
                levels=levels,
                vmax=bound,
                vmin=-bound,
                cmap=cmap,
                **kwargs
            )
            
            # Set title and labels
            axes[key].set_title(titles[key])
            axes[key].set_xlabel(r'$x$')
            axes[key].tick_params(axis='x', rotation=45)
            if idx == 0:
                axes[key].set_ylabel(r'$y$')
            
            # Add contour lines
            axes[key].tricontour(
                triangulation,
                field,
                colors='k',
                alpha=0.2,
                levels=levels,
                vmax=bound,
                vmin=-bound,
                linewidths=0.25,
            )
            
            axes[key].set_aspect('equal')
        
        # Add colorbars with equal spacing
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.ticker import ScalarFormatter

        # Create invisible left spacer for symmetry
        divider_left = make_axes_locatable(axes['corrector_x'])
        cax_left = divider_left.append_axes("left", size="5%", pad=0.1)
        cax_left.set_visible(False)

        # Create colorbar on right
        divider_right = make_axes_locatable(axes['corrector_y'])
        cax_right = divider_right.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax_right, label=field_label)
        
        # Set scientific notation for colorbar
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
        # Apply any additional colorbar properties
        if cbar_props:
            for key, value in cbar_props.items():
                setattr(cbar, key, value)
        
        # plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{save_name}.pdf", bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig, axes



    @conditional_style_context()
    def display_field(self, 
                    field: np.ndarray,
                    field_label: str,
                    ax: Optional[plt.Axes] = None,
                    title: Optional[str] = None,
                    show: bool = True,
                    kind: str = 'tricontour',
                    cmap = cmr.lavender,
                    num_levels: int = 30,
                    show_cbar: bool = True,
                    cbar_props: Optional[Dict] = None,
                    save_name: Optional[str] = None, 
                    **kwargs) -> plt.Axes:
        """Display a scalar field on the mesh.
        
        Args:
            field: Array of field values
            field_label: Label for the field (used in colorbar or z-axis)
            ax: Optional existing axes to plot on
            title: Optional title for the plot
            show: Whether to show the plot
            kind: Type of plot ('tricontour' or 'trisurface')
            cmap: Colormap to use
            num_levels: Number of contour levels
            show_cbar: Whether to show the colorbar
            cbar_props: Additional properties for colorbar
            save_name: If provided, save the figure to this path
            **kwargs: Additional arguments passed to plotting function
            
        Returns:
            The axis object containing the plot
        """
        if ax is None:
            fig = plt.figure()
        
        # Create triangulation
        triangulation = Triangulation(*self.nodes.T, self.elements)
        vmin, vmax = field.min(), field.max()
        bound = max(np.abs(vmin), np.abs(vmax))
        # print(f"{bound = }")
        levels = np.linspace(-bound, bound, num_levels)

        if kind == 'tricontour':
            if ax is None:
                ax = fig.add_subplot()
                ax.set(xlabel=r'$x$', ylabel=r'$y$')
            im = ax.tricontourf(
                triangulation, 
                field,
                levels=levels,
                vmax=bound,
                vmin=-bound,
                cmap=cmap,
                **kwargs
            )
            
            # Contour lines
            ax.tricontour(
                triangulation, 
                field,
                colors='k',
                alpha=0.2,
                levels=levels,
                linewidths=0.25,
            )
            if show_cbar:
                cbar = plt.colorbar(im, ax=ax, label=field_label)
                if cbar_props:
                    for key, value in cbar_props.items():
                        setattr(cbar, key, value)
            ax.set_aspect('equal')
            
        elif kind == 'trisurface':
            if ax is None:
                ax = fig.add_subplot(projection='3d')
                ax.set(xlabel=r'$x$', ylabel=r'$y$', zlabel=field_label)
            im = ax.plot_trisurf(
                triangulation,
                field,
                cmap=cmap,
                vmin=-bound,
                vmax=bound,
                **kwargs
            )
            ax.grid(True, alpha=0.3)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
            ax.set_aspect('equalxy')
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(f"{save_name}.pdf", bbox_inches='tight')
        
        if show:
            plt.show()
        
        return ax








@dataclass
class FEMMatrices:
    """Structure for FEM matrices storage with proper sparse matrix handling."""
    mass_matrix: sparse.spmatrix
    stiffness_matrix: sparse.spmatrix
    gradient_matrices: Optional[List[sparse.spmatrix]] = None
    
    def __post_init__(self):
        """Ensure all matrices are in CSR format."""
        self.mass_matrix = self.mass_matrix.tocsr()
        self.stiffness_matrix = self.stiffness_matrix.tocsr()
        if self.gradient_matrices:
            self.gradient_matrices = [m.tocsr() for m in self.gradient_matrices]    
if __name__ == '__main__':
    mesh_file = "meshes/test_mesh.msh"
    from hfem.mesh_manager.geometries import rectangular_mesh
    from hfem.mesh_manager import CustomTwoDimensionMesh
    from hfem.core.related_matrices import assemble_P_0
    rectangular_mesh(h=0.1, L_x=1, L_y=1, save_name=mesh_file)
    mesh = CustomTwoDimensionMesh(mesh_file)
    # mesh.display()
    print(mesh.h)
    
    P_0 = assemble_P_0(mesh)
    from pprint import pprint
    # pprint(P_0.toarray())
    data = FEMMatrices(mass_matrix=P_0, stiffness_matrix=P_0)
    
    d = data.to_dict()
    
    new_data = FEMMatrices.from_dict(d)
    print(new_data.mass_matrix.toarray())
    # print(data.to_dict())