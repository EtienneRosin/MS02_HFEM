# """
# Enhanced visualization module for finite element solutions and errors.
# Focuses on two main types of plots:
# - Contour fills (2D)
# - Triangular surfaces (3D)
# """

# from dataclasses import dataclass, replace, field
# from typing import Literal, Optional, Tuple, Union, Any
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.figure import Figure
# from matplotlib.axes import Axes
# from matplotlib.tri import Triangulation
# from enum import Enum

# from hfem.viz.custom_components import CustomFigure
# from hfem.viz.conditional_style_context import conditional_style_context, conditional_style_context_with_visualization_config

# class ErrorType(Enum):
#     """Types of errors that can be visualized."""
#     ABSOLUTE = 'absolute'
#     RELATIVE = 'relative'
#     LOG = 'log'


# @dataclass(frozen=True)
# class VisualizationConfig:
#     """Configuration for FEM solution visualization."""
    
#     # Plot type
#     kind: str = 'contourf'
    
#     # Visual parameters
#     cmap: str = 'viridis'
#     title: Optional[str] = None
#     vmin: Optional[float] = None
#     vmax: Optional[float] = None
#     alpha: float = 1.0
#     colorbar_label: str = r'$u_h$'
    
#     # Figure settings
#     figsize: Tuple[int, int] = None
#     dpi: int = 100
    
#     # Custom Colobar settings
#     cbar: bool = False
#     cbar_props: dict = field(default_factory=dict)
    
#     # 3D view settings (for trisurface)
#     view_elevation: float = 30
#     view_azimuth: float = -60
    
#     # Contour settings
#     # num_levels: int = 50
#     num_levels: int = 30
    
#     # Error specific settings
#     error_type: ErrorType = ErrorType.ABSOLUTE
#     symmetric_scale: bool = True
    
#     # save settings
#     save_name: Optional[str] = None
#     corrector_labels: Tuple[str, str] = (r'$\omega_1$', r'$\omega_2$')
#     corrector_error_labels: Tuple[str, str] = (r'$|\omega_1 - \omega_1^{ex}|$', r'$|\omega_2 - \omega_2^{ex}|$')


# def solution_config(
#     kind: str = 'contourf',
#     cmap: str = 'viridis',
#     figsize: Tuple[int, int] = (12, 5),
#     **kwargs
# ) -> VisualizationConfig:
#     """Create configuration for solution visualization.
    
#     Parameters
#     ----------
#     kind : str
#         Type of plot ('contourf' or 'trisurface')
#     cmap : str
#         Colormap name
#     figsize : Tuple[int, int]
#         Figure size in inches
#     **kwargs
#         Additional arguments passed to VisualizationConfig
        
#     Returns
#     -------
#     VisualizationConfig
#         Configuration for solution visualization
#     """
#     defaults = {
#         'kind': kind,
#         'cmap': cmap,
#         'figsize': figsize,
#         'cbar': True,
#         'cbar_props': {
#             'pad': 0.05,
#             'fraction': 0.046
#         },
#         'num_levels': 30,
#         'symmetric_scale': False
#     }
#     defaults.update(kwargs)
#     return VisualizationConfig(**defaults)

# def error_config(
#     kind: str = 'contourf',
#     error_type: ErrorType = ErrorType.ABSOLUTE,
#     cmap: str = 'cmr.lavender',
#     figsize: Tuple[int, int] = (12, 5),
#     **kwargs
# ) -> VisualizationConfig:
#     """Create configuration for error visualization.
    
#     Parameters
#     ----------
#     kind : str
#         Type of plot ('contourf' or 'trisurface')
#     error_type : ErrorType
#         Type of error to visualize
#     cmap : str
#         Colormap name
#     figsize : Tuple[int, int]
#         Figure size in inches
#     **kwargs
#         Additional arguments passed to VisualizationConfig
        
#     Returns
#     -------
#     VisualizationConfig
#         Configuration for error visualization
#     """
#     defaults = {
#         'kind': kind,
#         'error_type': error_type,
#         'cmap': cmap,
#         'figsize': figsize,
#         'cbar': True,
#         'cbar_props': {
#             'pad': 0.05,
#             'fraction': 0.046
#         },
#         'num_levels': 30,
#         'symmetric_scale': True,
#         'vmin': None if error_type == ErrorType.LOG else 0,
#         'colorbar_label': {
#             ErrorType.ABSOLUTE: r'$|u_h - u|$',
#             ErrorType.RELATIVE: r'$\frac{|u_h - u|}{|u|}$',
#             ErrorType.LOG: r'$\log\left(\frac{|u_h - u|}{|u|}\right)$'
#         }[error_type]
#     }
#     defaults.update(kwargs)
#     return VisualizationConfig(**defaults)


# class FEMVisualizer:
#     """Handles visualization of finite element solutions and errors."""
    
#     def __init__(self, node_coords: np.ndarray, triangles: np.ndarray):
#         """
#         Initialize visualizer with mesh data.
        
#         Parameters
#         ----------
#         node_coords : np.ndarray
#             Nodal coordinates (N x 2)
#         triangles : np.ndarray
#             Triangle connectivity (M x 3)
#         """
#         self.node_coords = node_coords
#         self.triangles = triangles
#         self.triangulation = Triangulation(
#             node_coords[:, 0], 
#             node_coords[:, 1], 
#             triangles
#         )

#     def create_figure(self, config: VisualizationConfig) -> Tuple[Figure, Axes]:
#         """Create figure and axes based on visualization type."""
#         fig = plt.figure(
#             figsize=config.figsize, 
#             dpi=config.dpi, 
#             FigureClass=CustomFigure)
#         # print(f"figure créée et c'est une custom figure : {isinstance(fig, CustomFigure)}")
        
#         if config.kind == 'trisurface':
#             ax = fig.add_subplot(111, projection='3d')
#             ax.set(aspect = 'equalxy', xlabel = r"$x$", ylabel = r"$y$")
#             ax.view_init(config.view_elevation, config.view_azimuth)
#         else:
#             ax = fig.add_subplot(111)
#             ax.set(aspect = 'equal', xlabel = r"$x$", ylabel = r"$y$")        
#         if config.title:
#             ax.set_title(config.title)
            
#         return fig, ax
    
#     def process_error_field(self, error_field: np.ndarray, 
#                           config: VisualizationConfig) -> np.ndarray:
#         """Process error field based on error type."""
#         if config.error_type == ErrorType.RELATIVE:
#             denominator = np.max(np.abs(error_field))
#             if denominator == 0:
#                 return error_field
#             return error_field / denominator
#         elif config.error_type == ErrorType.LOG:
#             min_positive = np.min(np.abs(error_field[error_field != 0]))
#             field = np.copy(error_field)
#             field[field == 0] = min_positive
#             return np.log(field)
#         else:  # ABSOLUTE
#             return error_field

#     def get_error_limits(self, error_field: np.ndarray, 
#                         config: VisualizationConfig) -> Tuple[float, float]:
#         """Compute appropriate limits for error visualization."""
        
#         if config.symmetric_scale and (config.vmin is not None or config.vmax is not None):
#             max_abs = np.max(np.abs(error_field))
#             return -max_abs, max_abs
        
#         if config.vmin is not None:
#             if config.vmax is not None:
#                 return config.vmin, config.vmax
#             else:
#                 return config.vmin, np.max(error_field)
#         if config.vmax is not None:
#             if config.vmin is not None:
#                 return config.vmin, config.vmax
#             else:
#                 return np.min(error_field), config.vmax
        
#         return np.min(error_field), np.max(error_field)

#     def add_colorbar(self, config: VisualizationConfig) -> Tuple[bool, dict]:
#         """
#         Determine if a colorbar should be added and with which properties.
        
#         Parameters
#         ----------
#         config : VisualizationConfig
#             Visualization configuration
            
#         Returns
#         -------
#         Tuple[bool, dict]
#             - Boolean indicating if colorbar should be added
#             - Dictionary of colorbar properties to use
#         """
#         # Si cbar est explicitement False, pas de colorbar quoi qu'il arrive
#         if config.cbar == False:
#             return False, {}
        
#         # Propriétés par défaut pour la colorbar
#         default_props = {
#             'label': config.colorbar_label,
#             'pad': '2.5%',
#             'size': '2%'
#         }
        
#         # Si des propriétés spécifiques sont fournies, les utiliser
#         if config.cbar_props:
#             default_props.update(config.cbar_props)
        
#         return True, default_props

#     def plot_solution(self, 
#                       solution: np.ndarray,
#                       config: VisualizationConfig) -> Tuple[Figure, Axes]:
#         """Plot numerical solution."""
#         return self.plot_field(solution, config, is_error=False)
    
#     def plot_error(self,
#                   numerical_sol: np.ndarray,
#                   exact_sol: np.ndarray,
#                   config: VisualizationConfig) -> Tuple[Figure, Axes]:
#         """Plot error between numerical and exact solutions."""
#         # Calcul de l'erreur
#         raw_error = exact_sol - numerical_sol
        
#         # Traitement selon le type d'erreur
#         if config.error_type == ErrorType.RELATIVE:
#             # Éviter la division par zéro
#             denominator = np.maximum(np.abs(exact_sol), np.finfo(float).eps)
#             error = np.abs(raw_error/denominator)
#         elif config.error_type == ErrorType.LOG:
#             # Éviter log(0) et valeurs négatives
#             denominator = np.maximum(np.abs(exact_sol), np.finfo(float).eps)
#             rel_error = np.abs(raw_error)/denominator
#             error = np.log(np.maximum(rel_error, np.finfo(float).eps))
#         else:  # ABSOLUTE
#             error = np.abs(raw_error)
            
#         # Mise à jour des limites si nécessaire
#         if config.vmin is None or config.vmax is None:
#             vmin, vmax = self.get_error_limits(error, config)
#             config = replace(config, vmin=vmin, vmax=vmax)
            
#         return self.plot_field(error, config, is_error=True)

#     # @conditional_style_context()
#     @conditional_style_context_with_visualization_config()
#     def plot_field(self, field: np.ndarray, config: VisualizationConfig,
#                   is_error: bool = False, ax: plt.Axes = None) -> Tuple[Figure, Axes]:
#         """Plot finite element field."""
#         if ax is None:        
#             fig, ax = self.create_figure(config)
        
#         if config.kind == 'contourf':
#             fig, ax = self._plot_contourf(field, fig, ax, config)
#         else:
#             fig, ax = self._plot_trisurface(field, fig, ax, config)
        
#         if isinstance(fig, CustomFigure):
#             # print(f"c'est une custom figure")
#             fig.adjust_layout()
#         else:
#             fig.tight_layout()
        
#         if config.save_name:
#             plt.savefig(config.save_name, bbox_inches='tight', dpi=config.dpi)

#         # if isinstance(fig, CustomFigure):
#         #     fig.adjust_layout()

#         return fig, ax    
    
#     def _plot_contourf(self, field: np.ndarray, fig: Figure, ax: Axes, 
#                       config: VisualizationConfig) -> Tuple[Figure, Axes]:
#         """Create filled contour plot."""
#         levels = np.linspace(
#             config.vmin if config.vmin is not None else np.min(field),
#             config.vmax if config.vmax is not None else np.max(field),
#             config.num_levels
#         )
        
#         tcf = ax.tricontourf(
#             self.triangulation, 
#             field,
#             levels=levels,
#             cmap=config.cmap,
#             alpha=config.alpha,
#             rasterized = True
#         )
        
#         # Add thin contour lines for better visualization
#         ax.tricontour(
#             self.triangulation, 
#             field,
#             levels=levels,
#             colors='k',
#             alpha=0.2,
#             linewidths=0.25
#         )
        
#         show_colorbar, colorbar_props = self.add_colorbar(config)
#         if show_colorbar:
#             fig.custom_colorbar(tcf, ax=ax, **colorbar_props)
#         return fig, ax

#     def _plot_trisurface(self, field: np.ndarray, fig: Figure, ax: Axes,
#                         config: VisualizationConfig) -> Tuple[Figure, Axes]:
#         """Create 3D surface plot."""
#         surf = ax.plot_trisurf(
#             self.triangulation,
#             field,
#             cmap=config.cmap,
#             alpha=config.alpha,
#             vmin=config.vmin,
#             vmax=config.vmax
#         )
        
#         show_colorbar, colorbar_props = self.add_colorbar(config)
#         if show_colorbar:
#             fig.custom_colorbar(surf, ax=ax, **colorbar_props)
#         else:
#             ax.set(zlabel=config.colorbar_label)
        
#         # Improve 3D visualization
#         ax.grid(True, alpha=0.3)
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False
        
#         return fig, ax
    
# #     @conditional_style_context_with_visualization_config()
# #     def plot_paired_fields(self, 
# #                           fields: list[np.ndarray], 
# #                           config: VisualizationConfig,
# #                           titles: list[str] = None,
# #                           is_error: bool = False) -> Tuple[Figure, list[Axes]]:
# #         """
# #         Plot two fields side by side sharing a colorbar.
        
# #         Parameters
# #         ----------
# #         fields : list[np.ndarray]
# #             List of two fields to plot
# #         config : VisualizationConfig
# #             Visualization configuration
# #         titles : list[str], optional
# #             List of titles for the subplots
# #         is_error : bool
# #             Whether the fields represent errors
            
# #         Returns
# #         -------
# #         Tuple[Figure, list[Axes]]
# #             Figure and list of axes objects
# #         """
# #         if len(fields) != 2:
# #             raise ValueError("Expected exactly two fields to plot")
            
# #         # Create figure with two subplots side by side
# #         # fig = plt.figure(figsize=config.figsize or (12, 5))
# #         fig = CustomFigure(figsize=config.figsize or (12, 5))
# #         axes = []
        
# #         # Compute shared color limits if not specified
# #         if config.vmin is None or config.vmax is None:
# #             if is_error and config.symmetric_scale:
# #                 max_abs = max(np.max(np.abs(field)) for field in fields)
# #                 vmin, vmax = -max_abs, max_abs
# #             else:
# #                 vmin = min(np.min(field) for field in fields)
# #                 vmax = max(np.max(field) for field in fields)
# #             config = replace(config, vmin=vmin, vmax=vmax)
        
# #         # Plot each field
# #         main_mappable = None
# #         for i, field in enumerate(fields):
# #             ax = fig.add_subplot(1, 2, i+1)
# #             if titles and i < len(titles):
# #                 ax.set_title(titles[i])
                
# #             # Create subplot config without colorbar (we'll add a shared one)
# #             subplot_config = replace(config, 
# #                                   figsize=None,
# #                                   cbar=False,
# #                                   save_name=None)
            
# #             if config.kind == 'contourf':
# #                 _, ax, mappable = self._plot_contourf_with_mappable(field, fig, ax, subplot_config)
# #             else:
# #                 _, ax, mappable = self._plot_trisurface_with_mappable(field, fig, ax, subplot_config)
            
# #             if main_mappable is None:
# #                 main_mappable = mappable
                
# #             axes.append(ax)
            
# #         # Add shared colorbar
# #         show_colorbar, colorbar_props = self.add_colorbar(config)
# #         if show_colorbar:
# #             fig.custom_colorbar(mappable, ax=ax, **colorbar_props)
# #         # if config.cbar:
# #         #     show_colorbar, colorbar_props = self.add_colorbar(config)
# #         #     if fig and isinstance(fig, CustomFigure):
# #         #         fig.custom_colorbar(main_mappable, ax=axes, **colorbar_props)
# #         #     # elif show_colorbar:
# #         #     #     plt.colorbar(main_mappable, ax=axes, **colorbar_props)
                
# #         plt.tight_layout()
        
# #         if config.save_name:
# #             plt.savefig(config.save_name, bbox_inches='tight', dpi=config.dpi)
            
# #         return fig, axes
        
# #     def plot_correctors(self, 
# #                        correctors: list[np.ndarray],
# #                        config: Optional[VisualizationConfig] = None,
# #                        **kwargs) -> Tuple[Figure, list[Axes]]:
# #         """Plot two correctors side by side with shared colorbar."""
# #         if config is None:
# #             config = solution_config(**kwargs)
            
# #         return self.plot_paired_fields(
# #             correctors,
# #             config,
# #             titles=[r'$\omega_1$', r'$\omega_2$']
# #         )
        
# #     def plot_corrector_errors(self,
# #                             computed_correctors: list[np.ndarray],
# #                             exact_correctors: list[np.ndarray],
# #                             config: Optional[VisualizationConfig] = None,
# #                             **kwargs) -> Tuple[Figure, list[Axes]]:
# #         """Plot errors for both correctors side by side with shared colorbar."""
# #         if config is None:
# #             config = error_config(**kwargs)
            
# #         # Compute errors
# #         errors = [exact - computed for exact, computed in zip(exact_correctors, computed_correctors)]
        
# #         return self.plot_paired_fields(
# #             errors,
# #             config,
# #             titles=[r'$\omega_1$', r'$\omega_2$'],
# #             is_error=True
# #         )
        
#     def _plot_contourf_with_mappable(self, field: np.ndarray, fig: Figure, ax: Axes, 
#                                     config: VisualizationConfig) -> Tuple[Figure, Axes, Any]:
#         """Modified contourf plot that returns the mappable for colorbar creation."""
#         levels = np.linspace(
#             config.vmin if config.vmin is not None else np.min(field),
#             config.vmax if config.vmax is not None else np.max(field),
#             config.num_levels
#         )
        
#         tcf = ax.tricontourf(
#             self.triangulation, 
#             field,
#             levels=levels,
#             cmap=config.cmap,
#             alpha=config.alpha
#         )
        
#         ax.tricontour(
#             self.triangulation, 
#             field,
#             levels=levels,
#             colors='k',
#             alpha=0.2,
#             linewidths=0.25
#         )
        
#         return fig, ax, tcf
        
#     def _plot_trisurface_with_mappable(self, field: np.ndarray, fig: Figure, ax: Axes,
#                                       config: VisualizationConfig) -> Tuple[Figure, Axes, Any]:
#         """Modified trisurface plot that returns the mappable for colorbar creation."""
#         surf = ax.plot_trisurf(
#             self.triangulation,
#             field,
#             cmap=config.cmap,
#             alpha=config.alpha,
#             vmin=config.vmin,
#             vmax=config.vmax
#         )
        
#         # Improve 3D visualization
#         ax.grid(True, alpha=0.3)
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False
        
#         return fig, ax, surf

# # # Configuration functions
# # def solution_config(kind: str = 'contourf', **kwargs) -> VisualizationConfig:
# #     """Create configuration for solution visualization."""
# #     defaults = {
# #         'cmap': 'viridis',
# #         # 'title': 'Numerical Solution',
# #         'colorbar_label': r'$u_h$'
# #     }
# #     defaults.update(kwargs)
# #     return VisualizationConfig(kind=kind, **defaults)

# # def error_config(
# #     kind: str = 'contourf',
# #     error_type: ErrorType = ErrorType.ABSOLUTE,
# #     **kwargs
# # ) -> VisualizationConfig:
# #     """Create configuration for error visualization."""
# #     defaults = {
# #         'cmap': 'cmr.lavender',
# #         'colorbar_label': {
# #             ErrorType.ABSOLUTE: r'$|u_h - u|$',
# #             ErrorType.RELATIVE: r'$\frac{|u_h - u|}{|u|}$',
# #             ErrorType.LOG: r'$\log\left(\frac{|u_h - u|}{|u|}\right)$'
# #         }[error_type],
# #         'error_type': error_type,
# #         'symmetric_scale': False,
# #         'vmin': None if error_type == ErrorType.LOG else 0,  # Pour LOG, on veut voir les valeurs négatives
# #         'cbar': True,  # On veut généralement une colorbar pour les erreurs
# #         'cbar_props': {
# #             'label': {
# #                 ErrorType.ABSOLUTE: r'$|u_h - u|$',
# #                 ErrorType.RELATIVE: r'$\frac{|u_h - u|}{|u|}$',
# #                 ErrorType.LOG: r'$\log\left(\frac{|u_h - u|}{|u|}\right)$'
# #             }[error_type],
# #             'pad': '4%',
# #             'size': '3%'
# #         }
# #     }
# #     if error_type == ErrorType.LOG:
# #         defaults['vmin'] = None  # Laisser matplotlib déterminer les limites pour LOG
# #         defaults['vmax'] = None
# #     defaults.update(kwargs)
# #     return VisualizationConfig(kind=kind, **defaults)
#     @conditional_style_context_with_visualization_config()
#     def plot_paired_fields(self, 
#                           fields: list[np.ndarray], 
#                           config: VisualizationConfig,
#                           titles: list[str] = None,
#                           is_error: bool = False) -> Tuple[Figure, list[Axes]]:
#         if len(fields) != 2:
#             raise ValueError("Expected exactly two fields to plot")
            
#         fig = CustomFigure(figsize=config.figsize or (12, 5))
#         axes = []
        
#         # Compute shared color limits if not specified
#         if config.vmin is None or config.vmax is None:
#             if is_error and config.symmetric_scale:
#                 max_abs = max(np.max(np.abs(field)) for field in fields)
#                 vmin, vmax = -max_abs, max_abs
#             else:
#                 vmin = min(np.min(field) for field in fields)
#                 vmax = max(np.max(field) for field in fields)
#             config = replace(config, vmin=vmin, vmax=vmax)
        
#         # Plot each field
#         for i, field in enumerate(fields):
#             ax = fig.add_subplot(1, 2, i+1)
#             ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
#             if titles and i < len(titles):
#                 ax.set_title(titles[i])
                
#             subplot_config = replace(config, figsize=None, cbar=False, save_name=None)
            
#             if config.kind == 'contourf':
#                 _, ax, mappable = self._plot_contourf_with_mappable(field, fig, ax, subplot_config)
#             else:
#                 _, ax, mappable = self._plot_trisurface_with_mappable(field, fig, ax, subplot_config)
                
#             axes.append(ax)
            
#         # Add shared colorbar
#         show_colorbar, colorbar_props = self.add_colorbar(config)
#         if show_colorbar:
#             fig.custom_colorbar(mappable, ax=axes[-1], **colorbar_props)
        
#         fig.adjust_layout()
        
#         if config.save_name:
#             fig.savefig(config.save_name, bbox_inches='tight', dpi=config.dpi)
            
#         return fig, axes

#     def plot_correctors(self, 
#                        correctors: list[np.ndarray],
#                        config: Optional[VisualizationConfig] = None,
#                        **kwargs) -> Tuple[Figure, list[Axes]]:
#         """Plot two correctors side by side with shared colorbar."""
#         if config is None:
#             config = solution_config(**kwargs)
        
#         config = replace(config, 
#                         colorbar_label=r'$\omega_i$',
#                         cmap='viridis')
        
#         return self.plot_paired_fields(
#             correctors,
#             config,
#             titles=config.corrector_labels
#         )
        
#     def plot_corrector_errors(self,
#                             computed_correctors: list[np.ndarray],
#                             exact_correctors: list[np.ndarray],
#                             config: Optional[VisualizationConfig] = None,
#                             **kwargs) -> Tuple[Figure, list[Axes]]:
#         """Plot errors for both correctors side by side with shared colorbar."""
#         if config is None:
#             config = error_config(**kwargs)
            
#         errors = [exact - computed for exact, computed in zip(exact_correctors, computed_correctors)]
        
#         config = replace(config, 
#                         colorbar_label=r'$|\omega_i - \omega_i^{ex}|$',
#                         cmap='cmr.lavender')
        
#         return self.plot_paired_fields(
#             errors,
#             config,
#             titles=config.corrector_error_labels,
#             is_error=True
#         )

"""
Enhanced visualization module for finite element solutions and errors.
Focuses on two main types of plots:
- Contour fills (2D)
- Triangular surfaces (3D)
"""

from dataclasses import dataclass, replace, field
from typing import Literal, Optional, Tuple, Union, Any, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation
from enum import Enum

from hfem.viz.custom_components import CustomFigure
from hfem.viz.conditional_style_context import conditional_style_context_with_visualization_config

class ErrorType(Enum):
    """Types of errors that can be visualized."""
    ABSOLUTE = 'absolute'
    RELATIVE = 'relative'
    LOG = 'log'

@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for FEM solution visualization."""
    # Plot type
    kind: str = 'contourf'
    
    # Visual parameters
    cmap: str = 'cmr.lavender'
    title: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    alpha: float = 1.0
    colorbar_label: str = r'$u_h$'
    
    # Figure settings
    figsize: Tuple[int, int] = None
    dpi: int = 100
    
    # Colorbar settings
    cbar: bool = True
    cbar_props: dict = field(default_factory=lambda: {'pad': 0.05, 'fraction': 0.046})
    
    # Contour settings
    num_levels: int = 30
    
    # 3D view settings
    view_elevation: float = 30
    view_azimuth: float = -60
    
    # Error specific settings
    error_type: ErrorType = ErrorType.ABSOLUTE
    symmetric_scale: bool = True
    
    # Labels for special cases
    corrector_labels: Tuple[str, str] = (r'$\omega_1$', r'$\omega_2$')
    corrector_error_labels: Tuple[str, str] = (r'$|\omega_1^\eta - \omega_1|$', r'$|\omega_2^\eta - \omega_2|$')
    
    # Save settings
    save_name: Optional[str] = None

class FEMVisualizer:
    """Handles visualization of finite element solutions and errors."""
    
    def __init__(self, node_coords: np.ndarray, triangles: np.ndarray):
        """Initialize with mesh data."""
        self.node_coords = node_coords
        self.triangles = triangles
        self.triangulation = Triangulation(node_coords[:, 0], node_coords[:, 1], triangles)

    def _get_plot_levels(self, field: np.ndarray, config: VisualizationConfig) -> np.ndarray:
        """Calculate plot levels."""
        vmin = config.vmin if config.vmin is not None else np.min(field)
        vmax = config.vmax if config.vmax is not None else np.max(field)
        return np.linspace(vmin, vmax, config.num_levels)

    def _configure_axes_2d(self, ax: Axes, config: VisualizationConfig):
        """Configure axes properties."""
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        if config.title:
            ax.set_title(config.title)
    
    def _configure_axes_3d(self, ax: Axes, config: VisualizationConfig):
        """Configure axes properties."""
        ax.set_aspect('equalxy')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        if config.title:
            ax.set_title(config.title)

    def _plot_2d(self, field: np.ndarray, ax: Axes, config: VisualizationConfig) -> Any:
        """Create 2D contour plot."""
        levels = self._get_plot_levels(field, config)
        
        im = ax.tricontourf(
            self.triangulation, 
            field,
            levels=levels,
            cmap=config.cmap,
            alpha=config.alpha
        )
        
        ax.tricontour(
            self.triangulation, 
            field,
            levels=levels,
            colors='k',
            alpha=0.2,
            linewidths=0.25
        )
        
        return im

    def _plot_3d(self, field: np.ndarray, ax: Axes, config: VisualizationConfig) -> Any:
        """Create 3D surface plot."""
        im = ax.plot_trisurf(
            self.triangulation,
            field,
            cmap=config.cmap,
            alpha=config.alpha,
            vmin=config.vmin,
            vmax=config.vmax
        )
        
        ax.view_init(config.view_elevation, config.view_azimuth)
        ax.grid(True, alpha=0.3)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            
        return im

    @conditional_style_context_with_visualization_config()
    def plot_field(self, field: np.ndarray, config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Plot a single field."""
        # Create figure
        if config.kind == 'trisurface':
            # fig = plt.figure(figsize=config.figsize)
            fig = plt.figure(figsize=config.figsize, FigureClass=CustomFigure)
            ax = fig.add_subplot(111, projection='3d')
            self._configure_axes_3d(ax, config)
        else:
            fig = plt.figure(figsize=config.figsize, FigureClass=CustomFigure)
            ax = fig.add_subplot(111)
            self._configure_axes_2d(ax, config)
            # fig, ax = plt.subplots(figsize=config.figsize)
        
        # Create plot
        im = self._plot_2d(field, ax, config) if config.kind == 'contourf' else self._plot_3d(field, ax, config)
        
        # Configure axes
        
        
        # Add colorbar
        if config.cbar:
            fig.custom_colorbar(im, ax=ax, label=config.colorbar_label, **config.cbar_props)
        # if config.cbar:
        #     plt.colorbar(im, ax=ax, label=config.colorbar_label, **config.cbar_props)
        fig.adjust_layout()
        # plt.tight_layout()
        
        if config.save_name:
            fig.savefig(config.save_name, bbox_inches='tight', dpi=config.dpi)
        
        return fig, ax

    @conditional_style_context_with_visualization_config()
    def plot_paired_fields(self, fields: List[np.ndarray], config: VisualizationConfig,
                          titles: Optional[List[str]] = None,
                          is_error: bool = False) -> Tuple[Figure, List[Axes]]:
        """Plot two fields side by side."""
        if len(fields) != 2:
            raise ValueError("Expected exactly two fields to plot")
        
        # Create figure
        if config.kind == 'trisurface':
            # fig = plt.figure(figsize=config.figsize)
            fig = plt.figure(figsize=config.figsize, FigureClass=CustomFigure)
            # axes = [fig.add_subplot(121, projection='3d'), 
            #        fig.add_subplot(122, projection='3d')]
            axes = fig.subplots(nrows=1, ncols=2, 
                            sharex=True,
                            sharey=True,
                            subplot_kw = dict(projection = '3d')
                            )
            
        else:
            fig = plt.figure(figsize=config.figsize, FigureClass=CustomFigure)
            # axes = [fig.add_subplot(121), 
            #        fig.add_subplot(122)]
            axes = fig.subplots(nrows=1, ncols=2, 
                            sharex=True,
                            sharey=True,
                            # subplot_kw = dict(projection = '3d')
                            )
            # fig, axes = plt.subplots(1, 2, figsize=config.figsize)
        
        # Compute shared color limits
        if config.vmin is None or config.vmax is None:
            vmin = min(np.min(field) for field in fields)
            vmax = max(np.max(field) for field in fields)
            # if is_error and config.symmetric_scale:
            #     max_abs = max(abs(vmin), abs(vmax))
            #     vmin, vmax = -max_abs, max_abs
            config = replace(config, vmin=vmin, vmax=vmax)
        
        # Plot fields
        for i, (field, ax) in enumerate(zip(fields, axes)):
            im = self._plot_2d(field, ax, config) if config.kind == 'contourf' else self._plot_3d(field, ax, config)
            self._configure_axes_2d(ax, config)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        
        # Add colorbar
        if config.cbar:
            fig.custom_colorbar(im, ax=axes[-1], label=config.colorbar_label, **config.cbar_props)
            # plt.colorbar(im, ax=axes, label=config.colorbar_label, **config.cbar_props)
        
        axes[-1].set_ylabel("")
        
        fig.adjust_layout()
        # plt.tight_layout()
        
        if config.save_name:
            fig.savefig(config.save_name, bbox_inches='tight', dpi=config.dpi)
        
        return fig, axes

    def plot_solution(self, solution: np.ndarray, config: Optional[VisualizationConfig] = None) -> Tuple[Figure, Axes]:
        """Plot numerical solution."""
        if config is None:
            config = solution_config()
        return self.plot_field(solution, config)

    def plot_error(self, numerical: np.ndarray, exact: np.ndarray, config: Optional[VisualizationConfig] = None) -> Tuple[Figure, Axes]:
        """Plot error between numerical and exact solutions."""
        if config is None:
            config = error_config()
            
        error = self._compute_error(numerical, exact, config.error_type)
        return self.plot_field(error, config)

    def plot_correctors(self, correctors: List[np.ndarray], config: Optional[VisualizationConfig] = None) -> Tuple[Figure, List[Axes]]:
        """Plot correctors side by side."""
        if config is None:
            config = solution_config()
        config = replace(config, colorbar_label=r'$\omega_i$')
        return self.plot_paired_fields(correctors, config, titles=config.corrector_labels)

    def plot_corrector_errors(self, computed: List[np.ndarray], exact: List[np.ndarray], config: Optional[VisualizationConfig] = None) -> Tuple[Figure, List[Axes]]:
        """Plot corrector errors side by side."""
        if config is None:
            config = error_config()
        errors = [e - c for c, e in zip(computed, exact)]
        config = replace(config, colorbar_label=r'$|\omega_i^\eta - \omega_i|$')
        return self.plot_paired_fields(np.abs(errors), config, titles=config.corrector_error_labels, is_error=True)

    def _compute_error(self, numerical: np.ndarray, exact: np.ndarray, error_type: ErrorType) -> np.ndarray:
        """Compute error based on specified type."""
        error = exact - numerical
        if error_type == ErrorType.RELATIVE:
            denominator = np.maximum(np.abs(exact), np.finfo(float).eps)
            return np.abs(error) / denominator
        elif error_type == ErrorType.LOG:
            denominator = np.maximum(np.abs(exact), np.finfo(float).eps)
            rel_error = np.abs(error) / denominator
            return np.log(np.maximum(rel_error, np.finfo(float).eps))
        return np.abs(error)  # ABSOLUTE

def solution_config(**kwargs) -> VisualizationConfig:
    """Create configuration for solution visualization."""
    return VisualizationConfig(**kwargs)

def error_config(error_type: ErrorType = ErrorType.ABSOLUTE, **kwargs) -> VisualizationConfig:
    """Create configuration for error visualization."""
    defaults = {
        'cmap': 'cmr.lavender',
        'error_type': error_type,
        'symmetric_scale': True,
        'vmin': None if error_type == ErrorType.LOG else 0,
    }
    defaults.update(kwargs)
    return VisualizationConfig(**defaults)