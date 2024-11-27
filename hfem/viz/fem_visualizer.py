"""
Enhanced visualization module for finite element solutions and errors.
Focuses on two main types of plots:
- Contour fills (2D)
- Triangular surfaces (3D)
"""

from dataclasses import dataclass, replace, field
from typing import Literal, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation
from enum import Enum

from hfem.viz.custom_components import CustomFigure
from hfem.viz.conditional_style_context import conditional_style_context, conditional_style_context_with_visualization_config

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
    cmap: str = 'viridis'
    title: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    alpha: float = 1.0
    colorbar_label: str = r'$u_h$'
    
    # Figure settings
    figsize: Tuple[int, int] = None
    dpi: int = 100
    
    # Custom Colobar settings
    cbar: bool = False
    cbar_props: dict = field(default_factory=dict)
    
    # 3D view settings (for trisurface)
    view_elevation: float = 30
    view_azimuth: float = -60
    
    # Contour settings
    num_levels: int = 50
    
    # Error specific settings
    error_type: ErrorType = ErrorType.ABSOLUTE
    symmetric_scale: bool = True
    
    # save settings
    save_name: Optional[str] = None

class FEMVisualizer:
    """Handles visualization of finite element solutions and errors."""
    
    def __init__(self, node_coords: np.ndarray, triangles: np.ndarray):
        """
        Initialize visualizer with mesh data.
        
        Parameters
        ----------
        node_coords : np.ndarray
            Nodal coordinates (N x 2)
        triangles : np.ndarray
            Triangle connectivity (M x 3)
        """
        self.node_coords = node_coords
        self.triangles = triangles
        self.triangulation = Triangulation(
            node_coords[:, 0], 
            node_coords[:, 1], 
            triangles
        )

    def create_figure(self, config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Create figure and axes based on visualization type."""
        fig = plt.figure(
            figsize=config.figsize, 
            dpi=config.dpi, 
            FigureClass=CustomFigure)
        # print(f"figure créée et c'est une custom figure : {isinstance(fig, CustomFigure)}")
        
        if config.kind == 'trisurface':
            ax = fig.add_subplot(111, projection='3d')
            ax.set(aspect = 'equalxy', xlabel = r"$x$", ylabel = r"$y$")
            ax.view_init(config.view_elevation, config.view_azimuth)
        else:
            ax = fig.add_subplot(111)
            ax.set(aspect = 'equal', xlabel = r"$x$", ylabel = r"$y$")        
        if config.title:
            ax.set_title(config.title)
            
        return fig, ax
    
    def process_error_field(self, error_field: np.ndarray, 
                          config: VisualizationConfig) -> np.ndarray:
        """Process error field based on error type."""
        if config.error_type == ErrorType.RELATIVE:
            denominator = np.max(np.abs(error_field))
            if denominator == 0:
                return error_field
            return error_field / denominator
        elif config.error_type == ErrorType.LOG:
            min_positive = np.min(np.abs(error_field[error_field != 0]))
            field = np.copy(error_field)
            field[field == 0] = min_positive
            return np.log(field)
        else:  # ABSOLUTE
            return error_field

    def get_error_limits(self, error_field: np.ndarray, 
                        config: VisualizationConfig) -> Tuple[float, float]:
        """Compute appropriate limits for error visualization."""
        
        if config.symmetric_scale and (config.vmin is not None or config.vmax is not None):
            max_abs = np.max(np.abs(error_field))
            return -max_abs, max_abs
        
        if config.vmin is not None:
            if config.vmax is not None:
                return config.vmin, config.vmax
            else:
                return config.vmin, np.max(error_field)
        if config.vmax is not None:
            if config.vmin is not None:
                return config.vmin, config.vmax
            else:
                return np.min(error_field), config.vmax
        
        return np.min(error_field), np.max(error_field)

    def add_colorbar(self, config: VisualizationConfig) -> Tuple[bool, dict]:
        """
        Determine if a colorbar should be added and with which properties.
        
        Parameters
        ----------
        config : VisualizationConfig
            Visualization configuration
            
        Returns
        -------
        Tuple[bool, dict]
            - Boolean indicating if colorbar should be added
            - Dictionary of colorbar properties to use
        """
        # Si cbar est explicitement False, pas de colorbar quoi qu'il arrive
        if config.cbar == False:
            return False, {}
        
        # Propriétés par défaut pour la colorbar
        default_props = {
            'label': config.colorbar_label,
            'pad': '2.5%',
            'size': '2%'
        }
        
        # Si des propriétés spécifiques sont fournies, les utiliser
        if config.cbar_props:
            default_props.update(config.cbar_props)
        
        return True, default_props

    def plot_solution(self, 
                      solution: np.ndarray,
                      config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Plot numerical solution."""
        return self.plot_field(solution, config, is_error=False)
    
    def plot_error(self,
                  numerical_sol: np.ndarray,
                  exact_sol: np.ndarray,
                  config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Plot error between numerical and exact solutions."""
        # Calcul de l'erreur
        raw_error = exact_sol - numerical_sol
        
        # Traitement selon le type d'erreur
        if config.error_type == ErrorType.RELATIVE:
            # Éviter la division par zéro
            denominator = np.maximum(np.abs(exact_sol), np.finfo(float).eps)
            error = np.abs(raw_error/denominator)
        elif config.error_type == ErrorType.LOG:
            # Éviter log(0) et valeurs négatives
            denominator = np.maximum(np.abs(exact_sol), np.finfo(float).eps)
            rel_error = np.abs(raw_error)/denominator
            error = np.log(np.maximum(rel_error, np.finfo(float).eps))
        else:  # ABSOLUTE
            error = np.abs(raw_error)
            
        # Mise à jour des limites si nécessaire
        if config.vmin is None or config.vmax is None:
            vmin, vmax = self.get_error_limits(error, config)
            config = replace(config, vmin=vmin, vmax=vmax)
            
        return self.plot_field(error, config, is_error=True)

    # @conditional_style_context()
    @conditional_style_context_with_visualization_config()
    def plot_field(self, field: np.ndarray, config: VisualizationConfig,
                  is_error: bool = False) -> Tuple[Figure, Axes]:
        """Plot finite element field."""        
        fig, ax = self.create_figure(config)
        
        if config.kind == 'contourf':
            fig, ax = self._plot_contourf(field, fig, ax, config)
        else:
            fig, ax = self._plot_trisurface(field, fig, ax, config)
        
        if isinstance(fig, CustomFigure):
            # print(f"c'est une custom figure")
            fig.adjust_layout()
        
        if config.save_name:
            plt.savefig(config.save_name, bbox_inches='tight', dpi=config.dpi)

        # if isinstance(fig, CustomFigure):
        #     fig.adjust_layout()

        return fig, ax    
    
    def _plot_contourf(self, field: np.ndarray, fig: Figure, ax: Axes, 
                      config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Create filled contour plot."""
        levels = np.linspace(
            config.vmin if config.vmin is not None else np.min(field),
            config.vmax if config.vmax is not None else np.max(field),
            config.num_levels
        )
        
        tcf = ax.tricontourf(
            self.triangulation, 
            field,
            levels=levels,
            cmap=config.cmap,
            alpha=config.alpha,
        )
        
        # Add thin contour lines for better visualization
        ax.tricontour(
            self.triangulation, 
            field,
            levels=levels,
            colors='k',
            alpha=0.2,
            linewidths=0.5
        )
        
        show_colorbar, colorbar_props = self.add_colorbar(config)
        if show_colorbar:
            fig.custom_colorbar(tcf, ax=ax, **colorbar_props)
        return fig, ax

    def _plot_trisurface(self, field: np.ndarray, fig: Figure, ax: Axes,
                        config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Create 3D surface plot."""
        surf = ax.plot_trisurf(
            self.triangulation,
            field,
            cmap=config.cmap,
            alpha=config.alpha,
            vmin=config.vmin,
            vmax=config.vmax
        )
        
        show_colorbar, colorbar_props = self.add_colorbar(config)
        if show_colorbar:
            fig.custom_colorbar(surf, ax=ax, **colorbar_props)
        else:
            ax.set(zlabel=config.colorbar_label)
        
        # Improve 3D visualization
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        return fig, ax

# Configuration functions
def solution_config(kind: str = 'contourf', **kwargs) -> VisualizationConfig:
    """Create configuration for solution visualization."""
    defaults = {
        'cmap': 'viridis',
        # 'title': 'Numerical Solution',
        'colorbar_label': r'$u_h$'
    }
    defaults.update(kwargs)
    return VisualizationConfig(kind=kind, **defaults)

def error_config(
    kind: str = 'contourf',
    error_type: ErrorType = ErrorType.ABSOLUTE,
    **kwargs
) -> VisualizationConfig:
    """Create configuration for error visualization."""
    defaults = {
        'cmap': 'cmr.lavender',
        'colorbar_label': {
            ErrorType.ABSOLUTE: r'$|u_h - u|$',
            ErrorType.RELATIVE: r'$\frac{|u_h - u|}{|u|}$',
            ErrorType.LOG: r'$\log\left(\frac{|u_h - u|}{|u|}\right)$'
        }[error_type],
        'error_type': error_type,
        'symmetric_scale': False,
        'vmin': None if error_type == ErrorType.LOG else 0,  # Pour LOG, on veut voir les valeurs négatives
        'cbar': True,  # On veut généralement une colorbar pour les erreurs
        'cbar_props': {
            'label': {
                ErrorType.ABSOLUTE: r'$|u_h - u|$',
                ErrorType.RELATIVE: r'$\frac{|u_h - u|}{|u|}$',
                ErrorType.LOG: r'$\log\left(\frac{|u_h - u|}{|u|}\right)$'
            }[error_type],
            'pad': '4%',
            'size': '3%'
        }
    }
    if error_type == ErrorType.LOG:
        defaults['vmin'] = None  # Laisser matplotlib déterminer les limites pour LOG
        defaults['vmax'] = None
    defaults.update(kwargs)
    return VisualizationConfig(kind=kind, **defaults)