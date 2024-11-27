"""
Enhanced visualization module for finite element solutions and errors.
Focuses on two main types of plots:
- Contour fills (2D)
- Triangular surfaces (3D)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation
from enum import Enum

class ErrorType(Enum):
    """Types of errors that can be visualized."""
    ABSOLUTE = 'absolute'
    RELATIVE = 'relative'
    LOG = 'log'

@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for FEM solution visualization."""
    
    # Plot type
    kind: Literal['contourf', 'trisurface'] = 'contourf'
    
    # Visual parameters
    cmap: str = 'viridis'
    title: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    alpha: float = 1.0
    colorbar_label: str = r'$u_h$'
    
    # Figure settings
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    
    # 3D view settings (for trisurface)
    view_elevation: float = 30
    view_azimuth: float = -60
    
    # Contour settings
    num_levels: int = 50
    
    # Error specific settings
    error_type: ErrorType = ErrorType.ABSOLUTE
    symmetric_scale: bool = True  # For error plots, center colorbar at 0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if any(size <= 0 for size in self.figsize):
            raise ValueError("figsize components must be positive")

class FEMVisualizer:
    """Handles visualization of finite element solutions and errors."""
    
    def __init__(self, node_coords, triangles):
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
        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        
        if config.kind == 'trisurface':
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(config.view_elevation, config.view_azimuth)
        else:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
        
        if config.title:
            ax.set_title(config.title)
            
        return fig, ax

    def process_error_field(self, error_field: np.ndarray, 
                          config: VisualizationConfig) -> np.ndarray:
        """
        Process error field based on error type.
        
        Parameters
        ----------
        error_field : np.ndarray
            Raw error field
        config : VisualizationConfig
            Visualization configuration
            
        Returns
        -------
        np.ndarray
            Processed error field
        """
        if config.error_type == ErrorType.RELATIVE:
            # Avoid division by zero
            denominator = np.max(np.abs(error_field))
            if denominator == 0:
                return error_field
            return error_field / denominator
        elif config.error_type == ErrorType.LOG:
            # Handle negative values for log scale
            min_positive = np.min(np.abs(error_field[error_field != 0]))
            field = np.copy(error_field)
            field[field == 0] = min_positive
            return np.sign(field) * np.log10(np.abs(field))
        else:  # ABSOLUTE
            return error_field

    def get_error_limits(self, error_field: np.ndarray, 
                        config: VisualizationConfig) -> Tuple[float, float]:
        """Compute appropriate limits for error visualization."""
        if config.vmin is not None and config.vmax is not None:
            return config.vmin, config.vmax
            
        if config.symmetric_scale:
            max_abs = np.max(np.abs(error_field))
            return -max_abs, max_abs
        else:
            return np.min(error_field), np.max(error_field)

    def plot_field(self, field: np.ndarray, config: VisualizationConfig,
                  is_error: bool = False) -> Tuple[Figure, Axes]:
        """
        Plot finite element field.
        
        Parameters
        ----------
        field : np.ndarray
            Field values at nodes
        config : VisualizationConfig
            Visualization configuration
        is_error : bool
            Whether the field represents an error
            
        Returns
        -------
        Figure, Axes
            Matplotlib figure and axes objects
        """
        if is_error:
            field = self.process_error_field(field, config)
            if config.vmin is None or config.vmax is None:
                config = dataclass.replace(
                    config,
                    vmin=self.get_error_limits(field, config)[0],
                    vmax=self.get_error_limits(field, config)[1]
                )
        
        fig, ax = self.create_figure(config)
        
        if config.kind == 'contourf':
            return self._plot_contourf(field, fig, ax, config)
        else:
            return self._plot_trisurface(field, fig, ax, config)

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
            extend='both'
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
        
        plt.colorbar(tcf, ax=ax, label=config.colorbar_label)
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
        
        plt.colorbar(surf, ax=ax, label=config.colorbar_label)
        
        # Improve 3D visualization
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        return fig, ax

# Configuration presets
def solution_config(kind: str = 'contourf', **kwargs) -> VisualizationConfig:
    """Create configuration for solution visualization."""
    defaults = {
        'cmap': 'viridis',
        'title': 'Numerical Solution',
        'colorbar_label': r'$u_h$'
    }
    defaults.update(kwargs)
    return VisualizationConfig(kind=kind, **defaults)

def error_config(kind: str = 'contourf', error_type: ErrorType = ErrorType.ABSOLUTE,
                **kwargs) -> VisualizationConfig:
    """Create configuration for error visualization."""
    defaults = {
        'cmap': 'RdBu_r',
        'title': f'{error_type.value.title()} Error',
        'colorbar_label': {
            ErrorType.ABSOLUTE: r'$|u_h - u|$',
            ErrorType.RELATIVE: r'$\frac{|u_h - u|}{\max|u|}$',
            ErrorType.LOG: r'$\log_{10}|u_h - u|$'
        }[error_type],
        'error_type': error_type,
        'symmetric_scale': True
    }
    defaults.update(kwargs)
    return VisualizationConfig(kind=kind, **defaults)