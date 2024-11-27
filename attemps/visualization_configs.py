"""
Visualization module for finite element solutions.
Focuses on two main types of plots:
- Contour fills (2D)
- Triangular surfaces (3D)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation

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
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if any(size <= 0 for size in self.figsize):
            raise ValueError("figsize components must be positive")

class FEMVisualizer:
    """Handles visualization of finite element solutions."""
    
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

    def plot_field(self, field: np.ndarray, config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """
        Plot finite element field.
        
        Parameters
        ----------
        field : np.ndarray
            Field values at nodes
        config : VisualizationConfig
            Visualization configuration
            
        Returns
        -------
        Figure, Axes
            Matplotlib figure and axes objects
        """
        fig, ax = self.create_figure(config)
        
        if config.kind == 'contourf':
            return self._plot_contourf(field, fig, ax, config)
        else:
            return self._plot_trisurface(field, fig, ax, config)

    def _plot_contourf(self, field: np.ndarray, fig: Figure, ax: Axes, 
                      config: VisualizationConfig) -> Tuple[Figure, Axes]:
        """Create filled contour plot."""
        tcf = ax.tricontourf(
            self.triangulation, 
            field,
            levels=config.num_levels,
            cmap=config.cmap,
            alpha=config.alpha,
            vmin=config.vmin,
            vmax=config.vmax
        )
        
        # Add thin contour lines for better visualization
        ax.tricontour(
            self.triangulation, 
            field,
            levels=config.num_levels,
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

def error_config(kind: str = 'contourf', **kwargs) -> VisualizationConfig:
    """Create configuration for error visualization."""
    defaults = {
        'cmap': 'RdBu_r',
        'title': 'Error',
        'colorbar_label': r'$u_h - u$'
    }
    defaults.update(kwargs)
    return VisualizationConfig(kind=kind, **defaults)