from typing import Any, Literal, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

style_path = "hfem/viz/custom_science.mplstyle"

class CustomFigure(Figure):
    """
    Custom Matplotlib Figure with enhanced functionality.
    
    Inherits from ``matplotlib.figure.Figure`` and adds custom methods
    for scientific visualization, with a focus on precise colorbar positioning.
    
    Attributes
    ----------
    Inherits all standard Matplotlib Figure attributes
    
    Methods
    -------
    custom_colorbar : Create a precisely positioned colorbar
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the CustomFigure with standard Figure parameters.
        
        All parameters are passed directly to the parent Figure class.
        """
        super().__init__(*args, **kwargs)
        
    def custom_colorbar(
        self,
        mappable,
        ax = None,
        position: Literal["left", "right", "bottom", "top"] = "right",
        size: str | float = "2%",
        pad: str | float = "2%",
        use_gridspec: bool = True,
        **kwargs
    ):
        """
        Create a precisely positioned colorbar for both 2D and 3D axes.

        Parameters
        ----------
        mappable : matplotlib.cm.ScalarMappable
            The data to be colored (e.g., ContourSet, AxesImage, 3D plot)
        ax : matplotlib.axes.Axes, optional
            Parent axes for the colorbar. If None, attempts to extract from mappable
        position : {"left", "right", "bottom", "top"}, default "right"
            Position of the colorbar relative to the main axes
        size : str or float, default "3.5%"
            Width or height of the colorbar 
        pad : str or float, default "2%"
            Padding between main axes and colorbar
        use_gridspec : bool, default True
            Use GridSpec for positioning if applicable

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The created colorbar

        Raises
        ------
        ValueError
            If no valid axes can be determined
        """
        # Validate and retrieve axes
        if ax is None:
            ax = getattr(mappable, "axes", None)
        
        if ax is None:
            raise ValueError("No axes provided or found. Cannot create colorbar.")
        
        # Check if it's a 3D plot
        is_3d = hasattr(ax, 'get_proj')
        
        if is_3d:
            # Specific handling for 3D plots
            # Convert size and pad to absolute values
            fig_width = self.get_figwidth()
            fig_height = self.get_figheight()
            
            # Convert percentage to absolute values
            def convert_to_absolute(value, total):
                if isinstance(value, str) and value.endswith('%'):
                    return float(value.rstrip('%')) / 100 * total
                return float(value)
            
            # Convert size and pad
            size_abs = convert_to_absolute(size, fig_width)
            pad_abs = convert_to_absolute(pad, fig_width)
            # pos = ax.get_position()
            # size_abs = convert_to_absolute(size, pos.width)
            # pad_abs = convert_to_absolute(pad, pos.width)
            
            # Get current position of 3D axes
            pos = ax.get_position()
            
            # Create new axes for colorbar based on position
            if position == 'right':
                cax = self.add_axes([
                    pos.x1 + pad_abs/fig_width, 
                    pos.y0, 
                    size_abs/fig_width, 
                    pos.height
                ])
            elif position == 'left':
                cax = self.add_axes([
                    pos.x0 - size_abs/fig_width - pad_abs/fig_width, 
                    pos.y0, 
                    size_abs/fig_width, 
                    pos.height
                ])
            else:
                # Fallback to default matplotlib colorbar for other positions
                return self.colorbar(mappable, ax=ax, **kwargs)
            
            return self.colorbar(mappable, cax=cax, **kwargs)
        else:
            # Existing 2D approach
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes(position, size=size, pad=pad)
            return self.colorbar(mappable, cax=cax, use_gridspec=use_gridspec, **kwargs)
    
    def adjust_layout(self, *args, **kwargs):
        """
        Custom layout adjustment method handling both 2D and 3D axes.
        
        Tries to adjust layout while avoiding common matplotlib layout issues.
        """
        try:
            # First, check if we have 3D axes
            has_3d_axes = any(hasattr(ax, 'get_proj') for ax in self.axes)
            
            if has_3d_axes:
                if len(self.axes) > 1: # may contains a colorbar
                    self.subplots_adjust(
                        left = 0,
                        bottom = 0.05,
                        right=0.85,
                        top = 0.925,
                        wspace = 0.1,
                        hspace = 0.1)
                else:
                    self.tight_layout(*args, **kwargs)
            else:
                # Standard tight layout for 2D
                self.tight_layout(*args, **kwargs)
        
        except Exception as e:
            print(f"Layout adjustment warning: {e}")
            # Fallback to no adjustment if something goes wrong

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import ma
    from matplotlib import cm, ticker

    # Generate data
    N = 100
    x = np.linspace(-3.0, 3.0, N)
    y = np.linspace(-2.0, 2.0, N)
    X, Y = np.meshgrid(x, y)
    
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
    z = Z1 + 50 * Z2
    
    # Simulate negative values for log scale demonstration
    z[:5, :5] = -1
    z = ma.masked_where(z <= 0, z)

    with plt.style.context(style_path):
        # Create figure using CustomFigure
        fig = plt.figure(FigureClass=CustomFigure)
        # ax = fig.add_subplot()
        ax = fig.add_subplot(projection = '3d')
        # cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
        cs = ax.plot_surface(X, Y, z, cmap=cm.PuBu_r)
        cbar = fig.custom_colorbar(cs, ax=ax, pad = "2.5%", size = "2%", label = r"$u_h - u$")
        
        
        ax.set_title("Custom Figure with Precise Colorbar")
        # plt.tight_layout()
        fig.adjust_layout()
        fig.savefig(fname="cizbyc.pdf")
        plt.show()