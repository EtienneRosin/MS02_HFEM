import functools
import matplotlib.pyplot as plt
import matplotlib as mpl

# from hfem.viz import VisualizationConfig

def conditional_style_context(style_path='hfem/viz/custom_science.mplstyle'):
    """
    A decorator that conditionally applies a custom Matplotlib style when saving a plot.

    This decorator allows you to apply a specific Matplotlib style only when saving a plot,
    while leaving the default style intact for display purposes.

    Parameters:
    -----------
    style_path : str, optional
        Path to the Matplotlib style file to be used when saving the plot.
        Defaults to 'hfem/viz/custom_science.mplstyle'.

    Returns:
    --------
    decorator : callable
        A decorator function that wraps the original plotting function.

    Usage:
    ------
    @conditional_style_context()
    def plot_data(data, save_name: str = None):
        plt.figure()
        plt.plot(data)
        if save_name:
            plt.savefig(save_name)

    # Plot without custom style
    plot_data(data)
    plt.show()

    # Plot with custom style and save
    plot_data(data, save_name='my_plot.pdf')
    plt.show()

    Notes:
    ------
    - The decorator checks if a save_name is provided
    - If save_name is not None, it applies the specified Matplotlib style when saving
    - If no save_name is given, the function runs with the default Matplotlib style
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if save_name is in kwargs and not None
            if kwargs.get('save_name') is not None:
                # If save_name is present and not None, use the style context
                with plt.style.context(style_path):
                    return func(*args, **kwargs)
            else:
                # Otherwise, execute the function normally
                return func(*args, **kwargs)
        return wrapper
    return decorator


def conditional_style_context_with_visualization_config(style_path='hfem/viz/custom_science.mplstyle'):
    """
    A decorator that applies a custom Matplotlib style when saving a plot,
    based on the presence of a VisualizationConfig.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Attempt to find `config` in args or kwargs
            config = kwargs.get('config', None)
            if config is None:
                # Try to locate VisualizationConfig in positional arguments
                for arg in args:
                    from hfem.viz import VisualizationConfig
                    if isinstance(arg, VisualizationConfig):
                        config = arg
                        break
            
            # If a VisualizationConfig is found and save_name is not None, apply the style
            if config and config.save_name:
                with plt.style.context(style_path):
                    return func(*args, **kwargs)
            else:
                # Otherwise, execute the function normally
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == '__main__':
    # Example usage
    @conditional_style_context()
    def plot_data(data, save_name: str = None):
        plt.figure()
        plt.plot(data)
        if save_name:
            plt.savefig(save_name)
        # plt.close()  # Uncomment if you want to automatically close the plot

    # Usages
    data = [1, 2, 3, 4, 5]
    
    # This will not use the custom style
    plot_data(data)
    plt.show()
    
    # This will use the custom style and save the file
    plot_data(data, save_name='my_plot.pdf')
    plt.show()