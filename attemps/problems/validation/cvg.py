from typing import Union, Optional, TypeVar, Protocol, Dict, List
from dataclasses import dataclass
from pathlib import Path
import abc
from enum import Enum, auto
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cmasher as cmr
from dataclasses import asdict

# Type definitions
T = TypeVar('T', bound='BaseConvergenceData')
PathLike = Union[str, Path]

class NormType(Enum):
    L2 = auto()
    H1 = auto()
    TENSOR = auto()

class DataValidator(Protocol):
    def validate(self) -> bool:
        ...

@dataclass
class BaseConvergenceData:
    """Base class for convergence data with common validation logic."""
    h: list[float]
    n_nodes: list[int]

    def validate(self) -> bool:
        """Validate basic data consistency."""
        if len(self.h) != len(self.n_nodes):
            raise ValueError("Length mismatch between h and n_nodes")
        if not all(x > 0 for x in self.h):
            raise ValueError("Mesh sizes must be positive")
        if not all(isinstance(x, int) and x > 0 for x in self.n_nodes):
            raise ValueError("Node counts must be positive integers")
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to pandas DataFrame."""
        return pd.DataFrame(asdict(self))

    @classmethod
    def from_dataframe(cls: type[T], df: pd.DataFrame) -> T:
        """Create instance from DataFrame."""
        return cls(**df.to_dict('list'))

@dataclass
class StandardConvergenceData(BaseConvergenceData):
    """Enhanced standard Poisson problem convergence data."""
    l2_errors: list[float]
    h1_errors: list[float]
    boundary_type: str
    
    def validate(self) -> bool:
        """Validate data consistency and error values."""
        super().validate()
        if not len(self.l2_errors) == len(self.h1_errors) == len(self.h):
            raise ValueError("Length mismatch in error arrays")
        if not all(x >= 0 for x in self.l2_errors + self.h1_errors):
            raise ValueError("Errors must be non-negative")
        return True

    def compute_convergence_rates(self) -> dict[str, list[float]]:
        """Compute convergence rates between consecutive mesh sizes."""
        rates = {}
        for error_type in ['l2', 'h1']:
            errors = getattr(self, f"{error_type}_errors")
            rates[f"{error_type}_rates"] = [
                np.log(errors[i]/errors[i+1]) / np.log(self.h[i]/self.h[i+1])
                for i in range(len(self.h)-1)
            ]
        return rates

@dataclass
class PenalizedCellConvergenceData(BaseConvergenceData):
    """Enhanced penalized cell problems convergence data."""
    eta: float
    l2_errors_corrector1: list[float]
    h1_errors_corrector1: list[float]
    l2_errors_corrector2: list[float]
    h1_errors_corrector2: list[float]
    tensor_errors: list[float]

    def validate(self) -> bool:
        """Validate data consistency and error values."""
        super().validate()
        if not self.eta > 0:
            raise ValueError("eta must be positive")
        
        error_lists = [
            self.l2_errors_corrector1,
            self.h1_errors_corrector1,
            self.l2_errors_corrector2,
            self.h1_errors_corrector2,
            self.tensor_errors
        ]
        
        if not all(len(errors) == len(self.h) for errors in error_lists):
            raise ValueError("Length mismatch in error arrays")
        if not all(all(x >= 0 for x in errors if x is not None) for errors in error_lists):
            raise ValueError("Errors must be non-negative when present")
        return True

    def get_error_data(self, norm_type: NormType, corrector: Optional[int] = None) -> list[float]:
        """Get error data for specific norm type and corrector."""
        if norm_type == NormType.TENSOR:
            return self.tensor_errors
        
        if corrector not in [1, 2]:
            raise ValueError("Corrector must be 1 or 2 for L2 and H1 norms")
            
        error_attr = f"{norm_type.name.lower()}_errors_corrector{corrector}"
        return getattr(self, error_attr)

@dataclass
class MultiEtaPenalizedCellConvergenceData:
    data: dict[float, PenalizedCellConvergenceData] 
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'MultiEtaPenalizedCellConvergenceData':
        """Create instance from DataFrame."""
        all_data = {}
        for eta in df['eta'].unique():
            eta_df = df[df['eta'] == eta]
            all_data[eta] = PenalizedCellConvergenceData(**eta_df.to_dict('list'))
        return cls(data=all_data)
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to pandas DataFrame."""
        all_data = []
        for eta, eta_data in self.data.items():
            df = pd.DataFrame(asdict(eta_data))
            all_data.append(df)
        return pd.concat(all_data)

class ConvergencePlotter:
    """Class to handle all plotting functionality."""
    
    @staticmethod
    def setup_figure(title: Optional[str] = None) -> tuple[plt.Figure, plt.Axes]:
        """Set up figure with common styling."""
        fig, ax = plt.subplots(layout='constrained')
        if title:
            ax.set_title(title)
        return fig, ax

    @staticmethod
    def add_convergence_triangle(ax: plt.Axes, x: np.ndarray, y: np.ndarray, 
                               rate: float, color: str) -> None:
        """Add improved convergence rate triangle to plot."""
        try:
            x_min, x_max = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            y_min = y[np.isfinite(y)].min()
            
            tri_x = np.array([x_max-1, x_max, x_max-1])
            tri_y = np.array([y_min, y_min, y_min+rate])
            
            ax.add_patch(plt.Polygon(np.column_stack((tri_x, tri_y)), 
                                   facecolor=color, alpha=0.3))
            
            ax.annotate(
                text=rf'$\boldsymbol{{{rate}}}$',
                xy=((tri_x[-1] + tri_x[0])/2, (tri_y[-1] + tri_y[0])/2),
                xytext=(2*rate, -2*rate),
                textcoords='offset points',
                va='center', ha='left',
                color=color,
                fontsize=12,
                weight='bold'
            )
        except ValueError as e:
            warnings.warn(f"Could not add convergence triangle: {e}")

    def plot_corrector_convergence(self, data: Dict[float, PenalizedCellConvergenceData],
                                 norm_type: NormType = NormType.H1,
                                 save_path: Optional[PathLike] = None) -> tuple[plt.Figure, plt.Axes]:
        """Enhanced plotting for corrector convergence."""
        fig, ax = self.setup_figure()
        colors = cmr.lavender(np.linspace(0, 1, len(data)))
        
        # Add improved legend handles
        legend_elements = self._create_legend_elements(norm_type)
        
        for (eta, conv_data), color in zip(sorted(data.items()), colors):
            self._plot_corrector_data(ax, conv_data, color, norm_type)
            
        self._finalize_plot(ax, fig, legend_elements, save_path)
        return fig, ax

    def _create_legend_elements(self, norm_type: NormType) -> list:
        """Create legend elements based on norm type."""
        return [
            plt.Line2D([0], [0], color="black", lw=2, linestyle=":", 
                      label=rf"$\|\cdot\|_{{{norm_type.name}}}$"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
                      markersize=8, label=r'$e_{1,h}^\eta$'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k',
                      markersize=8, label=r'$e_{2,h}^\eta$')
        ]

    def _plot_corrector_data(self, ax: plt.Axes, data: PenalizedCellConvergenceData,
                            color: str, norm_type: NormType) -> None:
        """Plot data for a single corrector."""
        for corrector in [1, 2]:
            marker = 'o' if corrector == 1 else '^'
            errors = data.get_error_data(norm_type, corrector)
            if any(e is not None for e in errors):
                ax.plot(data.h, errors, f'{marker}:', color=color, alpha=0.7)
                
    def plot_tensor_convergence(
        self, 
        data: Dict[float, PenalizedCellConvergenceData],
        rate: Optional[float] = None,
        save_path: Optional[PathLike] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot tensor convergence for multiple eta values.
        
        Args:
            data: Dictionary mapping eta values to convergence data
            rate: Expected convergence rate to show in triangle
            save_path: Optional path to save the figure
            
        Returns:
            Tuple of (Figure, Axes)
        """
        fig, ax = self.setup_figure()
        
        # Configure axes
        ax.set(
            xlabel=r'$\log\left(\frac{1}{h}\right)$',
            ylabel=r'$\log\left(\frac{\left\|A^* - A^*_\eta\right\|_F}{\left\|A^*\right\|_F}\right)$'
        )
        
        # Generate colors for different eta values
        colors = cmr.lavender(np.linspace(0, 1, len(data.data.items())))
        
        # Sort data by eta values for consistent plotting
        eta_sorted_data = sorted(data.data.items())
        
        # Plot data for each eta value
        for i, (eta, conv_data) in enumerate(eta_sorted_data):
            log_h = np.log(1/np.array(conv_data.h))
            tensor_errors = np.array(conv_data.get_error_data(NormType.TENSOR))
            
            # Filter out any None values
            # mask = tensor_errors is not None
            # if not any(mask):
            #     warnings.warn(f"No valid tensor errors for eta={eta}")
            #     continue
                
            ax.plot(
                log_h,
                np.log(tensor_errors),
                'o--',
                label=fr'${eta:.2g}$',
                color=colors[i],
                markersize=4
            )
        
        # Add convergence rate triangle if specified
        if rate is not None and eta_sorted_data:
            # Use first dataset for triangle placement
            data_first_eta = eta_sorted_data[0][1]
            log_h = np.log(1/np.array(data_first_eta.h))
            tensor_errors = np.array(data_first_eta.get_error_data(NormType.TENSOR))
            
            # Only add triangle if we have valid data
            if any(np.isfinite(tensor_errors)):
                self.add_convergence_triangle(
                    ax=ax,
                    x=log_h,
                    y=np.log(tensor_errors),
                    rate=rate,
                    color=colors[0]
                )
        
        # Add legend
        fig.legend(
            loc='outside right center',
            frameon=True,
            title=r"$\eta$ values"
        )
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig, ax

    def _validate_tensor_data(self, data: Dict[float, PenalizedCellConvergenceData]) -> bool:
        """Validate tensor convergence data."""
        if not data:
            raise ValueError("Empty data dictionary provided")
        
        for eta, conv_data in data.items():
            if not isinstance(conv_data, PenalizedCellConvergenceData):
                raise TypeError(f"Invalid data type for eta={eta}")
            if not any(error is not None for error in conv_data.tensor_errors):
                warnings.warn(f"No valid tensor errors for eta={eta}")
                
        return True

    def _finalize_plot(self, ax: plt.Axes, fig: plt.Figure, 
                      legend_elements: list, save_path: Optional[PathLike]) -> None:
        """Finalize plot with labels, legends, and saving."""
        ax.set_xlabel('h')
        ax.set_ylabel('Error')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        fig.legend(handles=legend_elements, loc='outside right upper',
                  frameon=True, title="Symbols")
        
        if save_path:
            plt.savefig(save_path)

class DataIO:
    """Class to handle data input/output operations."""
    
    @staticmethod
    def save_data(data: Union[BaseConvergenceData, Dict[float, BaseConvergenceData]],
                 save_dir: PathLike, filename: Optional[str] = None) -> Path:
        """Enhanced data saving with validation."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, dict):
            all_data = []
            for key, value in data.items():
                df = value.to_dataframe()
                df['group_key'] = key
                all_data.append(df)
            final_df = pd.concat(all_data)
        else:
            final_df = data.to_dataframe()
        
        if filename is None:
            filename = f"convergence_{data.__class__.__name__}.csv"
            
        save_path = save_dir / filename
        final_df.to_csv(save_path, index=False)
        return save_path

    @staticmethod
    def read_data(filepath: PathLike, data_type: type[T]) -> Union[T, Dict[float, T]]:
        """Enhanced data reading with type checking."""
        df = pd.read_csv(filepath)
        
        if 'group_key' in df.columns:
            result = {}
            for key, group in df.groupby('group_key'):
                group = group.drop('group_key', axis=1)
                result[key] = data_type.from_dataframe(group)
            return result
        
        return data_type.from_dataframe(df)

def generate_test_data(h_values: Optional[List[float]] = None,
                      eta_values: Optional[List[float]] = None) -> Dict[float, PenalizedCellConvergenceData]:
    """Generate test data with improved parameter control."""
    if h_values is None:
        h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
    if eta_values is None:
        eta_values = [1e-1, 1e-2, 1e-3, 1e-4]
        
    n_nodes = [int(1/h**2) for h in h_values]  # More realistic node count
    
    test_data = {}
    for eta in eta_values:
        base_rate = np.array(h_values)
        noise = lambda: 1 + 0.1 * np.random.randn(len(h_values))
        
        l2_rate = base_rate**2 * noise()
        h1_rate = base_rate * noise()
        eta_factor = np.sqrt(eta)
        
        test_data[eta] = PenalizedCellConvergenceData(
            eta=eta,
            h=h_values,
            n_nodes=n_nodes,
            l2_errors_corrector1=l2_rate * eta_factor,
            h1_errors_corrector1=h1_rate * eta_factor,
            l2_errors_corrector2=l2_rate * eta_factor * 1.2,
            h1_errors_corrector2=h1_rate * eta_factor * 1.2,
            tensor_errors=l2_rate * eta * 0.5
        )
        
    return test_data

if __name__ == '__main__':
    # fname = "cell_problem_case_ii"
    
    # data = read_data(filepath=f"results/convergences/{fname}.csv", data_type="multi-eta penalized cell")
    
    # # plot_corrector_convergence(data)
    # # plt.show()
    # plot_tensor_convergence(data, rate=2, save_name=f"results/convergences/convergence_{fname}")
    # plt.show()
    fname = "cell_problem_case_ii"
    results_dir = Path("results/convergences")
    input_path = results_dir / f"{fname}.csv"
    output_path = results_dir / f"convergence_{fname}.pdf"

    # Lecture des données avec typage explicite
    data = DataIO.read_data(
        filepath=input_path, 
        data_type=MultiEtaPenalizedCellConvergenceData
    )

    # Création du plotter
    plotter = ConvergencePlotter()

    # Plot des correctors si nécessaire
    # fig, ax = plotter.plot_corrector_convergence(data)
    # plt.show()

    # Plot de la convergence du tenseur
    fig, ax = plotter.plot_tensor_convergence(
        data=data,
        rate=2,
        # save_path=output_path
    )
    plt.show()