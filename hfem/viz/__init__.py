r"""
Visualization library for the HFEM (Homogenization Finite Element Method) project.

This module provides comprehensive visualization tools for:
- Solution visualization (2D and 3D)
- Error analysis
- Comparative plots

Examples
--------
>>> import hfem.viz as viz
>>> viz.mesh(problem.mesh)
>>> viz.solution(problem, cmap='viridis')
>>> viz.error_plot(problem)
"""

# from .visualization_configs import VisualizationConfig
from .fem_visualizer import FEMVisualizer, solution_config, error_config, VisualizationConfig, ErrorType