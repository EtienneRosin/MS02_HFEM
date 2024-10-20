"""
@file gauss_legendre_6_points.py
@brief Module providing triangle 6-points Gauss-Legendre quadrature info.
@authors Etienne Rosin
@version 0.1
@date 2024
@note source : ANN201 - TP Stokes
"""
import numpy as np

quadrature_points = np.array([
    [0.0915762135098,0.0915762135098],
    [0.8168475729805,0.0915762135098],
    [0.0915762135098,0.8168475729805],
    [0.1081030181681,0.4459484909160],
    [0.4459484909160,0.1081030181681],
    [0.4459484909160,0.4459484909160]]).astype(float)
quadrature_weights = np.array([0.05497587183,0.05497587183,0.05497587183,0.1116907948,0.1116907948,0.1116907948]).astype(float)