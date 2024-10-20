"""
@file gauss_legendre_quadrature.py
@brief Module providing Gauss-Legendre quadrature info.
@authors Etienne Rosin
@version 0.1
@date 2024
"""
import numpy as np

s_0 = 1/6
s_1 = 2/3
ω_0 = 1/6

quadrature_points = np.array([[s_0, s_0], [s_1, s_0], [s_0, s_1]]).astype(float)
quadrature_weights = np.array([ω_0 for _ in range(3)])

if __name__ == '__main__':
    print(ω_0)
    print(quadrature_points)
    print(quadrature_weights)