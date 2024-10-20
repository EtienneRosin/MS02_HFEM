"""
@file gauss_legendre_quadrature.py
@brief Module providing 4 points Gauss-Lobato quadrature info.
@authors Etienne Rosin
@version 0.1
@date 2024
"""
import numpy as np

s_0 = 1/3
s_1 = 1/5
s_2 = 3/5


quadrature_points = np.array([
    [s_0, s_0], 
    [s_1, s_1], 
    [s_1, s_2],
    [s_2, s_1]]).astype(float)

quadrature_weights = np.array([-9/32, 25/96, 25/96, 25/96]).astype(float)

if __name__ == '__main__':
    # print(quadrature_points)
    print(quadrature_weights)