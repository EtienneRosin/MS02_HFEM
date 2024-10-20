"""
@file gauss_legendre_quadrature.py
@brief Module providing Gauss-Legendre quadrature info.
@authors Etienne Rosin
@version 0.1
@date 2024
#note source : https://www.abcm.org.br/anais/cobem/2007/pdf/COBEM2007-1614.pdf
"""
import numpy as np


a: float = 3/7
b: float = 2/7
c: float = 6/7



quadrature_points = np.array([
    [0.188995e0, 0.188995e0], 
    [0.7053418e0, 0.1279915e0], 
    [0.1279915e0, 0.7053418e0],
    [0.4776709e0, 0.4776709e0]]).astype(float)
quadrature_weights = np.array([0.1971688e0, 0.125e0, 0.125e0, 0.5283122e-1]).astype(float)

if __name__ == '__main__':
    print(quadrature_points)
    print(quadrature_weights)