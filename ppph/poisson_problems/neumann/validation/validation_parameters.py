
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

mesh_fname: str = "mesh_manager/geometries/rectangle.msh"
# Problem parameters --------------------------------------------
def diffusion_tensor_expr(x, y):
    return np.eye(2)

def u(x, y):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y)

def f(x, y):
    return (1 + 5 * np.pi ** 2) * u(x, y)