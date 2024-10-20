import numpy as np
import scipy.optimize as opt

def apply_fit(modele: callable, lst_x: np.ndarray, lst_y: np.ndarray, p0: np.ndarray, names: list[str], verbose: bool = False):
    popt, pcov = opt.curve_fit(
        f = modele, 
        xdata = lst_x,
        ydata = lst_y, 
        absolute_sigma = False, 
        p0 = p0)
    perr = np.sqrt(np.diag(pcov))

    if verbose:
        print(f"Matrice de covariance : \n {pcov}")
        print("Valeurs minimisantes :")
        for i in range(len(names)):
            print(f"{names[i]} = {popt[i]:.1} ± {perr[i]:.1}")
    return popt, pcov, perr