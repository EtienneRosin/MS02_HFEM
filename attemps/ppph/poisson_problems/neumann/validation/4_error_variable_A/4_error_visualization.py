import numpy as np
import matplotlib.pyplot as plt
from ppph.utils import apply_fit

style_path = "ppph/utils/graphics/custom_science.mplstyle"

def model(h, *p):
    return p[0] + p[1] * h

def model_label(popt, perr):
    return fr"{popt[0]:1.1f} + {popt[1]:1.1f}h"


p0 = [10, 10] 
fname = "ppph/poisson_problems/neumann/validation/4_error_variable_A/error_measurements.csv"
data = np.genfromtxt(fname=fname, delimiter=',', names=True)
popt_L2, pcov_L2, perr_L2 = apply_fit(model, np.log(data['h']), np.log(data['L2_error']), p0, names=["a_0", "a_1"])
popt_H1, pcov_H1, perr_H1 = apply_fit(model, np.log(data['h']), np.log(data['H1_error']), p0, names=["a_0", "a_1"])

line_props = dict(marker = "o", markersize = 4, linestyle = "--", linewidth = 1)
fit_line_props = dict(linestyle = "--", linewidth = 0.5)
scatter_props = dict(marker = "o", s = 12)

with plt.style.context(style_path):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(np.log(data['h']), np.log(data['L2_error']), label = r"$\|.\|_{L^2(\Omega)}$", c=r"#5B9276", **scatter_props)
    ax.plot(
        np.log(data['h']),
        model(np.log(data['h']), *popt_L2), 
        **fit_line_props, 
        label = fr"${model_label(popt_L2, perr_L2)}$",
        c=r"#5B9276"
        )

    ax.scatter(
        np.log(data['h']), 
        np.log(data['H1_error']), 
        label = r"$|.|_{H^1(\Omega)}$", 
        c=r"#D1453D", **scatter_props
        )
    ax.plot(
        np.log(data['h']),  
        model(np.log(data['h']), *popt_H1), 
        **fit_line_props, 
        label = fr"${model_label(popt_H1, perr_H1)}$",
        c=r"#D1453D"
        )

    ax.set(
        xlabel = r"$\log{\left(h\right)}$", 
        ylabel = r"$\log{\left(\frac{\left\| u - u_h\right\|}{\left\| u_h\right\|}\right)}$",
        )
    # axes.prop_cycle : cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])
    ax.legend(loc='upper left', mode = "expand",borderaxespad=0)
    fig.savefig("Figures/neumann_error_variable_A.pdf")
    plt.show()


