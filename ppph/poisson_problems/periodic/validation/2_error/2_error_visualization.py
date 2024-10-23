import numpy as np
import matplotlib.pyplot as plt

from ppph.utils import apply_fit

import scienceplots
plt.style.use('science') 


def modele(h, *p):
    return p[0] + p[1] * h

def model_label(popt, perr):
    # return fr"{popt[0]:1.1f}(\pm {perr[0]:1.1f}) + {popt[1]:1.1f}(\pm {perr[1]:.1f})h + {popt[2]:1.1f}(\pm {perr[2]:.1f})h^2 + O(h^3)"
    return fr"{popt[0]:1.1f} + {popt[1]:1.1f}h"


p0 = [10, 10] 


fname = "ppph/poisson_problems/periodic/validation/2_error/error_measurements.csv"
data = np.genfromtxt(fname=fname, delimiter=',', names=True)
# "h", "L2_error", "H1_error"]
popt_L2, pcov_L2, perr_L2 = apply_fit(modele, np.log(data['h']), np.log(data['L2_error']), p0, names=["a_0", "a_1"])
popt_H1, pcov_H1, perr_H1 = apply_fit(modele, np.log(data['h']), np.log(data['H1_error']), p0, names=["a_0", "a_1"])

line_props = dict(marker = "o", markersize = 4, linestyle = "--", linewidth = 1)
fit_line_props = dict(marker = "o", markersize = 1.5, linestyle = "--", linewidth = 0.5)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(np.log(data['h']), np.log(data['L2_error']), **line_props, label = r"$\|.\|_{L^2(\Omega)}$")
ax.plot(np.log(data['h']),  modele(np.log(data['h']), *popt_L2), **fit_line_props, label = fr"${model_label(popt_L2, perr_L2)}$")

ax.plot(np.log(data['h']), np.log(data['H1_error']), **line_props, label = r"$|.|_{H^1(\Omega)}$")
ax.plot(np.log(data['h']),  modele(np.log(data['h']), *popt_H1), **fit_line_props, label = fr"${model_label(popt_H1, perr_H1)}$")

ax.set(
    xlabel = r"$\log{\left(h\right)}$", 
    ylabel = r"$\log{\left(\frac{\left\| u - u_h\right\|}{\left\| u_h\right\|}\right)}$",
    # xscale = "log",
    # yscale = "log"
    )

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
ax.legend(loc='upper left', mode = "expand",borderaxespad=0)
# ax.legend(bbox_to_anchor=(0, 0.9, 1, 1), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol = 2)
fig.savefig(fname="Figures/perdiodic_error.pdf")
# print(f"{np.exp(-0.25)}")
# print(f"{np.exp(-1.5)}")
plt.show()


