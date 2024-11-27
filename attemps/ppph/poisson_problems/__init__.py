r""" 
A library defining 3 types of Poisson problems.

The considered Poisson problem is the following:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega.

The library provides three variations of this problem based on different boundary conditions:

1. :mod:`PoissonDirichletProblem`
2. :mod:`PoissonNeumannProblem`
3. :mod:`PoissonPeriodicProblem`

---

:mod:`PoissonDirichletProblem`
------------------------------

This module defines a Poisson problem with Dirichlet boundary conditions. The problem is formulated as:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
   & u = 0, \quad \text{on } \partial\Omega.
   \end{aligned}


---

:mod:`PoissonNeumannProblem`
----------------------------

This module defines a Poisson problem with Neumann boundary conditions. The problem is expressed as:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
   & \boldsymbol{A} \nabla u \cdot \boldsymbol{n} = 0, \quad \text{on } \partial\Omega.
   \end{aligned}


---

:mod:`PoissonPeriodicProblem`
-----------------------------

This module defines a Poisson problem with periodic boundary conditions. The problem is expressed as:

.. math::
   \text{Find } u \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
   & u\vert_{x = 0} &= u\vert_{x = L} \\
   & u\vert_{y = 0} &= u\vert_{y = L} \\
   & \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_x\vert_{x = 0} = \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_x\vert_{x = L} \\ 
   & \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_y\vert_{y = 0} &= \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_y\vert_{y = L}
   \end{aligned}

---

Modules
-------

The library contains the following submodules:
- :mod:`neumann`: Defines the `PoissonNeumannProblem` class.
- :mod:`dirichlet`: Defines the `PoissonDirichletProblem` class.
- :mod:`periodic`: Defines the `PoissonPeriodicProblem` class.

References
----------
.. [1] [Finite Element Analysis](https://perso.ensta-paris.fr/~fliss/teaching-an201.html)
.. [2] [Project](https://perso.ensta-paris.fr/~fliss/ressources/Homogeneisation/TP.pdf)
"""

from .neumann import PoissonNeumannProblem
from .dirichlet import PoissonDirichletProblem
from .periodic import PoissonPeriodicProblem