r"""
HFEM (homogenization Finite Elements Method) the library of the project defined by [2].

It deals with the periodic homogeneization of the following diffusion (Poisson) problem :

.. math::
   \text{Find } u_varepsilon \in H^1(\Omega) \text{ such that: } \\
   \begin{aligned}
   & - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
   & u = 0, \quad \text{on } \partial\Omega.
   \end{aligned}
   
It is divided in 2 sub-libraries :
1. :mod:`poisson_problems`
    Define Poisson problems for 3 types of boundary conditions that are useful for the periodic homogeneization:
    - Neumann condition
        .. math::
            \text{Find } u \in H^1(\Omega) \text{ such that: } \\
            \begin{aligned}
            & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
            & u\vert_{x = 0} &= u\vert_{x = L} \\
            & u\vert_{y = 0} &= u\vert_{y = L} \\
            & \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_x\vert_{x = 0} = \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_x\vert_{x = L} \\ 
            & \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_y\vert_{y = 0} &= \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_y\vert_{y = L}
            \end{aligned}
    - Dirichlet condition
        .. math::
            \text{Find } u \in H^1(\Omega) \text{ such that: } \\
            \begin{aligned}
            & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
            & u = 0, \quad \text{on } \partial\Omega.
            \end{aligned}
    - Periodic conditions
        .. math::
            \text{Find } u \in H^1(\Omega) \text{ such that: } \\
            \begin{aligned}
            & u - \nabla \cdot (\boldsymbol{A} \nabla u) = f, \quad \text{in } \Omega, \\
            & u\vert_{x = 0} &= u\vert_{x = L} \\
            & u\vert_{y = 0} &= u\vert_{y = L} \\
            & \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_x\vert_{x = 0} = \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_x\vert_{x = L} \\ 
            & \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_y\vert_{y = 0} &= \boldsymbol{A}\nabla u \cdot \boldsymbol{e}_y\vert_{y = L}
            \end{aligned}
2. :mod:`periodic_microstruct_poisson_problem`
    Define the final problem of the Poisson problem in a periodic microstructured material.

References
----------
.. [1] [Finite Element Analysis](https://perso.ensta-paris.fr/~fliss/teaching-an201.html)
.. [2] [Project](https://perso.ensta-paris.fr/~fliss/ressources/Homogeneisation/TP.pdf)
"""