import sympy as sp

# Définition des variables symboliques
x, y = sp.symbols('x y')
epsilon = sp.symbols('epsilon', real = True, constant = True)
sp.init_printing(use_unicode=True)


# Définition de u = sin(pi*x)sin(pi*y)
u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
print(f"{u = }")
# Définition de A = [[2 + sin(2pi*x), 0], [0, 4]]
# A = sp.Matrix([[2 + sp.sin(2 * sp.pi * x), 0],
#                [0, 4]])
A = sp.Matrix([[2 + sp.sin(2 * sp.pi * x/epsilon), 0],
               [0, 4 + sp.sin(2 * sp.pi * y/epsilon)]])
print(f"{A = }")
# A = sp.Matrix([[1, 0],
#                [0, 2]])

# Calcul du gradient de u (nabla u)
grad_u = sp.Matrix([sp.diff(u, x), sp.diff(u, y)])

# Calcul de A * grad(u)
A_grad_u = A * grad_u

# Calcul de la divergence de A_grad_u
div_A_grad_u = sp.diff(A_grad_u[0], x) + sp.diff(A_grad_u[1], y)
div_A_grad_u_simplified = sp.simplify(div_A_grad_u)
sp.pprint(-div_A_grad_u_simplified)
# print(f"{-div_A_grad_u = }")
# print(sp.simplify(-div_A_grad_u))
# print(-div_A_grad_u.subs(x, 0))

# f = sp.lambdify([x, y], -div_A_grad_u)
# print(f"{f(1, 1) = }, {f(0, 1) = }")

# sp.ev
# Affichage du résultat
# sp.pprint(div_A_grad_u)