# -*- coding: utf-8 -*-

import numpy as np
import argparse


parser = argparse.ArgumentParser(description='EAO exercises')
parser.add_argument('-v', "--verbose", type=int,
                    help='If verbosity, then -v 1', default=0)

args = parser.parse_args()

if args.verbose == 1:
    print("\nModo Debugger\n")
################ Ejercicio 1
"""
    Ptos criticos, 0, 2, 4. (0, 0) mínimo, (2, 2) punto de silla, (4, 4) mínimo.
"""

"""
    Condición suficiente de optimalidad de segundo orden: dado que los autovalores
    de la matriz Hessiana son mayores que 0, ésta es definida positiva. Por lo tanto
    se satisface la condición de optimalidad de segundo orden.
"""

def f(x):
    return 9*x[0]**2 - 2*x[0]*x[1]+x[1]**2-4*x[0]**3 + 0.5*x[0]**4

def g(x):
    return np.array([18*x[0]-2*x[1] -12*x[0]**2 + 2*x[0]**3, -2*x[0]+2*x[1]])

def H(x):
    return np.array([[-24*x[0]+6*x[0]**2+18, -2], [-2, 2]])


def eigvals(x1=0, x2=0):
    hessian = H([x1, x2])
    return np.linalg.eig(hessian)[0]

def is_DP():
    return eigvals().all() > 0

print("¿Es definida positiva la matriz hessiana? {}".format(is_DP()))

################# Ejercicio 2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

n_points = 50

x = np.linspace(-1, 5, n_points)
y = np.linspace(-3, 8, n_points)

[x1, y1] = np.meshgrid(x, y)

z1 = f([x1, y1]).reshape(n_points, n_points)

plt.title("Superficie")
ax.plot_surface(x1, y1, z1, cmap=plt.cm.coolwarm,
                       linewidth=1)
plt.show()


################# Ejercicio 3
plt.title("Curvas de nivel")
plt.contour(x1, y1, z1, 50) # 50 curvas de nivel
plt.show()

################# Ejercicio 4
def gradient_descent(x0=[0.4, -1.9], it=40):
    x = x0
    pts, steps_gr = [], []
    m, M = np.linalg.eig(H([0, 0]))[0]
    alpha = 2/(m+M)
    steps_gr.append(f(x.copy()))

    for i in range(it):
        pts.append(x.copy())
        x += -alpha*g(x)
        steps_gr.append(f(x.copy()))

    pts.append(x.copy())

    return f(x), np.array(pts), steps_gr

gradient_descent_minimum, pts_gd, steps_gr = gradient_descent(x0=[0.4, -1.9], it=40)

if args.verbose == 1:
    print("\n\nGradiente Descendente")
    print("\n", steps_gr, "\n")

print("Gradiente Descendente mínimo {}".format(gradient_descent_minimum))

plt.title("Gradiente descendente. Alpha constante")
plt.contour(x1, y1, z1, 50)
plt.plot(pts_gd[:, 0], pts_gd[:, 1], marker="o")
plt.show()


################# Ejercicio 5
def metodo_newton(x0=[0.4, -1.9], it=5):
    x = x0
    z = x0
    pts, steps_new = [], []

    steps_new.append(f(x.copy()))

    for i in range(it):
        pts.append(x.copy())
        x += np.linalg.solve(H(x), -g(x))
        #z += -np.linalg.inv(H(z))@g(z)
        steps_new.append(f(x.copy()))

    pts.append(x.copy())

    return f(x), np.array(pts), steps_new


newton_minimum, pts_newton, steps_new = metodo_newton(x0=[0.4, -1.9], it=5)

if args.verbose == 1:
    print("\n\nMétodo de Newton")
    print("\n", steps_new, "\n")

print("Newton mínimo {}".format(newton_minimum))

plt.title("Newton")
plt.contour(x1, y1, z1, 50)
plt.plot(pts_newton[:, 0], pts_newton[:, 1], marker="o")
plt.show()


##################### Ejercicio 6
from scipy.optimize import minimize

def busqueda_lineal_exacta(x0=[0.4, -1.9], it=40):
    x = x0
    pts, steps_ls = [], []

    steps_ls.append(f(x.copy()))

    for i in range(it):
        pts.append(x.copy())
        alpha = minimize(lambda alpha : f(x - alpha*g(x)), x0=1 )
        x += -alpha.x*g(x)
        steps_ls.append(f(x.copy()))

    return f(x), np.array(pts), steps_ls

linear_search_minimum, pts_ls, steps_ls = busqueda_lineal_exacta(x0=[0.4, -1.9], it=40)

if args.verbose == 1:
    print("\n\nBúsqueda lineal exacta")
    print("\n", steps_ls, "\n")

print("Búsqueda lineal exacta mínimo {}".format(linear_search_minimum))

plt.title("Descenso del Gradiente. Búsqueda exacta.")
plt.contour(x1, y1, z1, 50)
plt.plot(pts_ls[:, 0], pts_ls[:, 1], marker="o")
plt.show()

"""
    El método de Newton encuentra el mínimo, con un menor número de iteraciones
    que el resto de métodos.
    El método del gradiente con la búsqueda lineal exacta, como cabe esperar, encuentra el mínimo
    con menos iteraciones que el gradiente con el tamaño de paso constante. Sin embargo,
    hay que tener en cuenta, que la búsqueda lineal exacta es más costosa computacionalmente.
"""

plt.contour(x1, y1, z1, 50)

plt.title("Comparativa")
plt.plot(pts_gd[:, 0], pts_gd[:, 1], marker="o", label="Descenso del Gradiente. Alpha constante.")
plt.plot(pts_newton[:, 0], pts_newton[:, 1], marker="o", label="Newton.")
plt.plot(pts_ls[:, 0], pts_ls[:, 1], marker="o", label="Descenso del Gradiente. Búsqueda exacta.")
plt.legend(loc="upper right")
plt.show()

if args.verbose == 1:
    # Nota: Para poder hacer la comparativa, se ha puesto 40 iteraciones en el método de Newton.q
    newton_minimum, pts_new, steps_new = metodo_newton(x0=[0.4, -1.9], it=40)
    iterations = np.arange(30)

    plt.title("Comparativa")
    plt.xlabel("iteraciones")
    plt.ylabel("||Xk - Xminimo||")

    distancia_gradiente = [ np.linalg.norm([pts_gd[i, ...], pts_gd[-1, ...]]) for i in iterations ]
    distancia_blineal = [ np.linalg.norm([pts_ls[i, ...], pts_ls[-1, ...]]) for i in iterations ]
    distancia_newton = [ np.linalg.norm([pts_new[i, ...], pts_new[-1, ...]]) for i in iterations ]

    plt.plot(iterations, distancia_newton, marker = "*", label="Newton.")
    plt.plot(iterations, distancia_blineal, marker = "*" , label="Descenso del Gradiente. Búsqueda exacta.")
    plt.plot(iterations, distancia_gradiente, marker = "*", label="Descenso del Gradiente. Alpha constante.")
    plt.legend(loc="upper right")
    plt.show()

##################### Ejercicio 7
def newton_gradient(alpha, x=[0.4, -1.9]):
    pts_g = []
    pts_n = []
    for c_alpha in alpha:
        pg = -g(x)/np.linalg.norm(g(x))

        #hessian_inverse = np.linalg.solve(H(x), -g(x))
        #pn = hessian_inverse / np.linalg.norm(hessian_inverse)
        pn = - (np.linalg.inv(H(x)))@(g(x))  / np.linalg.norm((np.linalg.inv(H(x)))@(g(x)))


        phi_g = f(x+c_alpha*pg)
        phi_n = f(x+c_alpha*pn)

        pts_g.append(phi_g)
        pts_n.append(phi_n)

    return pts_g, pts_n


alphas = np.arange(0, 2, 0.01)
pts_g, pts_n = newton_gradient(alpha = alphas)

index_gd = pts_g.index(min(pts_g))
index_new = pts_n.index(min(pts_n))

print("Minimo en el Gradiente {} con alpha {}".format(pts_g[index_gd], alphas[index_gd]) )
print("Minimo en el Newton {} con alpha {}".format(pts_n[index_new], alphas[index_new]) )

plt.scatter([alphas[index_gd], alphas[index_new]], [pts_g[index_gd], pts_n[index_new]])

plt.title("Comparativa")
plt.xlabel("valor de alpha")
plt.ylabel("f(x)")

plt.plot(alphas, pts_g, c="blue", label="gradiente")
plt.plot(alphas, pts_n, c="red", label="newton")
plt.legend(loc="upper left")
plt.show()

"""
    a) Para una búsqueda lineal exacta sería mejor la dirección de newton dado que tiene un mínimo
    inferior al mínimo de la función con el método del gradiente.

    b) La función con el método del gradiente decrece más rápido que con el método de newton
    para el tramo inicial de valores de alpha. Al tener tamaños de paso muy pequeños nos interesa lo que
    ocurre en las iteraciones "iniciales" y por lo tanto el método del gradiente es mejor.
"""
