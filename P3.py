import numpy as np
from time import perf_counter
import keras
import keras.datasets.mnist as mnist
import os, sys
import matplotlib.pyplot as plt
import argparse

"""
#################################################################################
¿Cual de los tres metodos proporciona una mejor aproximación al mınimo global de f?

Bajo las mismas condiciones (norma > tol and iterations < maxit; tol=1x10^-3,maxit=20000)
el método mejor proporciona una aproximación al mínimo global de f, es el método de Wolfe.
Esto es así, dado que a diferencia del método de backtracking que únicamente satisface la condición de
decrecimiento sufiente (Armijo), éste también satisface la condición de curvatura para el alpha del descenso
en cada iteración. En cambio, el método constante, usa en todas las iteraciones un valor de tamaño de paso
que no varía, y que no garantiza el decrecimiento de la función.

Además de la aproximación del mínimo también pueden compararse por tiempos, y
el método de wolfe, y backtracking tardan con la actual implementación poco más de un segundo, aunque
el método de wolfe (13) converge en menos iteraciones que el de backtracking (26).

El método constante, sin embargo, converge en un total de 2211 iteraciones, con un tiempo de 49.19 segundos.
Es el algoritmo que más tarda en converger.

#################################################################################
Compara el valor de f(θ) y∥∇f(θ)∥con los valores obtenidos en el apartado anterior.

Para comparar el valor de f y la norma con el resto de métodos anteriores, se ha adjuntado una gráfica con 20 epochs
para el sgd con comparación al resto de métodos. Esto se ha hecho así debido a que, he guardado f y la norma para cada epoch
y con únicamente 2, no se visualizaba bien, además de que, de la epoch 2, a la 10 no había casi mejoría.
En cuanto a los anteriores valores, puede observarse en la gráfica, como, tanto su norma
como el valor de f, es superior al resto de métodos, esto podría deberse a que, decrece el valor del tamaño de paso en función
de cada iteración k, eso podría provocar que el tamaño de paso decrezca rápidamente, llegando a ser tan pequeño,
que a penas habría avance.

Al ejecutar el SGD, con 2 epochs, el valor de la función decrece significativamente, sin embargo, a partir de éstas,
el mínimo se estabiliza, y la función presenta oscilaciones poco significativas en torno a un punto, debido a
su naturaleza estocástica. Para comprobar lo anterior, se ha visualizado con 20 epochs.

#################################################################################
¿Que método es mas rápido?
Sin duda, el método más rápido es el SGD, ya que, ejecuta un total de epoch x k iteraciones (25330 en concreto)
en un total de 0.93 segundos. Esto explicaría en parte, porqué es uno de los métodos más utilizados actualmente.
Además de que su naturaleza estocástica dificulta el sobreajuste.


#################################################################################
Compara las predicciones los 4 metodos en el conjunto de imagenes train_x y test_x.
En cuanto a un gamma = 0.1, el método de backtracking, wolfe y descenso constante, presentan los mismos errores
tanto en train (32), como en test (2). Para problema en concreto, con estos mismos parámetros,
la elección de un método u otro, podría venir dada por su complejidad algorítmica, el número de iteraciones,
y el tiempo de cómputo. En mi opinión, me quedaría con wolfe, aunque el de backtracking es bastante competente en cuanto
a los parámetros anteriormente mencionados.

#################################################################################
Elige un valor mas grande y otro mas pequeno de γ y compara el resultado.

Con gamma = 0.01
Con este valor de gamma, se tiene para el SGD, 133 errores en train, y 12532 correctas,
con una precisión del 98.94%. En cambio, para el test, se tiene
2103 correctas, 12 erróneas, y una precisión del 99.43%.

Con wolfe para el train se tienen 20 erróneas y 12645 aciertos, con un total de 99.84% de precisión.
En test, se tienen 2113 correctas y 2 erróneas, con una precisión del 99.9%

Con back, se tiene para el train 24 errores y 12641 aciertos con una precisión del 99.81%.
En cambio para el test se tiene 2113 aciertos y 2 errores, con una precisión del 99.9%.

Finalmente para const, se tiene en train 12640 predicciones correctas, y 25 erróneas, con una precisión del 99.8%.
Y para el test, se tienen los mismos datos que con los métodos anteriores.

Con gamma = 1

Con el sgd, se tiene en train, 12548 predicciones correctas, y 117 erróneas, con una precisión del 99.07%.
Para el test, 2106 correctas, 9 fallos, con un 99.57%.

En wolfe, en el train, se tiene 44 errores y 12621 correctas, con una precisión del 99.65%.
En test se tienen 4 erróneas y 2111 correctas. Con una precisión del 99.81%.

Tanto back como const tienen los mismos errores en train y test que el método de wolfe.


Parece que hay una tendencia en la que, al aumentar el valor de gamma, aumentan ligeramente los fallos en las predicciones de test
para el método constante, wolfe y backtracking. En cambio con un gamma de 0.01, en estos mismos métodos comentados,
aunque presenta la misma cantidad de fallos en test, presenta una inferior cantidad de fallos en train.

En cambio, con el sgd, ocurre lo contrario, con un gamma de 0.01 tiene más fallos que con un gamma de 0.1,
y con un gamma de 1, tiene menos fallos que con 0.1.

#################################################################################

	Resultados:

	 constante shuffle

Tiempo total 558.8684212959997 segundos.
Iteraciones totales 25330

	Estadísticas de train

Correctas 12605
Erróneas 60
Porcentaje correctas 99.5263

	Estadísticas de test

Correctas 2110
Erróneas 5
Porcentaje correctas 99.7636

##############################################################
##############################################################
##############################################################
##############################################################

	SGD (cogiendo imagen a imagen en el bucle interno)

Tiempo total 0.936039902999255 segundos.
Iteraciones totales 25330

	Estadísticas de train

Correctas 12534
Erróneas 127
Porcentaje correctas 98.9657

	Estadísticas de test

Correctas 2104
Erróneas 10
Porcentaje correctas 99.4799

##############################################################
##############################################################
##############################################################
##############################################################

	 WOLFE


Tiempo total 1.4020892860007734 segundos.
Iteraciones totales 13

	Estadísticas de train

Correctas 12633
Erróneas 32
Porcentaje correctas 99.7473

	Estadísticas de test

Correctas 2113
Erróneas 2
Porcentaje correctas 99.9054

##############################################################
##############################################################
##############################################################
##############################################################

	 BACK


Tiempo total 1.3959984079992864 segundos.
Iteraciones totales 26

	Estadísticas de train

Correctas 12633
Erróneas 32
Porcentaje correctas 99.7473

	Estadísticas de test

Correctas 2113
Erróneas 2
Porcentaje correctas 99.9054

##############################################################
##############################################################
##############################################################
##############################################################

	 CONST


Tiempo total 48.95094340399919 segundos.
Iteraciones totales 2211

	Estadísticas de train

Correctas 12633
Erróneas 32
Porcentaje correctas 99.7473

	Estadísticas de test

Correctas 2113
Erróneas 2
Porcentaje correctas 99.9054

"""


parser = argparse.ArgumentParser(description='EAO P4.')
parser.add_argument('-t', "--train", type=int,
                    help='Approaches for training. \
                    0 loading weights. \
                    1 training', default=1, choices=[0, 1])

parser.add_argument('-v', "--verbosity", type=int,
                    help='Approaches for training. \
                    0 non-verbose. \
                    1 verbose', default=0, choices=[0, 1])

args = parser.parse_args()

path_logs = "./data_logs"
gamma = 0.1


"""
    Devuelve las n imágenes y clases de un determinado tipo.
        Caso -> 0 y 1.
"""
def get_n_labels(imgs, labels, n):
    i = 0
    X, y = [], []
    for i in range(n):
        if labels[i] == 0 or labels[i] == 1:
            X.append(imgs[i])
            y.append(labels[i])
    return np.array(X), np.array(y)

"""
    Función que:
        1) Carga los datos de train y test.
        2) Selecciona:
            2.1) Para train las 3000 primeras imágenes que contienen 1 o 0.
            2.2) Para test, todas las imágenes de su conjunto que son 1 o 0.
        3) Normaliza las imágenes en el rango 1, 0.
        4) Aplana las imágenes: T: (24x24) -> (784)
        5) Inserta un 1 al inicio de cada imagen.
"""
def load_dataset():
    (train_x, train_y),(test_x,test_y)=mnist.load_data()

	# Se obtienen aquellas cuyas clases sean 0 o 1.
    train_x, train_y = get_n_labels(train_x, train_y, len(train_y))
    test_x, test_y = get_n_labels( test_x, test_y, len(test_y) )

	# Se normalizan las imágenes
    train_x = train_x/255.
    test_x = test_x/255.

	# Se aplica una operación de "aplanamiento" en cada imagen y se inserta un 1 en la posición 0.
    train_x = np.array([np.insert(img, 0, 1, axis=0) for img in train_x.reshape(len(train_y), -1)])
    test_x = np.array([np.insert(img, 0, 1, axis=0) for img in test_x.reshape(len(test_y), -1)])

    return (train_x,train_y), (test_x,test_y)

# Se cargan las particiones de datos.
(train_x,train_y), (test_x,test_y) = load_dataset()
m = len(train_y)


def sigmoide(z):
    return 1/(1+np.exp(-z))

"""
	Función f espcífica para el SGD, y procesar una imagen a la vez.
"""
def f_sgd(theta, _train_x, _train_y):
    izda = _train_y*np.log(sigmoide(np.dot(theta.T,_train_x)))
    dcha = (1-_train_y)*np.log(1 - sigmoide(np.dot(theta.T, _train_x) ))

    resultado = ((-1/m)*izda) + (((-1/m)*dcha) + (gamma/2)*np.square(np.linalg.norm(theta)) )
    return resultado

"""
	Función g espcífica para el SGD, y procesar una imagen a la vez.
"""
def g_sgd(theta, _train_x, _train_y):
    resultado = (-1/m)*((_train_y - sigmoide(theta.T*_train_x))*_train_x)+gamma*theta

    return resultado

"""
	Función f optimizada.
"""
def f(theta):
    izda_opt = np.dot(train_y, np.log(sigmoide(np.dot(theta.T, train_x.T) )))
    dcha_opt = np.dot(np.subtract(1, train_y), np.log(np.subtract(1, sigmoide(np.dot(theta.T, train_x.T) ))))

    resultado_opt = ((-1/m)*izda_opt) + (((-1/m)*dcha_opt + (gamma/2)*np.square(np.linalg.norm(theta)) ))
    return resultado_opt



"""
	Función g optimizada.
"""
def g(theta):
    return (-1/m)*np.dot(np.subtract(train_y, sigmoide(np.dot(theta.T, train_x.T) ) ), train_x ) + gamma*theta


"""
    Función para calcular un valor de alpha que satisfaga la condición de Armijo.
"""
def backtracking(f, g, xk, pk, alpha=1, beta=0.5, c=1e-4):
    while f(xk+alpha*pk) > f(xk) + (c*alpha* np.dot(g(xk).T, pk) ):
        alpha=alpha*beta
    return alpha


"""
    Función para calcular un valor de alpha que satisfaga las condiciones de Wolfe. Esto es:
	* La condición de decrecimiento suficiente.
	* La condición de curvatura.
"""
def Wolfe(f, g, xk, pk, c1=1e-3, c2=0.9):
	a = 0
	b = 2
	continua = True
	f_xk = f(xk)
	dir_grad = np.dot(g(xk).T, pk)

	while f(xk+b*pk) <= f_xk + (c1*b*dir_grad):
	    b = 2*b

	while continua:
	    alpha = 0.5*(a+b)
	    if f(xk+alpha*pk) > f_xk + c1*alpha*dir_grad:
	        b = alpha
	    elif np.dot(g(xk+alpha*pk).T, pk) < c2*dir_grad:
	        a = alpha
	    else:
	        continua = False

	return alpha

x, y = train_x.copy(), train_y.copy()

from sklearn.utils import shuffle

"""
	Versión implementada del SGD, en el que cada iteración k, se aplica f y g, para una única imagen del dataset,
	Aplicando f y g para todas, una vez se han finalizado las k iteraciones. Esto se realiza m veces.
	El learning rate se reinicializa por cada epoch, porque si no se hiciese, llegaría un momento en el que
	es tan pequeño que no habría avance.
"""
def SGD_v2(_thetas, epochs=2, decay=0.0, lr=0.01):
	epochs=2
	decay = 0.01/epochs # lr/epochs
	thetas = _thetas.copy()
	normas, steps_f = [], []

	start_time = perf_counter()

	for epoch in range(epochs):
		train_x, train_y = shuffle(x, y, random_state=epoch)
		steps_f.append( f(thetas.copy()))
		normas.append(np.linalg.norm(g(thetas.copy())))

		for k in range(m):
			thetas += - lr*g_sgd(thetas.copy(), train_x[k], train_y[k])
			lr = lr/(1+decay*k)

		lr=0.01

	steps_f.append( f(thetas.copy()))
	normas.append(np.linalg.norm(g(thetas.copy())))
	end_time = perf_counter() - start_time

	return np.array(thetas), m*epochs, np.array(steps_f), np.array(normas), end_time

"""
	Última aproximación del SGD, en el que cada iteración k, se aplica f y g, para una única imagen del dataset,
	Aplicando f y g para todas, una vez se han finalizado las k iteraciones. El decrecimiento en cada iteración k
	depende de sí misma, ya que es el único factor variable en lr/(1+decay*k).
"""
def SGD(_thetas, epochs=2, decay=0.0, lr=0.01):
	epochs=2
	decay = 0.01/epochs # lr/epochs
	thetas = _thetas.copy()
	normas, steps_f = [], []
	paso = 0.01

	start_time = perf_counter()

	for epoch in range(epochs):
		train_x, train_y = shuffle(x, y, random_state=epoch)
		steps_f.append( f(thetas.copy()))
		normas.append(np.linalg.norm(g(thetas.copy())))

		for k in range(m):
			paso = lr/(1+decay*k)
			thetas += - paso*g_sgd(thetas.copy(), train_x[k], train_y[k])


	steps_f.append( f(thetas.copy()))
	normas.append(np.linalg.norm(g(thetas.copy())))
	end_time = perf_counter() - start_time
	#plt.title(f"SGD: {epochs} epochs")
	#plt.plot(np.arange(len(steps_f)), steps_f)
	#plt.show()
	return np.array(thetas), m*epochs, np.array(steps_f), np.array(normas), end_time

"""
	Primer acercamiento al SGD, ya que, entendí que debía ser así el SGD, pero,
	me di cuenta de que no deja de ser una versión con shuffle epochs veces del dataset,
	pero que se iba a quedar "estancado" debido a su pequeño lr.
"""
def const_shuffle(_thetas, epochs=2, decay=0.0, lr=0.01):
    decay = 0.01/epochs
    thetas = _thetas.copy()
    normas, steps_f = [], []
    start_time = perf_counter()
    for epoch in range(epochs):
        train_x, train_y = shuffle(x, y, random_state=epoch)

        for k in range(m):
            normas.append(np.linalg.norm(g(thetas.copy())))
            steps_f.append( f(thetas.copy() ))

            thetas += - lr*g(thetas.copy())
            lr = lr/(1+decay*k)

    end_time = perf_counter() - start_time

    return np.array(thetas), m*epochs, np.array(steps_f), np.array(normas), end_time

"""
    Función del descenso más rápido para minimizar una función f dada por parámetro.
"""
def descenso(f, g, x0, paso="back", tol=1e-3, maxit=20000):
    start_time = perf_counter()

    iterations = 0
    xk = x0.copy()
    steps_f, normas, points = [], [], []

    norma = np.linalg.norm(g(xk.copy()))

    while norma > tol and iterations < maxit:
        steps_f.append(f(xk.copy()))
        normas.append(norma)

        if paso == "back":
            alpha = backtracking(f, g, xk=xk, pk=-g(xk.copy()), alpha=1, beta=0.5, c=1e-4)

        elif paso == "const":
            alpha = 0.01

        elif paso == "Wolfe":
            alpha = Wolfe(f=f, g=g, xk=xk, pk=-g(xk.copy()), c1=1e-3, c2=0.9)

        xk += -alpha*g(xk.copy())
        iterations += 1
        norma = np.linalg.norm(g(xk.copy()))

    end_time = perf_counter() - start_time

    if iterations == maxit:
        print("\nSe ha excedido el numero de iteraciones\n")

    steps_f.append(f(xk.copy()))
    normas.append(norma)

    return xk, iterations, np.array(steps_f), np.array(normas), end_time

"""

"""
def train_model(mode, optimizador="descenso"):
    thetas = np.zeros(shape=(785), dtype=np.float32)
    if mode == "sgd":
        thetas, iterations, steps, norma, time = SGD(thetas, epochs=2, decay=0.0, lr=0.01)
    elif mode in ["const", "back", "Wolfe"]:
        thetas, iterations, steps, norma, time = descenso(f=f, g=g, x0=thetas, paso = mode)

    if not os.path.isdir(path_logs):
        os.mkdir(path_logs)
    np.savez(os.path.join(path_logs, f"thetas_{mode}.npz"), x=thetas)
    np.savez(os.path.join(path_logs, f"steps_f_{mode}.npz"), x=steps)
    np.savez(os.path.join(path_logs, f"norma_{mode}.npz"), x=norma)

    # PREDICTIONS
    print(f"\n\n\t {mode.upper()} \n\n")
    print(f"Tiempo total {time} segundos.")
    print(f"Iteraciones totales {iterations}")
    check_predictions(thetas)
    test_images(mode, thetas)

    return steps, norma, thetas

"""
    Función auxiliar para realizar una predicción sobre 9 imágenes elegidas aleatoriamente
    del conjunto de test, y plotearlas junto con la clase predicha.
"""
def test_images(mode=None, thetas=None, filename=None):
    indexes = np.random.randint(low=0, high=test_y.shape[0], size=(9,))
    fig = plt.figure(figsize=(8, 8))

    if filename is not None:
        thetas = load_files(filename)
        mode = filename.split("/")[-1][7:-4]

    ax = []
    fig.suptitle(f"Predicciones en test con modo : {mode.upper()}", fontsize=16)

    for i, index in enumerate(indexes):

        y_pred = make_predictions(thetas, test_x[index])

        ax.append( fig.add_subplot(3, 3, i+1) )
        ax[-1].set_title(str(y_pred))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.imshow(test_x[index][1:].reshape(28, 28))

    plt.show()

"""
    Función auxiliar para hacer las predicciones.
    xtest es la imagen 'i' del conjunto de datos de test.
"""
def make_predictions(thetas, xtest):
    y_preds = sigmoide(np.dot(thetas.T, xtest))
    y_preds = 1 if y_preds >= 0.5 else 0

    return y_preds

"""
    Recibe como parámetros los thetas.
    Hace predicciones en train y test con estos thetas.
"""
def check_predictions(thetas):
    y_pred_test, y_pred_train = [], []

    for i in range(len(test_y)):
        y_pred_test.append(make_predictions(thetas, test_x[i]))

    for i in range(len(train_y)):
        y_pred_train.append(make_predictions(thetas, train_x[i]))

    print(f"\n\tEstadísticas de train\n")
    log_predictions(y_pred_train, train_y)
    print(f"\n\tEstadísticas de test\n")
    log_predictions(y_pred_test, test_y)


"""
    Función para plotear con las escala logaritmica en el eje Y las normas,
    y el valor de f de ambos modelos.
"""
def create_plots(normas_back=None, steps_f_back=None, \
                normas_const=None, steps_f_const=None, \
                normas_wolfe=None, steps_f_wolfe=None, \
                normas_sgd=None, steps_f_sgd=None, \
				minimum=False, logscale_norma=True,
				include_sgd=False):
	# Paso backtracking
	if normas_back is None:
	    normas_back = load_files(filename="./data_logs/norma_back.npz")

	if steps_f_back is None:
	    steps_f_back = load_files(filename="./data_logs/steps_f_back.npz")

	# Paso constante
	if normas_const is None:
	    normas_const = load_files(filename="./data_logs/norma_const.npz")

	if steps_f_const is None:
	    steps_f_const = load_files(filename="./data_logs/steps_f_const.npz")

	# Paso wolfe
	if normas_wolfe is None:
	    normas_wolfe = load_files(filename="./data_logs/norma_Wolfe.npz")

	if steps_f_wolfe is None:
	    steps_f_wolfe = load_files(filename="./data_logs/steps_f_Wolfe.npz")

	# SGD
	if include_sgd:
		if normas_sgd is None:
			normas_sgd = load_files(filename="./data_logs/norma_sgd.npz")

		if steps_f_sgd is None:
			steps_f_sgd = load_files(filename="./data_logs/steps_f_sgd.npz")

	fig, axs = plt.subplots(2, figsize=(10, 10))
	plt.subplots_adjust(top=0.9)


	axs[0].set_title("Valor de f")

	if minimum:
		its = min(steps_f_back.shape[0], steps_f_const.shape[0], steps_f_wolfe.shape[0], steps_f_sgd.shape[0] if include_sgd else steps_f_wolfe.shape[0])
		n_iterations = np.arange(its)
		axs[0].plot(n_iterations, steps_f_back[:n_iterations.shape[0]], label="backtracking")
		axs[0].plot(n_iterations, steps_f_wolfe[:n_iterations.shape[0]], label="wolfe")
		axs[0].plot(n_iterations, steps_f_const[:n_iterations.shape[0]], label="const")
		if include_sgd:
			axs[0].plot(n_iterations, steps_f_sgd[:n_iterations.shape[0]], label="sgd")
	else:
		axs[0].plot(np.arange(steps_f_back.shape[0]), steps_f_back, label="backtracking")
		axs[0].plot(np.arange(steps_f_wolfe.shape[0]), steps_f_wolfe, label="wolfe")
		axs[0].plot(np.arange(steps_f_const.shape[0]), steps_f_const, label="const")
		if include_sgd:
			axs[0].plot(np.arange(steps_f_sgd.shape[0]), steps_f_sgd, label="sgd")

	axs[0].legend()
	axs[0].semilogy()


	axs[1].set_title("Normas")

	if minimum:
		its = min(normas_back.shape[0], normas_const.shape[0], normas_wolfe.shape[0], normas_sgd.shape[0] if include_sgd else normas_wolfe.shape[0])
		n_iterations = np.arange(its)
		axs[1].plot(n_iterations, normas_back[:n_iterations.shape[0]], label="backtracking")
		axs[1].plot(n_iterations, normas_wolfe[:n_iterations.shape[0]], label="wolfe")
		axs[1].plot(n_iterations, normas_const[:n_iterations.shape[0]], label="const")
		if include_sgd:
			axs[1].plot(n_iterations, normas_sgd[:n_iterations.shape[0]], label="sgd")
	else:
		axs[1].plot(np.arange(normas_back.shape[0]), normas_back, label="backtracking")
		axs[1].plot(np.arange(normas_wolfe.shape[0]), normas_wolfe, label="wolfe")
		axs[1].plot(np.arange(normas_const.shape[0]), normas_const, label="const")
		if include_sgd:
			axs[1].plot(np.arange(normas_sgd.shape[0]), normas_sgd, label="sgd")

	axs[1].legend()

	if logscale_norma:
		axs[1].semilogy()

	plt.savefig("Compara.pdf")

	plt.show()

"""
    Función auxiliar para crear un log con los resultados
    tras entrenar el modelo y obtener los y_hats.
"""
def log_predictions(y_pred, true_y):
    correct = 0
    for i, pred in enumerate(y_pred):
        if pred == true_y[i]:
            correct += 1

    print("Correctas {}".format(correct))
    print("Erróneas {}".format(len(y_pred)-correct))
    print("Porcentaje correctas {:.6f}".format((correct/len(y_pred))*100.0) )

"""
    Carga los thetas de backtracking y constantes, y realiza predicciones con ellos.
"""
def load_and_test():
    thetas_back = load_files("./data_logs/thetas_back.npz")
    thetas_const = load_files("./data_logs/thetas_const.npz")
    thetas_wolfe = load_files("./data_logs/thetas_Wolfe.npz")
    thetas_sgd = load_files("./data_logs/thetas_sgd.npz")

    print('\033[92m'+"\n\n\tBACKTRACKING")
    check_predictions(thetas_back)

    print('\033[95m'+"\n\n\tWolfe")
    check_predictions(thetas_wolfe)

    print('\033[96m'+"\n\n\tCONSTANTE. ALPHA 0.01")
    check_predictions(thetas_const)

    print('\033[93m'+"\n\n\tSGD")
    check_predictions(thetas_sgd)

"""
    Carga un fichero en formato *.npz y devuelve un array de numpy
"""
def load_files(filename):
    return np.load(filename)['x']


def main():
	np.random.seed(0)
	if args.train == 1:
		steps_f_sgd, normas_sgd, thetas_sgd = train_model("sgd")
		steps_wolfe, norma_wolfe, thetas_wolfe = train_model("Wolfe")
		steps_back, norma_back, thetas_back = train_model("back")
		steps_const, norma_const, thetas_const = train_model("const")
		create_plots(normas_back=norma_back, steps_f_back=steps_back, \
		        normas_const=norma_const, steps_f_const=steps_const, \
		        normas_wolfe=norma_wolfe, steps_f_wolfe=steps_wolfe, \
				#normas_sgd=normas_sgd, steps_f_sgd=steps_f_sgd,\
				include_sgd=False)


		if args.verbosity:
			print(f"Minimo con Wolfe {steps_wolfe[-1]}. ||∇f(θ)|| {norma_wolfe}")
			print(f"Minimo con back {steps_back[-1]}. ||∇f(θ)|| {norma_back}")
			print(f"Minimo constante α = 0.1 {steps_const[-1]}. ||∇f(θ)|| {norma_const}")
			print(f"SGD {steps_f_sgd[-1]}. ||∇f(θ)|| {normas_sgd}")
	else:
		load_and_test()
		test_images(filename="./data_logs/thetas_sgd.npz")
		test_images(filename="./data_logs/thetas_back.npz")
		test_images(filename="./data_logs/thetas_const.npz")
		test_images(filename="./data_logs/thetas_Wolfe.npz")
		create_plots()



if __name__ == "__main__":
    main()
