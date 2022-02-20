import numpy as np
from time import perf_counter
import keras
import keras.datasets.mnist as mnist
import os, sys
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='EAO P2.')
parser.add_argument('-t', "--train", type=int,
                    help='Approaches for training. \
                    0 loading weights. \
                    1 training', default=1, choices=[0, 1])

args = parser.parse_args()

path_logs = "./data_logs"

"""
    Devuelve las n imágenes y clases de un determinado tipo.
        Caso -> 0 y 1.
"""
def get_n_labels(imgs, labels, n=3000):
    i = 0
    X, y = [], []
    while len(y) < n:
        if labels[i] == 0 or labels[i] == 1:
            X.append(imgs[i])
            y.append(labels[i])
        i+=1
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
def load_dataset(m=3000):
    (train_x, train_y),(test_x,test_y)=mnist.load_data()

    train_x, train_y = get_n_labels(train_x, train_y, 3000)
    non_zero_test = np.array(np.where(np.logical_or(test_y == 0, test_y == 1))).shape[1]

    test_x, test_y = get_n_labels( test_x, test_y, non_zero_test )

    train_x = train_x/255
    test_x = test_x/255

    train_x = np.array([np.insert(img, 0, 1, axis=0) for img in train_x.reshape(3000, -1)])#.reshape(-1)
    test_x = np.array([np.insert(img, 0, 1, axis=0) for img in test_x.reshape(non_zero_test, -1)])#.reshape(-1)

    return (train_x,train_y), (test_x,test_y)

def sigmoide(z):
    return 1/(1+np.exp(-z))


"""
    Función f sin optimizar
"""
def f_base(theta):
    izda, dcha = 0, 0
    for i in range(m):
        izda += train_y[i]*np.log(sigmoide(np.dot(theta.T, train_x[i])))
        dcha += (1-train_y[i])*np.log(1 - sigmoide(np.dot(theta.T, train_x[i]) ))

    resultado = ((-1/m)*izda) + ((-1/m)*dcha)
    return resultado

"""
    Función f optimizada
"""
def f(theta):

    izda_opt = np.dot(train_y, np.log(sigmoide(np.dot(theta.T, train_x.T) )))
    dcha_opt = np.dot(np.subtract(1, train_y), np.log(np.subtract(1, sigmoide(np.dot(theta.T, train_x.T) ))))

    resultado_opt = ((-1/m)*izda_opt) + ((-1/m)*dcha_opt)
    return resultado_opt

"""
    Función g optimizada
"""
def g(theta):
    return (-1/m)*np.dot(np.subtract(train_y, sigmoide(np.dot(theta.T, train_x.T) ) ), train_x )

"""
    Funcióng sin optimizar
"""
def g_base(theta):
    resultado = 0
    for i in range(m):
        resultado += np.dot(np.subtract(train_y[i], sigmoide( \
            np.dot(theta.T, train_x[i]))), \
            train_x[i])

    resultado = (-1/m)*resultado
    return resultado

"""
    Función para calcular un valor de alpha que satisface la condición de Armijo.
"""
def backtracking(f, g, xk, pk, alpha=1, beta=0.5, c=1e-4):
    while f(xk+alpha*pk) > f(xk) + c*alpha*( np.dot(g(xk).T, pk) ):
        alpha = alpha*beta
    return alpha


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
    Función para ajustar un modelo en función del parámetro.
"""
def train_model(mode):
    thetas = np.zeros(shape=(785), dtype=np.float32)
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

        * Si minimum = True, muestra min(back, const) iteraciones, permitiendo que se vean
        mejor los valores de backtracking.
        * Si logscale_norma = True, se usa la escala logarítmica en el eje Y para las normas.
"""
def create_plots(normas_back=None, steps_f_back=None, normas_const=None, steps_f_const=None, minimum=False, logscale_norma=True):
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

    fig, axs = plt.subplots(2, figsize=(10, 10))

    plt.subplots_adjust(top=0.9)


    axs[0].set_title("Valor de f")
    if minimum:
        n_iterations = np.arange(steps_f_back.shape[0])
        axs[0].plot(n_iterations, steps_f_back, label="backtracking")
        axs[0].plot(n_iterations, steps_f_const[:n_iterations.shape[0]], label="const")
    else:
        axs[0].plot(np.arange(steps_f_back.shape[0]), steps_f_back, label="backtracking")
        axs[0].plot(np.arange(steps_f_const.shape[0]), steps_f_const, label="const")

    axs[0].legend()

    if logscale_norma:
        axs[0].semilogy()

    axs[1].set_title("Normas")

    if minimum:
        n_iterations = np.arange(normas_back.shape[0])
        axs[1].plot(n_iterations, normas_back, label="backtracking")
        axs[1].plot(n_iterations, normas_const[:n_iterations.shape[0]], label="const")
    else:
        axs[1].plot(np.arange(normas_back.shape[0]), normas_back, label="backtracking")
        axs[1].plot(np.arange(normas_const.shape[0]), normas_const, label="const")

    axs[1].legend()
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
    print("Porcentaje correctas {:.6f}".format((correct/len(y_pred))*100.0 ))

"""
    Carga los thetas de backtracking y constantes, y realiza predicciones con ellos.
"""
def load_and_test():
    thetas_back = load_files("./data_logs/thetas_back.npz")
    thetas_const = load_files("./data_logs/thetas_const.npz")

    print('\033[92m'+"\n\n\tBACKTRACKING")
    check_predictions(thetas_back)

    print('\033[96m'+"\n\n\tCONSTANTE. ALPHA 0.01")
    check_predictions(thetas_const)

"""
    Carga un fichero en formato *.npz y devuelve un array de numpy
"""
def load_files(filename):
    return np.load(filename)['x']

m = 3000
(train_x,train_y), (test_x,test_y) = load_dataset()

def main():
    np.random.seed(0)
    if args.train == 1:
        steps_back, norma_back, thetas_back = train_model("back")
        steps_const, norma_const, thetas_const = train_model("const")
        create_plots(normas_back=norma_back, steps_f_back=steps_back, normas_const=norma_const, steps_f_const=steps_const)
    else:
        load_and_test()
        test_images(filename="./data_logs/thetas_back.npz")
        test_images(filename="./data_logs/thetas_const.npz")
        create_plots(minimum=False)

if __name__ == "__main__":
    main()
