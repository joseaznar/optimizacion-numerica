import pandas as pd
import numpy
import math
import time
from decimal import getcontext
from utils import lvm, datos, sigma

getcontext().prec = 3

epsilon = 1/1000
max_iter = 500

alpha = 1000
tau = 0.8
gamma = 0.5


def grad_F(X, y, beta):
    '''
    Método que calcula el gradiente de F para la función de regresión logística

        (X,y):  Datos de entrenamiento [X.shape=(m,p), y.shape=(m,)]
                X,y matrices de numpy
        beta:   Arreglo de numpy beta.shape=(p,)
    '''
    x_t = pd.DataFrame(X).T
    X_aux = pd.DataFrame(X).copy()
    z = X_aux.apply(lambda x_i: sigma(beta @ x_i), axis=1)
    z = pd.DataFrame(z-y)
    resp = (x_t @ z)

    resp = resp.iloc[:, 0] if resp.iloc[:,
                                        0].size > resp.iloc[0, :].size else resp.iloc[0, :]

    return resp


def hess_F(X, y, beta):
    """
    Método que calcula la hessiana de F.

        (X,y):  Datos de entrenamiento [X.shape=(m,p), y.shape=(m,)]
                X,y matrices de numpy
        beta:   Arreglo de numpy beta.shape=(p,)
    """
    x_t = pd.DataFrame(X).T
    X_aux = pd.DataFrame(X).copy()
    s_ii = X_aux.apply(lambda x_i: sigma(beta @ x_i), axis=1)
    s_ii = s_ii*(1-s_ii)

    S = numpy.diag(s_ii)

    resp = S @ X
    resp = x_t @ resp

    return resp


def armijo(beta_k, s_k, grad_k, y, X):
    global alpha

    lvm_val = lvm(beta_k, y, X)
    beta_aux = beta_k + alpha*s_k
    lvm_k = lvm(beta_aux, y, X)
    while lvm_k > lvm_val + gamma*alpha*(grad_k @ s_k):
        alpha = tau*alpha
        lvm_k = lvm(beta_k + alpha*s_k, y, X)

    return alpha


def busqueda_lineal(X, y):
    beta = numpy.random.normal(loc=100, size=X.shape[1])
    beta = beta/numpy.linalg.norm(beta)

    gradiente = grad_F(X, y, beta)

    i = 0
    while numpy.linalg.norm(gradiente) > epsilon and i < max_iter:
        s_k = -gradiente.copy()
        a_k = armijo(beta, s_k, gradiente, y, X)
        beta = beta + a_k*s_k
        beta = beta/numpy.linalg.norm(beta)
        gradiente = grad_F(X, y, beta)
        i += 1

    return beta, i


def newton(X, y):
    beta = numpy.random.normal(loc=1, size=X.shape[1])
    beta = beta/numpy.linalg.norm(beta)

    gradiente = grad_F(X, y, beta)

    i = 0
    while numpy.linalg.norm(gradiente) > epsilon and i < max_iter:
        hessiana = hess_F(X, y, beta)
        try:
            s_k = -numpy.linalg.solve(hessiana, gradiente.copy())
        except Exception:
            break
        a_k = armijo(beta, s_k, gradiente, y, X)
        beta = beta + a_k*s_k
        beta = beta/numpy.linalg.norm(beta)
        gradiente = grad_F(X, y, beta)
        i += 1

    return beta, i


def bfgs(X, y):
    beta_k = numpy.random.normal(loc=1, size=X.shape[1])
    beta_k = beta_k/numpy.linalg.norm(beta_k)
    beta_k2 = beta_k

    gradiente_k = grad_F(X, y, beta_k)
    gradiente_k2 = gradiente_k

    i = 0
    while numpy.linalg.norm(gradiente_k2) > epsilon and i < max_iter:
        if i == 0:
            B_k = hess_F(X, y, beta_k)
        else:
            z_k = beta_k2 - beta_k
            w_k = gradiente_k2 - gradiente_k

            if w_k @ z_k == 0:
                gradiente_k2 = gradiente_k

            C = (w_k @ w_k.T)/(w_k @ z_k)
            D = ((B_k @ z_k) @ (B_k @ z_k).T)/(z_k.T @ (B_k @ z_k))
            B_k = B_k + C - D
        
        try:
            s_k = -numpy.linalg.solve(B_k, gradiente_k2.copy())
        except Exception:
            break
        a_k = armijo(beta_k, s_k, gradiente_k2, y, X)

        beta_aux = beta_k2
        beta_k2 = beta_k2 + a_k*s_k
        beta_k = beta_aux

        gradiente_aux = gradiente_k2
        gradiente_k2 = grad_F(X, y, beta_k2)
        gradiente_k = gradiente_aux

        i += 1

    return beta_k2, i


def pred(x, beta_hat):
    """
        x:        Punto a clasificar (vector de numpy)
        betahat:  Arreglo de numpy beta.shape=(p,)
        yhat:     Vector binario de predicciones
    """
    return 1 if sigma(beta_hat @ x) >= 0.5 else 0


if __name__ == "__main__":
    for i in range(1, 10):
        X_entrena, y_entrena = datos()
        X_prueba, y_prueba = datos(modo='prueba')
        X_prueba = pd.DataFrame(X_prueba)

        t = time.time()
        beta_hat, iter = busqueda_lineal(X_entrena, y_entrena)
        elapsed = time.time() - t
        y_hat = X_prueba.apply(lambda X_i: pred(X_i, beta_hat), axis=1)
        print('\n==================== BÚSQUEDA LINEAL ({}) =============================='.format(i))
        print('Acertó el {}% de las veces en {} segundos en {} iteraciones.'.format(
            numpy.sum(y_prueba == y_hat)*100/y_hat.size, elapsed, iter))

        t = time.time()
        beta_hat, iter = newton(X_entrena, y_entrena)
        elapsed = time.time() - t
        y_hat = X_prueba.apply(lambda X_i: pred(X_i, beta_hat), axis=1)
        print('\n======================= NEWTON ({}) ===================================='.format(i))
        print('Acertó el {}% de las veces en {} segundos en {} iteraciones.'.format(
            numpy.sum(y_prueba == y_hat)*100/y_hat.size, elapsed, iter))
        
        t = time.time()
        beta_hat, iter = bfgs(X_entrena, y_entrena)
        elapsed = time.time() - t
        y_hat = X_prueba.apply(lambda X_i: pred(X_i, beta_hat), axis=1)
        print('\n======================== BFGS ({}) ====================================='.format(i))
        print('Acertó el {}% de las veces en {} segundos en {} iteraciones.'.format(
            numpy.sum(y_prueba == y_hat)*100/y_hat.size, elapsed, iter))

        print('\n***********************************************************************')
        print('***********************************************************************')
