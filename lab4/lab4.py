import numpy as np
import random
import cvxopt
import scipy.optimize as so


def bp (X, y):
    '''
    (X,y): Matriz de datos y vector de respuesta [X.shape=(m,p), y.shape=(m,)]
    (betahat): Soluci\'on [betahat.shape=(1,)]
    '''
    unos = np.ones(X.shape[1])
    zeros = np.zeros([X.shape[0], X.shape[1]])
    zeros_2 = np.zeros(len(X))
    betahat = so.linprog(c=unos, A_eq=X, b_eq=y)

    return betahat


def datos(m, p, k):
    np.random.seed(1111)
    random.seed(1111)
    X = np.random.normal(0, 1, (m, p))
    beta = np.random.normal(0, 1, (p, 1))
    beta[random.sample(range(0, p), k=p-k)] = 0
    y = np.matmul(X, beta)

    return X, y.squeeze(), beta.squeeze()