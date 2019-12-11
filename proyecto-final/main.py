from random import random
import numpy as np
from utils import supportSet, errorVector, solveProblem
from scipy.optimize import linprog

if __name__ == "__main__":
    # 1) Select n and m and generate A as a n by m matrix
    m = 5
    n = 2 * m

    A = np.random.normal(0, 1, (m, n))

    # 2) Select S as a percentage of m
    S = round(random() * m)
    S = 1

    # 3) Select a support set with a size of S uniformly at random and use it to sample a
    # vector e on T
    T = supportSet(m, S)

    e = errorVector(T)

    # 4) Select x at random, compute y and obtain x* by solving P1'
    x = np.random.normal(0, 1, (n, 1))

    y = np.matmul(A, x) + e

    x_star = solveProblem(y[0], A, m, n);
    print(A.shape)
    print(x_star.shape)
    print(e.shape)
    y_star = np.matmul(A, x_star) + e
    print(x)
    print(x_star)
    print(y)
    print(y_star)