from random import random
import numpy as np
from utils import supportSet, errorVector
from scipy.optimize import linprog

if __name__ == "__main__":
    # 1) Select n and m and generate A as a n by m matrix
    m = 5
    n = 2 * m

    A = np.random.normal(0, 1, (m, n))

    # 2) Select S as a percentage of m
    S = round(random() * m)

    # 3) Select a support set with a size of S uniformly at random and use it to sample a
    # vector e on T
    T = supportSet(m, S)

    e = errorVector(T)

    # 4) Select x at random, compute y and obtain x* by solving P1'
    x = np.random.normal(0, 1, (n, 1))

    y = np.matmul(A, x) + e

    ones = [[1] * m]
    A_ub = np.concatenate((A, A), axis=1)
    print(A_ub.shape)
    t_star = linprog(ones, )