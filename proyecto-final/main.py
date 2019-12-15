from random import random
import numpy as np
from utils import supportSet, errorVector, solveProblem, scatterPlot
from scipy.optimize import linprog

if __name__ == "__main__":
    acc = {}
    xAxis = []
    yAxis = []
    max_value = 20
    for i in range(1, max_value):
        # 1) Select n and m and generate A as a n by m matrix
        n = 512
        m = 2 * n

        A = np.random.normal(0, 1/10, (m, n))

        # 2) Select S as a percentage of m
        S = (1 / max_value) * i * m

        # 3) Select a support set with a size of S uniformly at random and use it to sample a
        # vector e on T
        T = supportSet(m, S)

        e = errorVector(T)

        # 4) Select x at random, compute y and obtain x* by solving P1'
        x = np.random.normal(0, 1, (n, 1))

        y = np.matmul(A, x) + e

        x_star = solveProblem(y[:, 0], A, m, n)
        # x_star = x_star[..., np.newaxis]
        y_star = np.matmul(A, x_star)

        acc[np.count_nonzero(e)/m] = {
            'y': np.count_nonzero((y[:, 0]-y_star) < 0.0001)/m,
            'x': np.count_nonzero(abs(x[:, 0]-x_star) < 0.0001)/n,
            'normX': np.linalg.norm(x[:, 0]-x_star)
        }

        yAxis.append(acc[np.count_nonzero(e)/m]['x'])
        xAxis.append(np.count_nonzero(e)/m)

        print(i)
        ''' print('No zeros: ', np.count_nonzero(e))
        print('S: ', S)
        print('S ratio: ', S/m)
        print('norm(x-x*)', np.linalg.norm(x[:,0]-x_star))
        print('norm(e)', np.linalg.norm(e))
        print('norm(y-y*)', np.linalg.norm(y[:,0]-y_star)) '''

    print(acc)
    scatterPlot(xAxis, yAxis)
