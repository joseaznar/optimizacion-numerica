from cvxopt import matrix, solvers
import numpy as np
import time
from utils import achlioptas
import pprint

N = 500
M = 100000
t_normal = []
t_red = []
for i in range(0, 10):
    t = time.time()

    # ponemos G y h para agregar la restricción de x no negativa
    G = matrix(-1 * np.identity(N))
    h = matrix(np.zeros(N))

    # construimos A, x no negativa, c y B
    A = matrix(np.random.normal(0, 1, (M, N)))
    x = matrix(abs(np.random.normal(0, 1, N)))
    c = matrix(np.random.normal(0, 1, N))
    b = matrix(np.matmul(A, x))

    sol = solvers.lp(c, G=G, h=h, A=A, b=b, solver='glpk')

    t0 = -(t-time.time())

    t = time.time()

    P = achlioptas(M)
    A = matrix(np.matmul(P, A))
    b = matrix(np.matmul(P, b))

    sol = solvers.lp(c, G=A, h=b, A=A, b=b, solver='glpk')

    t1 = -(t-time.time())

    print('Tiempo 1: %s segundos' % (t0))
    print('Tiempo 2: %s segundos' % (t1))
    t_normal.append(t0)
    t_red.append(t1)

pp = pprint.PrettyPrinter(depth=1)
pp.pprint(t_normal)
pp.pprint(t_red)
