import numpy as np
import cvxpy as cp


def supportSet(m, S):
    """
    Choose the support set given a size m uniformly at random with probability 1/S.
    """
    probs = [1-1/(m-S), 1/(m-S)]
    return np.random.choice([0, 1], size=(m, 1), p=probs)


def errorVector(T):
    """
    Generate error vector given the support vector T with iid Gaussian entries}
    """
    return T * np.random.normal(0, 1, T.shape)


def solveProblem(y, A, m, n):
    """
    Solve the problem argmin |y - Ag| with the l1 norm using the l1 alternative.
    """
    t = cp.Variable(m)
    g = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum(t)), [-t <= y-A@g, y-A@g <= t])
    optimal_value = prob.solve()

    print(t.value)
    print(g.value)
    print(optimal_value)

    return g.value
