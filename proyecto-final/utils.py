import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def supportSet(m, S):
    """
    Choose the support set given a size m uniformly at random with probability 1/S.
    """
    probs = [1 - S/m, S/m]
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

    try:
        optimal_value = prob.solve()
        return g.value
    except Exception as e:
        print('Error del solver.')
        g = np.zeros((m,1))
        return g

def scatterPlot(x, y):
    """
    Generates the scatterplot
    """
    plt.scatter(x, y)
    plt.title('Predicción')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()