import numpy as np


def supportSet(m, S):
    """
    Choose the support set given a size m uniformly at random with probability 1/S.
    """
    probs = [1-1/(m-S), 1/(m-S)]
    return np.random.choice([0, 1], size=(m,1), p=probs)

def errorVector(T):
    """
    Generate error vector given the support vector T with iid Gaussian entries}
    """
    return T * np.random.normal(0, 1, T.shape)