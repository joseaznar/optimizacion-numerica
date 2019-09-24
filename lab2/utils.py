import numpy
import math
import pandas as pd


def sigma(z):
    """
    Método que calcula la función sigma.
    """
    if z < -300:
        return 0
    return 1/(1 + math.exp(-z))


def lvm(beta, y, X):
    """
    Método que calcula la logverosimilitud.
    """
    X_aux = pd.DataFrame(X).copy() 
    miu = X_aux.apply(lambda x_i: sigma(beta @ x_i), axis=1)
    return -sum(y*numpy.log(miu) + (1-y)*numpy.log(numpy.ones(miu.size)-miu))

def datos(modo='entrena'):
    gid ='d932a3cf4d6bdeef36a7230fff959301'
    tail ='64b604aedff376b7757b533d1c93685ce19b2077/bcdata'
    url ='https://gist.githubusercontent.com/rodrgo/%s/raw/%s'% (gid, tail)
    df = pd.read_csv(url, sep=',')
    df = df.drop(columns=['Unnamed: 32','id'])
    var ='diagnosis'
    df[var] = df[var].apply(lambda val: 1 if val == 'M' else 0)
    df = df.apply(lambda x_i: x_i/numpy.mean(x_i))
    X_cols = [c for c in df.columns if not c is var]
    X, y = df[X_cols].to_numpy(), df[var].to_numpy()
    idx = numpy.random.permutation(X.shape[0])
    train_idx, test_idx = idx[:69], idx[69:]
    idx = train_idx if modo == 'entrena' else test_idx
    return X[idx,:], y[idx]
