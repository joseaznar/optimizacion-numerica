import numpy as np
import math

def achlioptas(M: int):
    """
    Construye la matriz de proyección de Achlioptas.
    :param n: tamaño de la matriz a generar. 
    """
    K = (1.8/(0.2*0.2))*math.log(M)
    K = round(K)
    mat = np.random.choice([1, -1, 0], size=(K, M), p=[1/6, 1/6, 2/3])

    return mat
