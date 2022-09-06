import numpy as np


def softmax(Z):
        f = np.exp(Z - np.max(Z))  # shift values
        return f / f.sum(axis=0)
        """
        A = np.exp(Z) / sum(np.exp(Z))
        return A
        """

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))


def sigmoid_deriv(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def Leaky_ReLU(Z):
        Z = np.where(Z > 0, Z, Z * 0.1)
        return Z
    
def Leaky_ReLU_deriv(Z):
        Z = np.where(Z > 0, 1, 0.1)
        return Z