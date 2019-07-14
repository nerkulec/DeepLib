import numpy as np


def relu(X, deriv=False):
    if not deriv:
        return np.max(X, 0)
    else:
        return X > 0


def sigmoid(X, deriv=False):
    if not deriv:
        return 1/(1+np.exp(-X))
    else:
        s = sigmoid(X)
        return s*(1-s)


def tanh(X, deriv=False):
    if not deriv:
        exp = np.exp(X)
        inv_exp = 1/exp
        return (exp-inv_exp)/(exp+inv_exp)
    else:
        t = tanh(X)
        return 1-t**2


def softmax(X, deriv=False):
    if not deriv:
        exp = np.exp(X)
        sum_exp = np.sum(exp, axis=0, keepdims=True)
        return exp/sum_exp
    else:
        raise NotImplementedError


def get_activation(name):
    return {'relu': relu,
            'sigmoid': sigmoid,
            'tanh': tanh}[name]
