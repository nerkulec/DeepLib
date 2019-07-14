import numpy as np


def xavier_initializer(shape):
    coeff = np.sqrt(2/(shape[0]+shape[1]))
    return normal_initializer(shape)*coeff


def normal_initializer(shape):
    return np.random.randn(shape[0], shape[1])


def get_initializer(name):
    return {'xavier': xavier_initializer,
            'normal': normal_initializer}[name]
