import numpy as np


def MSE():
    pass


def MAE():
    pass


def cross_entropy(Y, Y_hat, deriv=False):
    m = Y.shape[1]
    if not deriv:
        cost = np.sum(Y_hat*np.log(Y)+(1-Y_hat)*np.log(1-Y), axis=1)/m
        return cost


def cross_entropy_with_logits():
    pass


def get_loss(name):
    return {'MSE': MSE,
            'mse': MSE,
            'MAE': MAE,
            'mae': MSE,
            'cross_entropy': cross_entropy,
            'cross_entropy_with_logits': cross_entropy_with_logits}[name]
