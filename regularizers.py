import numpy as np


class L2:
    def __init__(self, lambd):
        self.lambd = lambd


def get_regularizer(name):
    return {'L2': L2(0.01)}[name]

