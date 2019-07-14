from DeepLib import initializers, activations, regularizers
import numpy as np


class Layer:
    def __init__(self, shape, activation, initializer, regularizer, batch_norm):
        self.shape = shape
        if type(activation) is str:
            self.activation = activations.get_activation(activation)
        else:
            self.activation = activation
        if type(initializer) is str:
            self.initializer = initializers.get_initializer(initializer)
        else:
            self.initializer = initializer
        if type(regularizer) is str:
            self.regularizer = regularizers.get_regularizer(regularizer)
        else:
            self.regularizer = regularizer
        self.batch_norm = batch_norm

    def forward(self, X):
        pass

    def backward(self, dA):
        pass


class Dense(Layer):
    def __init__(self, shape, activation='relu', initializer='xavier', regularizer='L2', batch_norm=False):
        super().__init__(shape, activation, initializer, regularizer, batch_norm)
        self.W = self.initializer(self.shape)
        if not self.batch_norm:
            self.b = np.zeros(shape[0], 1)
        else:
            self.mi = np.zeros(shape[0], 1)
            self.sigma = np.ones(shape[0], 1)

    def forward(self, X):
        if not self.batch_norm:
            Z = np.dot(self.W, X) + self.b
        else:
            Z = np.dot(self.W, X)
            mean = np.mean(Z, axis=1, keepdims=True)
            Z = Z-mean
            std_dev = np.sqrt(np.mean(Z**2, axis=1, keepdims=True))
            Z = Z/std_dev
            Z = Z*self.sigma + self.mi
        self.Z, self.A_prev = Z, X # caching for backprop
        A = self.activation(Z)
        return A

    def backward(self, dA):
        m = dA.shape[1]
        dZ = dA*self.activation(self.Z, deriv=True)
        self.dW = np.dot(dZ, self.A_prev)/m
        self.db = np.mean(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev
