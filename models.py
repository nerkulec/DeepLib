from DeepLib.layers import Dense


class DenseModel:
    def __init__(self, shape, activations=None, initializer='Xavier', batch_norm=False):
        """Initialize Dense model

        :param shape: tuple of (input_size, hidden_size_1, ..., output_size)
        :param activations: list of activations for consecutive layers
        if len(activations) == 2 it is treated as [hidden activations, output activation]
        :param initializer: DeepLib initializer
        :param batch_norm: whether to use batch_norm
        """
        if len(activations) == 2:
            activations = [activations[0]]*(len(shape)-2) + [activations[1]]
        self.shape = shape
        self.layers = []
        self.batch_norm = batch_norm
        for l in range(len(shape)-1):
            self.layers.append(Dense((shape[l+1], shape[l]), activation=activations[l], initializer=initializer, batch_norm=batch_norm))

    def forward(self, X):
        """Forward pass

        :param X: numpy array of examples stored column by column
        :return: model output
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self):
        """Backward pass - computes gradients

