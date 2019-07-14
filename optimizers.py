import numpy as np


class Optimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
        self.iteration = 1

    def _update(self):
        pass

    def update(self):
        self._update()
        self.iteration += 1


class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model, lr)

    def _update(self):
        for layer in self.model.layers:
            layer.W = layer.W - self.lr*layer.dW

            if not layer.batch_norm:
                layer.b = layer.b - self.lr*layer.dB
            else:
                layer.mi = layer.mi - self.lr*layer.dmi
                layer.sigma = layer.sigma - self.lr*layer.dsigma


class Momentum(Optimizer):
    def __init__(self, model, lr=0.01, beta=0.9):
        """Momentum optimizer

        :param model: DeepLib Model
        :param lr: learning rate
        :param beta: "momentum" EWMA parameter
        """
        super().__init__(model, lr)
        self.beta = beta
        for layer in model.layers:
            layer.VdW = np.zeros_like(layer.W)
            if not layer.batch_norm:
                layer.Vdb = np.zeros_like(layer.b)
            else:
                layer.Vdmi = np.zeros_like(layer.mi)
                layer.Vdsigma = np.ones_like(layer.sigma)

    def _update(self):
        for layer in self.model.layers:
            layer.VdW = self.beta*layer.VdW + (1-self.beta)*layer.dW
            VdW_corrected = layer.VdW / (1-self.beta**self.iteration)
            layer.W = layer.W - self.lr*VdW_corrected

            if not layer.batch_norm:
                layer.Vdb = self.beta*layer.Vdb + (1-self.beta)*layer.db
                Vdb_corrected = layer.Vdb / (1-self.beta**self.iteration)
                layer.b = layer.b - self.lr*Vdb_corrected
            else:
                layer.Vdmi = self.beta*layer.Vdmi + (1-self.beta)*layer.dmi
                Vdmi_corrected = layer.Vdmi / (1-self.beta**self.iteration)
                layer.mi = layer.mi - self.lr*Vdmi_corrected

                layer.Vdsigma = self.beta*layer.Vdsigma + (1-self.beta)*layer.dsigma
                Vdsigma_corrected = layer.Vdsigma / (1-self.beta**self.iteration)
                layer.sigma = layer.sigma - self.lr*Vdsigma_corrected


class RMSProp(Optimizer):
    def __init__(self, model, lr=0.01, beta=0.999, epsilon=0.00000001):
        """RMSProp optimizer

        :param model: DeepLib Model
        :param lr: learning rate
        :param beta: "oscilation" EWMA parameter
        :param epsilon: for numerical stability
        """
        super().__init__(model, lr)
        self.beta = beta
        self.epsilon = epsilon
        for layer in model.layers:
            layer.SdW = np.zeros_like(layer.W)

            if not layer.batch_norm:
                layer.Sdb = np.zeros_like(layer.b)
            else:
                layer.Sdmi = np.zeros_like(layer.mi)
                layer.Sdsigma = np.ones_like(layer.sigma)

    def _update(self):
        for layer in self.model.layers:
            layer.SdW = self.beta*layer.SdW + (1-self.beta)*layer.dW**2
            SdW_corrected = layer.SdW / (1-self.beta**self.iteration)
            layer.W = layer.W - self.lr*layer.dW / (np.sqrt(SdW_corrected)+self.epsilon)

            if not layer.batch_norm:
                layer.Sdb = self.beta*layer.Sdb + (1-self.beta)*layer.db**2
                Sdb_corrected = layer.Sdb / (1-self.beta**self.iteration)
                layer.b = layer.b - self.lr*layer.db / (np.sqrt(Sdb_corrected)+self.epsilon)
            else:
                layer.Sdmi = self.beta2 * layer.Sdmi + (1 - self.beta2) * layer.dmi ** 2
                Sdmi_corrected = layer.Sdmi / (1 - self.beta2 ** self.iteration)
                layer.mi = layer.mi - self.lr * layer.dmi / (np.sqrt(Sdmi_corrected) + self.epsilon)

                layer.Sdsigma = self.beta2 * layer.Sdsigma + (1 - self.beta2) * layer.dsigma ** 2
                Sdsigma_corrected = layer.Sdsigma / (1 - self.beta2 ** self.iteration)
                layer.sigma = layer.sigma - self.lr * layer.dsigma / (np.sqrt(Sdsigma_corrected) + self.epsilon)


class Adam(Optimizer):
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999, epsilon=0.00000001):
        """Adam optimizer as described in arXiv:1412.6980

        :param model: DeepLib Model
        :param lr: learning rate
        :param beta1: "momentum" EWMA parameter
        :param beta2: "oscilation" EWMA parameter
        :param epsilon: for numerical stability
        """
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        for layer in model.layers:
            layer.VdW = np.zeros_like(layer.W)
            layer.SdW = np.zeros_like(layer.W)
            if not layer.batch_norm:
                layer.Vdb = np.zeros_like(layer.b)
                layer.Sdb = np.zeros_like(layer.b)
            else:
                layer.Vdmi = np.zeros_like(layer.mi)
                layer.Sdmi = np.zeros_like(layer.mi)
                layer.Vdsigma = np.ones_like(layer.sigma)
                layer.Sdsigma = np.ones_like(layer.sigma)


    def _update(self):
        for layer in self.model.layers:
            layer.VdW = self.beta1 * layer.VdW + (1 - self.beta1) * layer.dW
            layer.SdW = self.beta2*layer.SdW + (1-self.beta2)*layer.dW**2
            VdW_corrected = layer.VdW / (1 - self.beta1 ** self.iteration)
            SdW_corrected = layer.SdW / (1-self.beta2**self.iteration)
            layer.W = layer.W - self.lr*VdW_corrected / (np.sqrt(SdW_corrected)+self.epsilon)

            if not layer.batch_norm:
                layer.Vdb = self.beta2 * layer.Vdb + (1 - self.beta2) * layer.db
                layer.Sdb = self.beta2*layer.Sdb + (1-self.beta2)*layer.db**2
                Vdb_corrected = layer.Vdb / (1 - self.beta1 ** self.iteration)
                Sdb_corrected = layer.Sdb / (1-self.beta2**self.iteration)
                layer.b = layer.b - self.lr*Vdb_corrected / (np.sqrt(Sdb_corrected)+self.epsilon)
            else:
                layer.Vdmi = self.beta1 * layer.Vdmi + (1 - self.beta1) * layer.dmi
                layer.Sdmi = self.beta2 * layer.Sdmi + (1 - self.beta2) * layer.dmi ** 2
                Vdmi_corrected = layer.Vdmi / (1 - self.beta1 ** self.iteration)
                Sdmi_corrected = layer.Sdmi / (1 - self.beta2 ** self.iteration)
                layer.mi = layer.mi - self.lr * Vdmi_corrected / (np.sqrt(Sdmi_corrected) + self.epsilon)

                layer.Vdsigma = self.beta1 * layer.Vdsigma + (1 - self.beta1) * layer.dsigma
                layer.Sdsigma = self.beta2 * layer.Sdsigma + (1 - self.beta2) * layer.dsigma ** 2
                Vdsigma_corrected = layer.Vdsigma / (1 - self.beta1 ** self.iteration)
                Sdsigma_corrected = layer.Sdsigma / (1 - self.beta2 ** self.iteration)
                layer.sigma = layer.sigma - self.lr * Vdsigma_corrected / (np.sqrt(Sdsigma_corrected) + self.epsilon)



