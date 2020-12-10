import numpy as np


class Activator(object):
    def forward(self, z):
        pass

    def backward(self, z, a, delta):
        pass


class Identity(Activator):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta, a


class Sigmoid(Activator):
    def forward(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1 - a)
        dz = np.multiply(delta, da)
        return dz, da


class Tanh(Activator):
    def forward(self, z):
        a = 2 / (1 + np.exp(-2 * z)) - 1
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return dz, da


class Relu(Activator):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.zeros(z.shape)
        da[z > 0] = 1
        dz = np.multiply(delta, da)
        return dz, da
