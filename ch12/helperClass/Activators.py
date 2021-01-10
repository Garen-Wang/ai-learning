import numpy as np


class Activator(object):
    def forward(self, z):
        pass

    def backward(self, z, a, delta):
        # return (dz, da)
        # dz: delta of previous layer(going through chain rule)
        # da: derivative of current layer(nothing of previous layer)
        pass


class Identity(Activator):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta, a


class Sigmoid(Activator):
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z, a, delta):
        da = a * (1 - a)
        dz = delta * da
        return dz, da


class Tanh(Activator):
    def forward(self, z):
        return 2.0 / (1.0 + np.exp(-2 * z)) - 1.0

    def backward(self, z, a, delta):
        da = (1 + a) * (1 - a)
        dz = delta * da
        return dz, da


class Relu(Activator):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.zeros(z.shape())
        da[z>0] = 1
        dz = delta * da
        return dz, da

