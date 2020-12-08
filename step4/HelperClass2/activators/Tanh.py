import numpy as np

class Tanh(object):
    def forward(self, z):
        a = 2 / (1 + np.exp(-2 * z)) - 1
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return da, dz
