import numpy as np

class Softplus(object):
    def forward(self, z):
        return np.log(1 + np.exp(z))

    def backward(self, z, a, delta):
        p = np.exp(z)
        da = p / (1 + p)
        dz = np.multiply(delta, da)
        return da, dz