import numpy as np

class Elu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * (np.exp(z) - 1) for x in z])

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else self.alpha * np.exp(x) for x in z])
        dz = np.multiply(delta, da)
        return da, dz