import numpy as np

class LeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * x for x in z])

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else self.alpha for x in a])
        dz = 0
        return da, dz