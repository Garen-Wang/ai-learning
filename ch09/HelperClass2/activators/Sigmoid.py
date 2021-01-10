import numpy as np
import matplotlib.pyplot as plt

class Sigmoid(object):
    def forward(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1 - a)
        dz = np.multiply(delta, da)
        return da, dz
