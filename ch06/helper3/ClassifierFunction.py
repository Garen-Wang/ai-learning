import numpy as npy

class Logistic(object):
    def forward(self, z):
        return 1.0 / (1.0 + npy.exp(-z))

