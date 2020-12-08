import numpy as np

class Step(object):
    def __init__(self, thresold):
        self.thresold = thresold

    def forward(self, z):
        return np.array([1 if x > self.thresold else 0 for x in z])

    def backward(self, z, a, delta):
        da = np.zeros(a.shape)
        dz = da
        return da, dz