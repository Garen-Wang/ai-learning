import numpy as np

class Relu(object):
    def forward(self, z):
        self.mem = np.zeros(z.shape)
        self.mem[z > 0] = 1
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else 0 for x in a])
        dz = self.mem * delta
        return da, dz
