import numpy as npy

class Logistic(object):
    def forward(self, z):
        return 1.0 / (1.0 + npy.exp(-z))


class Softmax(object):
    def forward(self, z):
        shift_z = z - npy.max(z, axis=1, keepdims=True)
        exp_z = npy.exp(shift_z)
        a = exp_z / npy.sum(exp_z, axis=1, keepdims=True)
        return a