import numpy as np


class Classifier(object):
    def forward(self, z):
        pass


class Logistic(Classifier):
    def forward(self, z):
        a = 1 / (1 + np.exp(-z))
        return a


class Softmax(Classifier):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a
