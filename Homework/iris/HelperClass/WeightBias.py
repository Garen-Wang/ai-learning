import numpy as np
from pathlib import Path

from Homework.iris.HelperClass.EnumDef import *


class WeightBias(object):
    def __init__(self, num_input, num_output, init_method, eta):
        self.num_input = num_input
        self.num_output = num_output
        self.init_method = init_method
        self.eta = eta
        self.initial_value_filename = "w_{}_{}_{}_init".format(num_input, num_output, init_method)

    def update(self):
        self.W -= self.eta * self.dW
        self.B -= self.eta * self.dB

    def InitializeWeights(self):
        self.W, self.B = self.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    @staticmethod
    def InitialParameters(num_input, num_output, init_method):
        if init_method == InitialMethod.Zero:
            W = np.zeros((num_input, num_output))
        elif init_method == InitialMethod.Normal:
            W = np.random.normal((num_input, num_output))
        elif init_method == InitialMethod.Xavier:
            W = np.random.uniform(-np.sqrt(6 / (num_input + num_output)),
                                  np.sqrt(6 / (num_input + num_output)),
                                  size=(num_input, num_output))
        else:
            W = None

        B = np.zeros((1, num_output))
        return W, B
