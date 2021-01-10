from ch12.helperClass.EnumDef import InitMethod
import numpy as np


class WeightsBias(object):
    def __init__(self, num_input, num_output, init_method, eta):
        self.num_input = num_input
        self.num_output = num_output
        self.init_method = init_method
        self.eta = eta

        # initialization after definition
        self.W, self.B = WeightsBias.initial_parameters(num_input, num_output, init_method)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def update(self):
        self.W -= self.eta * self.dW
        self.B -= self.eta * self.dB

    @staticmethod
    def initial_parameters(num_input, num_output, init_method):
        if init_method == InitMethod.Zero:
            W = np.zeros((num_input, num_output))
        if init_method == InitMethod.Normal:
            W = np.random.normal(size=(num_input, num_output))
        if init_method == InitMethod.Xavier:
            W = np.random.uniform(-np.sqrt(6 / (num_input + num_output)), np.sqrt(6 / (num_input + num_output)), size=(num_input, num_output))

        B = np.zeros((1, num_output))
        return W, B
