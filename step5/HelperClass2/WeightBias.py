import numpy as np
from pathlib import Path

from step5.HelperClass2.EnumDef import *


class WeightBias(object):
    def __init__(self, num_input, num_output, init_method, eta):
        self.num_input = num_input
        self.num_output = num_output
        self.init_method = init_method
        self.eta = eta
        self.initial_value_filename = 'w_{0}_{1}_{2}_init'.format(self.num_input, self.num_output,
                                                                  self.init_method.name)

    def __CreateNew(self):
        self.W, self.B = self.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.__SaveInitialValue()

    def __LoadExistingParameters(self):
        file_name = "{0}/{1}.npz".format(self.folder, self.initial_value_filename)
        file = Path(file_name)
        if file.exists():
            self.__LoadInitialValue()
        else:
            self.__CreateNew()

    def InitializeWeights(self, folder, create_new):
        self.folder = folder
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()

        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def update(self):
        self.W -= self.eta * self.dW
        self.B -= self.eta * self.dB

    def __SaveInitialValue(self):
        file_name = "{0}/{1}.npz".format(self.folder, self.initial_value_filename)
        np.savez(file_name, weights=self.W, bias=self.B)

    def __LoadInitialValue(self):
        file_name = "{0}/{1}.npz".format(self.folder, self.initial_value_filename)
        data = np.load(file_name)
        self.W = data['weights']
        self.B = data['bias']

    def SaveResultValue(self, folder, name):
        file_name = "{0}/{1}.npz".format(folder, name)
        np.savez(file_name, weights=self.W, bias=self.B)

    def LoadResultValue(self, folder, name):
        file_name = "{0}/{1}.npz".format(folder, name)
        data = np.load(file_name)
        self.W = data['weights']
        self.B = data['bias']

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.Zero:
            W = np.zeros((num_input, num_output))
        elif method == InitialMethod.Normal:
            W = np.random.normal(size=(num_input, num_output))
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2 / num_output), size=(num_input, num_output))
        elif method == InitialMethod.Xavier:
            W = np.random.uniform(-np.sqrt(6 / (num_input + num_output)),
                                  np.sqrt(6 / (num_input + num_output)),
                                  size=(num_input, num_output))
        B = np.zeros((1, num_output))
        return W, B
