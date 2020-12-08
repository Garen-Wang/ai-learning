import numpy as np
from pathlib import Path

from step6.HelperClass2.EnumDef import *

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

    def InitializeWeights(self, folder, create_new):
        self.folder = folder
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()

        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def __LoadExistingParameters(self):
        file_name = '{}/{}.npz'.format(self.folder, self.initial_value_filename)
        path = Path(file_name)
        if path.exists():
            self.__LoadInitialValues()
        else:
            self.__CreateNew()

    def __CreateNew(self):
        self.W, self.B = self.InitialParameters()
        self.__SaveInitialValues()

    def __LoadInitialValues(self):
        file_name = "{}/{}.npz".format(self.folder, self.initial_value_filename)
        data = np.load(file_name)
        self.W = data['weights']
        self.B = data['bias']

    def __SaveInitialValues(self):
        file_name = "{}/{}.npz".format(self.folder, self.initial_value_filename)
        np.savez(file_name, weights=self.W, bias=self.B)

    def LoadResultValues(self, folder, name):
        file_name = '{}/{}.npz'.format(folder, name)
        data = np.load(file_name)
        self.W = data['weights']
        self.B = data['bias']

    def SaveResultValues(self, folder, name):
        file_name = '{}/{}.npz'.format(folder, name)
        np.savez(file_name, weights=self.W, bias=self.B)


    @staticmethod
    def __InitialParameters(num_input, num_output, init_method):
        if init_method == InitialMethod.Zero:
            W = np.zeros((num_input, num_output))
        elif init_method == InitialMethod.Normal:
            W = np.random.normal((num_input, num_output))
        elif init_method == InitialMethod.Xavier:
            W = np.random.uniform(-np.sqrt(6 / (num_input + num_output)), np.sqrt(6 / (num_input + num_output)), size=(num_input, num_output))

        B = np.zeros((1, num_output))
        return W, B

