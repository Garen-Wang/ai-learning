import numpy as np
import matplotlib.pyplot as plt

from ch10.HelperClass2.NeuralNet import *

# 0 xor 0 = 0
# 0 xor 1 = 1
# 1 xor 0 = 1
# 1 xor 1 = 0

class XOR_DataReader(DataReader_2_0):
    def __init__(self):
        pass

    def ReadData(self):
        self.XTrain = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        self.YTrain = np.array([0, 1, 1, 0]).reshape(4, 1)

        self.num_train = self.XTrain.shape[0]
        self.num_feature = self.XTrain.shape[1]
        self.num_category = 1  # if wrong, debug here

        self.XTest = self.XTrain
        self.YTest = self.YTrain
        self.XDev = self.XTrain
        self.YDev = self.YTrain
        self.num_test = self.num_train
        # self.num_validation = self.num_train

def Test(reader, net):
    print('testing...')
    X, Y = reader.GetTestSet()
    A2 = net.inference(X)
    print("A2 = ", A2)
    diff = np.abs(A2 - Y)
    result = np.where(diff < 1e-2, True, False)
    if result.sum() == reader.num_test:
        return True
    else:
        return False

if __name__ == '__main__':
    reader = XOR_DataReader()
    reader.ReadData()

    n_input, n_hidden, n_output = 2, 2, 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 1e-3

    params = HyperParameters(n_input, n_hidden, n_output,
                             eta, max_epoch, batch_size, eps,
                             NetType.BinaryClassifier,
                             InitialMethod.Xavier)
    net = NeuralNet(params, "XOR_221")

    net.train(reader, 100, True)
    net.ShowTrainingHistory()

    print(Test(reader, net))