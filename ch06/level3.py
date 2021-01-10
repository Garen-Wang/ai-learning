import numpy as np
import matplotlib.pyplot as plt
import math

from helper3.NeuralNet import *


class LogicDataReader(DataReader):
    def __init__(self):
        pass
    def read_AND_data(self):
        self.xRaw = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        self.yRaw = np.array([0, 0, 0, 1]).reshape(4, 1)
        self.xTrain = self.xRaw
        self.yTrain = self.yRaw
        self.num_train = 4

    def read_OR_data(self):
        self.xRaw = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        self.yRaw = np.array([0, 1, 1, 1]).reshape(4, 1)
        self.xTrain = self.xRaw
        self.yTrain = self.yRaw
        self.num_train = 4

    def read_NAND_data(self):
        self.xRaw = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        self.yRaw = np.array([1, 1, 1, 0]).reshape(4, 1)
        self.xTrain = self.xRaw
        self.yTrain = self.yRaw
        self.num_train = 4

    def read_NOR_data(self):
        self.xRaw = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        self.yRaw = np.array([1, 0, 0, 0]).reshape(4, 1)
        self.xTrain = self.xRaw
        self.yTrain = self.yRaw
        self.num_train = 4

    def read_NOT_data(self):
        self.xRaw = np.array([0, 1]).reshape(2, 1)
        self.yRaw = np.array([1, 0]).reshape(2, 1)
        self.xTrain = self.xRaw
        self.yTrain = self.yRaw
        self.num_train = 2

def showResult(reader, neural):
    plt.grid()
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    X, Y = reader.getWholeTrainSamples()
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], marker='^')
    else:
        plt.scatter(X[:, 0], np.zeros(X.shape), marker='^')

    if neural.W.shape[0] == 2:
        w = -neural.W[0, 0] / neural.W[1, 0]
        b = -neural.B[0, 0] / neural.W[1, 0]
    else:
        w = neural.W[0, 0]
        b = neural.B[0, 0]
    print("w = {}, b = {}".format(w, b))
    x = np.array([-0.1, 1.1])
    y = w * x + b
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    reader = LogicDataReader()
    # reader.read_NOT_data()
    # reader.read_AND_data()
    # reader.read_OR_data()
    params = HyperParameters(2, 1, eta=0.1, max_epoch=10000, batch_size=-1, eps=1e-6, net_type=NetType.BinaryClassifier)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=1)

    showResult(reader, neural)
