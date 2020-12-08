import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from step4.HelperClass2.NeuralNet import *
from step4.HelperClass2.DataReader_2_0 import *

train_file_name = '../ai-data/Data/ch09.train.npz'
test_file_name = '../ai-data/Data/ch09.test.npz'


def ShowResult(net, reader, title):
    X, Y = reader.XTrain, reader.YTrain
    plt.plot(X[:, 0], Y[:, 0], '.', c='b')
    TX = np.linspace(0, 1, 100).reshape(100, 1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    reader = DataReader_2_0(train_file_name, test_file_name)
    reader.ReadData()
    reader.GetValidationSet()

    num_input, num_hidden, num_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 10000
    eps = 1e-3

    params = HyperParameters(num_input, num_hidden, num_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet(params, 'complex131')

    net.train(reader, 50, True)
    net.ShowTrainingHistory()
    ShowResult(net, reader, params.toString())