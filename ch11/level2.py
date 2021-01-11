import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ch11.HelperClass2.NeuralNet import *
from ch11.HelperClass2.Visualizer import *
from ch11.HelperClass2.DataReader import *

train_file_name = '../Data/ch11.train.npz'
test_file_name = '../Data/ch11.test.npz'

def Show3D(net, reader):
    X, Y = reader.GetTestSet()
    A = net.inference(X)

    colors = ['b', 'r', 'g']
    shapes = ['o', 'x', 's']
    fig = plt.figure(figsize=(6, 6))
    axes = Axes3D(fig)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i, j] == 1:
                axes.scatter(net.Z1[i, 0], net.Z1[i, 1], net.Z1[i, 2], color=colors[j], marker=shapes[j])
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    axes = Axes3D(fig)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i, j] == 1:
                axes.scatter(net.A1[i, 0], net.A1[i, 1], net.A1[i, 2], color=colors[j], marker=shapes[j])
    plt.show()


if __name__ == '__main__':
    reader = DataReader(train_file_name, test_file_name)
    reader.ReadData()
    reader.NormalizeY(NetType.MultipleClassifier, base=1)
    reader.NormalizeX()
    reader.GenerateValidationSet()

    num_input = reader.num_feature
    num_hidden = 3
    num_output = reader.num_category
    eta = 0.1
    batch_size = 10
    max_epoch = 5000
    eps = 1e-1

    params = HyperParameters(num_input, num_hidden, num_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet(params, "Bank_233_2")

    net.train(reader, 100, True)
    net.ShowTrainingHistory()

    Show3D(net, reader)
