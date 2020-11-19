import numpy as npy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper2.NeuralNet import *


file_name = '../../ai-data/Data/ch05.npz'


def showResult(neural, reader):
    X, Y = reader.getWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], Y)
    len1 = 50
    len2 = 50
    p = npy.linspace(0, 1, len1)
    q = npy.linspace(0, 1, len2)
    P, Q = npy.meshgrid(p, q)
    R = npy.hstack((P.ravel().reshape(2500, 1), Q.ravel().reshape(2500, 1)))
    Z = neural.forwardBatch(R)
    Z = Z.reshape(50, 50)
    ax.plot_surface(P, Q, Z, cmap="rainbow")
    plt.show()


if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    reader.normalizeX()
    params = HyperParameters(2, 1, eta=0.1, max_epoch=10, batch_size=1, eps=1e-5)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=0.1)
    x1 = 15
    x2 = 93
    x = npy.array([x1, x2]).reshape(1, 2)
    print("z = ", neural.forwardBatch(x))

    showResult(neural, reader)
