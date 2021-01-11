import numpy as npy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper2.NeuralNet import *

file_name = "../Data/ch05.npz"


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
    R = npy.hstack((P.ravel().reshape(len1 * len2, 1), Q.ravel().reshape(len1 * len2, 1)))
    Z = neural.forwardBatch(R)
    Z = Z.reshape(len1, len2)
    ax.plot_surface(P, Q, Z, cmap="rainbow")
    plt.show()


def denormalize(neural, reader):
    W_real = npy.zeros_like(neural.W)
    for i in range(W_real.shape[0]):
        W_real[i, 0] = neural.W[i, 0] / reader.xNorm[i, 1]
    B_real = neural.B - W_real[0, 0] * reader.xNorm[0, 0] - W_real[1, 0] * reader.xNorm[1, 0]
    return W_real, B_real


if __name__ == "__main__":
    reader = DataReader(file_name)
    reader.readData()
    reader.normalizeX()
    reader.normalizeY()
    params = HyperParameters(2, 1, eta=0.01, max_epoch=200, batch_size=10, eps=1e-5)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=0.1)
    x1 = 15
    x2 = 93
    x = npy.array([x1, x2]).reshape(1, 2)
    x_new = reader.normalizePredicateData(x)
    Z = neural.forwardBatch(x_new)
    print('Z = ', Z)
    Z_real = Z * reader.yNorm[0, 1] + reader.yNorm[0, 0]
    print('Z_real = ', Z_real)
