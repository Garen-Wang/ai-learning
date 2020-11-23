import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

from Assignment1.HelperClass.NeuralNet import *

file_name = 'Dataset/mlm.csv'

def showResult(reader, neural):
    X, Y = reader.getWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    p = np.linspace(0, 1)
    q = np.linspace(0, 1)
    P, Q = np.meshgrid(p, q)
    R = np.hstack((P.ravel().reshape(2500, 1), Q.ravel().reshape(2500, 1)))
    Z = neural.inference(R)
    Z = Z.reshape(50, 50)
    ax.plot_surface(P, Q, Z, cmap="rainbow")
    plt.show()

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    print(reader.xRaw)
    reader.normalizeX()
    reader.normalizeY()
    print(reader.xTrain)
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=5, eps=1e-4)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=0.1)

    showResult(reader, neural)