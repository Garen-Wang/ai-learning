import numpy as npy
from pathlib import Path
import matplotlib.pyplot as plt
import math

from helper4.NeuralNet import *
from helper4.visualizer import *
file_name = '../ai-data/Data/ch07.npz'

def showResult(neural, reader, X, output):
    fig = plt.figure(figsize=(6, 6))

    drawThreeCategoryPoints(reader.xTrain[:, 0], reader.xTrain[:, 1], reader.yTrain[:], show=False)

    b13 = (neural.B[0, 0] - neural.B[0, 2]) / (neural.W[1, 2] - neural.W[1, 0])
    w13 = (neural.W[0, 0] - neural.W[0, 2]) / (neural.W[1, 2] - neural.W[1, 0])

    b23 = (neural.B[0, 2] - neural.B[0, 1]) / (neural.W[1, 1] - neural.W[1, 2])
    w23 = (neural.W[0, 2] - neural.W[0, 1]) / (neural.W[1, 1] - neural.W[1, 2])

    b12 = (neural.B[0, 1] - neural.B[0, 0]) / (neural.W[1, 0] - neural.W[1, 1])
    w12 = (neural.W[0, 1] - neural.W[0, 0]) / (neural.W[1, 0] - neural.W[1, 1])

    x = npy.linspace(0, 1, 2)
    y = w13 * x + b13
    p13, = plt.plot(x, y, c='r')

    x = npy.linspace(0, 1, 2)
    y = w23 * x + b23
    p23, = plt.plot(x, y, c='b')

    x = npy.linspace(0, 1, 2)
    y = w12 * x + b12
    p12, = plt.plot(x, y, c='g')

    plt.legend([p13, p23, p12], ['13', '23', '12'])
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    drawThreeCategoryPoints(X[:, 0], X[:, 1], output[:], show=True, isPredicate=True)

if __name__ == '__main__' :
    reader = DataReader(file_name)
    reader.readData()
    reader.normalizeX()
    reader.toOneHot(3, 1)

    params = HyperParameters(2, 3, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=1)

    x = npy.array([5, 1, 7, 6, 5, 6, 2, 7]).reshape(4, 2)
    x_norm = reader.normalizePredicateData(x)
    print("x_norm = ", x_norm)
    output = neural.forwardBatch(x_norm)
    print(output)

    showResult(neural, reader, x_norm, output)
