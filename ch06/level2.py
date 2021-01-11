import numpy as npy
import matplotlib.pyplot as plt
from helper3.NeuralNet import *

file_name = '../Data/ch06.npz'

def draw_split_line(neural):
    w_real = -neural.W[0, 0] / neural.W[1, 0]
    b_real = -neural.B[0, 0] / neural.W[1, 0]
    print("w_real = {}, b_real = {}".format(w_real, b_real))
    x = npy.linspace(0, 1, 10)
    y = w_real * x + b_real
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.plot(x, y)
    plt.show()


def draw_source_data(reader):
    fig = plt.figure(figsize=(6.5, 6.5))
    X, Y = reader.getWholeTrainSamples()
    for i in range(200):
        if Y[i, 0] == 1:
            plt.scatter(X[i, 0], X[i, 1], marker='x', c='g')
        else:
            plt.scatter(X[i, 0], X[i, 1], marker='o', c='r')
    pass

def draw_predicate_data(neural, reader):
    x = npy.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3, 2)
    z = neural.forwardBatch(x)
    for i in range(3):
        if z[i, 0] > 0.5:
            plt.scatter(x[i, 0], x[i, 1], marker='^', c='g')
        else:
            plt.scatter(x[i, 0], x[i, 1], marker='^', c='r')

    pass

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    params = HyperParameters(2, 1, eta=0.1, max_epoch=10000, batch_size=10, eps=1e-4, net_type=NetType.BinaryClassifier)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=1)
    draw_source_data(reader)
    draw_predicate_data(neural, reader)
    draw_split_line(neural)

