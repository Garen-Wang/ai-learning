import numpy as npy

from helper2.NeuralNet import *

file_name = '../ai-data/Data/ch05.npz'

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    reader.normalizeX()
    params = HyperParameters(2, 1, eta=0.1, max_epoch=10, batch_size=1, eps=1e-2)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=0.1)
    x1 = 15
    x2 = 93
    x = npy.array([x1, x2]).reshape(1, 2)
    print(neural.forwardBatch(x))