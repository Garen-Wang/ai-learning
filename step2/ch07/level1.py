import numpy as npy
import matplotlib.pyplot as plt

from helper4.NeuralNet import *

file_name = '../../ai-data/Data/ch07.npz'

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    reader.normalizeX()
    reader.toOneHot(3, base=1)
    params = HyperParameters(2, 3, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=1)

    x = npy.array([5, 1, 7, 6, 5, 6, 2, 7]).reshape(4, 2)
    x_norm = reader.normalizePredicateData(x)
    z_norm = neural.forwardBatch(x_norm)
    ans = npy.argmax(z_norm, axis=1) + 1
    print("output = ", z_norm)
    print('r = ', ans)
